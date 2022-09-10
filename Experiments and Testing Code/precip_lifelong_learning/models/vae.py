"""
Reimplementation of Vab-al Variational Autencoder, original code:
|J. Choi,  K. M. Yi,  J. Kim,  J. Choo,  B. Kim,  J.-Y. Chang,  Y. Gwon,  and H. J.Chang,
|“Vab-al:   Incorporating  class  imbalance  and  difficulty  with  variationalbayes for active learning,” 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 


def init_weights(net, std=0.1):
    classname = net.__class__.__name__
    if classname.find('Linear') != -1:
        net.weight.data.normal_(0.0, std)


class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', batch_norm=False):
        super(FCLayer, self).__init__()
        self.use_batch_norm = batch_norm

        self.fc = nn.Linear(in_channels, out_channels)
        self.bc = nn.BatchNorm1d(out_channels)

        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()

    def forward(self, x):
        out = self.bc(self.fc(x)) if self.use_batch_norm else self.fc(x)
        out = self.activation(self.fc(x))
        return out


class VAE(nn.Module):
    def __init__(self, input_size, fc_size, num_classes, class_latent_size, lambda_val):
        super(VAE, self).__init__()

        #Variables
        self.num_classes = num_classes
        self.class_latent_size = class_latent_size
        self.lambda_val = lambda_val
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
       
        # Encoder Architecture
        # # First Layer Input = Encode size * number of preprocess fully connected sections
        self.encoder_fc11 = FCLayer(input_size, fc_size)

        self.encoder_fc12 = FCLayer(fc_size, fc_size)
        self.encoder_fc13 = FCLayer(fc_size, fc_size)

        # Layers to output Mean and Log Variance Latent Embeddings
        self.encoder_mean_fc = nn.Linear(fc_size, class_latent_size*num_classes)
        self.encoder_logvar_fc = nn.Linear(fc_size, class_latent_size*num_classes)

        # Decoder Architecture
        # # First Layer Input = Latent Class Size * number of classes
        self.decoder_fc11 = FCLayer(class_latent_size*num_classes, fc_size)

        self.decoder_fc12 = FCLayer(fc_size, fc_size)
        self.decoder_fc13 = FCLayer(fc_size, fc_size)
        
        # Output Layer shape = Encoder Input Layer Shape
        self.decoder_fc2 = FCLayer(fc_size, input_size,activation='sigmoid')

        #TODO CHECK THIS!        
        self.apply(init_weights)
        self.sigmoid = nn.Sigmoid()

    def decode(self, z):
        out1 = self.decoder_fc11(z)
        out2 = self.decoder_fc12(out1)
        out3 = self.decoder_fc13(out2)

        return self.decoder_fc2(out3)

    def reparameterize(self, mu, logvar):        
        # std_deviation = exp(0.5 * log(variance)
        std_dev = logvar.mul(0.5).exp_()
        #Reparameterization trick. Allows us to backprop through deterministic nodes by using a specified random epsilon node
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std_dev.size()).normal_()
        else:
            eps = torch.FloatTensor(std_dev.size()).normal_()
        eps = Variable(eps)
        z = mu+std_dev * eps
        return z

    def encode(self, x):
        # Pass through each FC layer
        out1 = self.encoder_fc11(x)
        out2 = self.encoder_fc12(out1)
        out3 = self.encoder_fc13(out2)
        
        # Output Mean and LogVar vectors
        return self.encoder_mean_fc(out3), self.encoder_logvar_fc(out3)

    def get_latent(self, features):
        #x = self.preprocess(features)
        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        return z

    def preprocess(self, features):
        # Global Average Pooling of input features from Classifier Network
        gap_vectors = [F.avg_pool2d(feature, kernel_size=(feature.size(2), feature.size(3))).view(feature.size(0),-1) for feature in features] 
        # for i in range(len(gap_vectors)):
        #     print(gap_vectors[i].shape,self.preprocess_batch_norms[i])
        # Apply Batch Normalisation to Average Pool Vectors
        bn_vectors = [self.preprocess_batch_norms[i](gap_vector) for (i,gap_vector) in enumerate(gap_vectors)]

        # Encode each vector as Fully Connected Linear Layer
        fc_vectors = [self.preprocess_fcs[i](bn_vector) for (i,bn_vector) in enumerate(bn_vectors)]
        # Concatenate input vectors into a single FC Layer
        input_layer = torch.cat(fc_vectors, dim=1)
        # Pass through sigmoid
        return self.sigmoid(input_layer)

    def forward(self, features):
        #x = self.preprocess(features)
        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return features, z, recon_x, mu, logvar

    def loss_fnc (self, recon_x, z, x, mu, logvar, targets):
        # Binary Cross Entropy
        #BCE = F.binary_cross_entropy(recon_x,x,size_average=False)
        BCE = torch.sum((recon_x-x.detach())**2)
        # Kulback-Liebler Divergence TODO CHECK THIS
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        
        # Class
        z_class = torch.sum(z.view(-1, self.num_classes, self.class_latent_size).pow(2), dim=2)
        L_class = self.criterion(-z_class, torch.cuda.LongTensor([targets for _ in range(len(z_class))]))

        return (BCE, KLD, self.lambda_val*L_class)

    def estimate_likelihood(self, features):
        # Pass Features through net into latent space
        x = features#self.preprocess(features)
        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)

        # Learn a latent space where the absence of parts are related to predicted label
        # Min value associated with the predicted class
        z_reshape = z.view(-1, self.num_classes, self.class_latent_size)
        predicted_labels = torch.argmin(torch.sum(z_reshape.pow(2), dim=2), dim=1)
        # Mask class to get likelihood of that class (VAE works by absence of parts)
        likelihood = torch.zeros([x.size(0), self.num_classes], dtype=torch.float64)
        for i in range(self.num_classes):
            # Deep Copy of Clone
            z_masked = z_reshape.clone()
            # Mask This classes latent space
            z_masked[:,i,:] = 0
            # Reshape
            z_masked = z_masked.view(-1, self.num_classes*self.class_latent_size)

            #reconstruct
            recon_x_masked = self.decode(z_masked)

            #Loss Function
            BCE_masked = torch.sum((recon_x_masked-x)**2, dim=1)
            KLD_element_masked = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD_masked = torch.sum(KLD_element_masked, dim=0).mul_(-0.5)
            #KLD_masked = -0.5*torch.mean(1+logvar - mu.pow(2) - logvar.exp())
            #KLD = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp())

            # Likelihood

            likelihood[:, i] = BCE_masked + KLD_masked
        return (likelihood, predicted_labels)

    def predict(self, features):
        z = self.get_latent(features)
        z_reshape = z.view(-1, self.num_classes, self.class_latent_size)
        predicted_labels = torch.argmin(torch.sum(z_reshape.pow(2), dim=2), dim=1)
        return predicted_labels
    
