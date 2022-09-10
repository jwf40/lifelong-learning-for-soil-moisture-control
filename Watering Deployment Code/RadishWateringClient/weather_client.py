import requests
import pandas as pd
import torch
from torch import optim
import json 
import datetime
import pickle
import time
from models.lstm_ff import LSTM_FF
from data_loader import WeatherData
from ewc_dataloader import EWCData
from utils import EWC, ewc_train, normal_train, test

class RequestHandler():
    """
    Connects to the ESP32 and collects the data, saving it to the given dataset.
    Expected data format: string with CSV
    """
    def __init__(self,url: str, endpoint: str, test_endpoints: list, save_path: str, load_path: str):
        self.url = url
        self.endpoint = url + endpoint
        self.test_endpoints = [url + ep for ep in test_endpoints]
        self.data_save_path = save_path
        self.data = pd.read_csv(load_path)

    def sensors_working(self):
        """
        Queries the end point for each sensor
        If response isn't okay, throw assertion
        """
        for ep in self.test_endpoints:
            response = requests.get(ep)
            print(response.text)
            if response.status_code != 200:
                return False
            time.sleep(1)
        return True
            

    def get_new_data(self,url=None) -> str:
        """
        Request data from Arduino
        """
        url = self.url if url==None else url
        response = requests.get(url)
        assert int(response.status_code) == 200, "Error Receiving data from server!"
        return response.text


    def append_to_data(self, data: str):
        """
        Append data to dataframe
        """
        data = data.split(',')
        self.data.loc[len(self.data)] = data
        return self.data
    
    def save_backup(self):
        """
        Saves current data to a new file
        """
        time = datetime.datetime.now()
        strtime = time.strftime('%H-%M-%S_%Y-%m-%d')
        backup_path = self.data_save_path[:-4] + '_' + strtime + '_'+ self.data_save_path[-4:]
        self.data.to_csv(backup_path,index=False)
    
    def save(self):
        """
        Save data to standard file
        """
        self.data.to_csv(self.data_save_path,index=False)    


class ModelHandler():
    def __init__(self,model_config_path: str, ewc_config_path: str, data_config_path: str):
        #Open Configs
        with open(model_config_path, 'r') as js:
            self.model_config = json.load(js)
        with open(ewc_config_path, 'r') as js:
            self.ewc_config = json.load(js)
        with open(data_config_path, 'r') as js:
            self.data_config = json.load(js)

        self.data_root = self.data_config['data_dir']
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        #LSTM features
        self.in_features = self.model_config['in_features']# 5
        self.hidden_size = self.model_config['hidden_size']#128
        self.num_classes = self.model_config['num_classes']#2
        self.sequence_len = self.model_config['sequence_len']#7
        
        #Save paths
        self.loss_history = self.model_config['loss_history']
        self.ewc_acc_history = self.model_config['ewc_acc_history']
        self.lstm_acc_history = self.model_config['lstm_acc_history']
        self.prediction_history = self.model_config['pred_history']
        
        #LSTM models        
        self.lstm = self.load_model(self.model_config['lstm_dict'])
        self.ewc_lstm = self.load_model(self.model_config['ewc_dict'])
        
        #EWC train params
        self.lr = self.ewc_config['lr']
        self.importance = self.ewc_config['importance']
        self.sample_size = self.ewc_config['sample_size']

        #Get DataLoaders
        self.precip_loader = torch.utils.data.DataLoader(WeatherData(self.data_config['old_data_dir'], 'daily_precip', self.sequence_len),batch_size=128,num_workers=4, shuffle=False)
        self.ewc_dataloader = torch.utils.data.DataLoader(EWCData(data_config_path, 'ewc_sm', self.sequence_len), batch_size=1, num_workers=2, shuffle=False)
        self.prediction_dataloader = torch.utils.data.DataLoader(EWCData(data_config_path, 'ewc_sm', self.sequence_len, pred=True), batch_size=1, num_workers=2)
        self.record_dataloader = torch.utils.data.DataLoader(EWCData(data_config_path, 'ewc_sm', self.sequence_len, pred=False, is_recording=True), batch_size=1, num_workers=2)

    

    def load_model(self, state_dict):
        """
        Load state dictionary for LSTM model, load to device
        """
        model = LSTM_FF(self.in_features, self.hidden_size, self.num_classes, self.sequence_len,batch_first=True, bidirectional=False)
        model.load_state_dict(torch.load(state_dict))
        model = model.to(self.device)
        return model

    def predict(self):
        """
        Get and print prediction from both models
        """
        inputs,_ = next(iter(self.prediction_dataloader))
        inputs = inputs.to(self.device)
        lstm_out = self.lstm(inputs)
        ewc_lstm_out = self.ewc_lstm(inputs)
        lstm_pred = torch.max(lstm_out.data,1)
        ewc_lstm_pred = torch.max(ewc_lstm_out.data,1)
        print(lstm_pred, ewc_lstm_pred)
        #return (lstm_pred, ewc_lstm_pred)

    def record_performance(self):
        """
        Get prediction from both models, save the responses to file and print them
        """
        inputs,target = next(iter(self.record_dataloader))
        inputs = inputs.to(self.device)
        lstm_out = self.lstm(inputs)
        ewc_lstm_out = self.ewc_lstm(inputs)
        
        lstm_val,lstm_pred = torch.max(lstm_out.data,1)
        ewc_val,ewc_lstm_pred = torch.max(ewc_lstm_out.data,1)
        lstm_pred = lstm_pred.item()
        ewc_lstm_pred = ewc_lstm_pred.item()

        with open(self.prediction_history, 'rb') as f:
            try:
                preds = pickle.load(f)
            except:
                preds = []
            print(preds)
        with open(self.prediction_history, 'wb') as f:
            preds.append(([ewc_val,ewc_lstm_pred], [lstm_val,lstm_pred]))
            pickle.dump(preds, f)
            

        if target[0] == -1: #if model did well, set target to prediction
            target[0]=ewc_lstm_pred
        if target[1] ==-1:
            target[1]=lstm_pred
        print("Predictions: ", ewc_lstm_pred, lstm_pred)
        print("TARGETS: ", target)
        ewc_acc = 1 if ewc_lstm_pred == target[0] else 0
        lstm_acc = 1 if lstm_pred == target[1] else 0
        with open(self.ewc_acc_history, 'rb') as f:
            try:
                ewc_hist = pickle.load(f)
                print("SUCCESS")
            except:
                ewc_hist = []
            print(ewc_hist)
        with open(self.ewc_acc_history, 'wb') as f:
            ewc_hist.append(ewc_acc)
            pickle.dump(ewc_hist, f)
        
        with open(self.lstm_acc_history, 'rb') as f:
            try:
                lstm_hist = pickle.load(f)
                print("SUCCESS")
            except:
                lstm_hist = []
            print(lstm_hist)
        with open(self.lstm_acc_history, 'wb') as f:
            lstm_hist.append(lstm_acc)
            pickle.dump(lstm_hist, f)


    def ewc(self):
        """
        Train EWC Network
        """
        optimiser = optim.Adam(params=self.ewc_lstm.parameters(), lr=self.lr)
        old_tasks = self.precip_loader.dataset.get_sample(self.sample_size)
        loss = ewc_train(self.ewc_lstm, optimiser, self.ewc_dataloader, EWC(self.ewc_lstm,old_tasks),self.importance,loss_weights=torch.Tensor([1,1]).to(self.device))
        print(loss)
        with open(self.loss_history, 'rb') as f:
            try:
                loss_hist = pickle.load(f)
                print("SUCCESS")
            except:
                loss_hist = []
            print(loss_hist)
        with open(self.loss_history, 'wb') as f:
            loss_hist.append(loss)
            pickle.dump(loss_hist, f)
        self.save_updated_ewc_net()

    def save_updated_ewc_net(self):
        time = datetime.datetime.now()
        strtime = time.strftime('%H-%M-%S_%Y-%m-%d')
        backup_path = self.model_config['ewc_dict']+'_'+strtime
        torch.save(self.ewc_lstm.state_dict(), backup_path)
        torch.save(self.ewc_lstm.state_dict(), self.model_config['ewc_dict'])


class Main():
    def __init__(self, model_config_path:str, ewc_config_path:str,data_config_path:str):
        self.model_handler = ModelHandler(model_config_path, ewc_config_path, data_config_path)
        self.request_handler = RequestHandler(
            'http://192.168.1.116/', 'get_data', ['relay1', 'relay2', 'dht11', 'sm', 'bmp'],'data/test.csv', 'data/test.csv')#data/cambridge_sensor_data.csv','data/cambridge_sensor_data.csv')
        self.data = None

    def main(self):
        if self.update_data():
            self.model_handler.ewc()
            self.model_handler.record_performance()
            self.model_handler.predict()
                

    def update_data(self):
        if self.request_handler.sensors_working():
            self.request_handler.save_backup()
            new_data = self.request_handler.get_new_data()     
            self.data =  self.request_handler.append_to_data(new_data)
            self.request_handler.save()
            return True
        else:
            print("ERROR: SENSORS NOT WORKING")
            return False


if __name__ =='__main__':
    main_ = Main("configs/model_config.json", "configs/ewc_config.json", "configs/data_config.json")
    main_.main()