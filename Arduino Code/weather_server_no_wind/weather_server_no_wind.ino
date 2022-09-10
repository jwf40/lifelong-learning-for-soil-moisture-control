#include <Adafruit_BusIO_Register.h>
#include <Adafruit_I2CDevice.h>
#include <Adafruit_I2CRegister.h>
#include <Adafruit_SPIDevice.h>

#include <Adafruit_Sensor.h>

//brainy-bits.com/post/how-to-use-the-dht11-temperature-and-humidity-sensor-with-an-arduino
#include <dht.h>
//https://github.com/adafruit/Adafruit_BMP280_Library
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>

#include <WiFi.h>
#include <WebServer.h>

/* Put your SSID & Password */
const char* ssid = "WIFI SSID HERE";
const char* password = "WIFI PASSWORD HERE";

WebServer server(80);
bool received_request = false;


uint8_t Relay1pin = 18;
bool Relay1status = LOW;
uint8_t Relay2pin = 19;
bool Relay2status = LOW;


uint8_t dht_pin = 17;//analogue pin
dht DHT;

//uint8_t bmp_pin = 20;
Adafruit_BMP280 bmp;
#define i2caddr 0x76

//anenometer vals - https://www.geeky-gadgets.com/arduino-wind-speed-meter-anemometer-project-30032016/
uint8_t anen_pin = 27;
float analog_to_actual_volt_conversion= .004882814;
float windspd_conv = 2.232694;
uint8_t windspd_interval = 2000; //ms
uint8_t windspd_query_time = 600000;
uint8_t num_windspd_readings = windspd_query_time/windspd_interval;

float min_volt = .4; 
float wind_speedMin = 0; 

float max_volt = 2.0; 
float wind_speedMax = 32;
/*
# Soil Moisture value description
# 0  ~300     dry soil
# 300~700     humid soil
# 700~950     in water*/
//pins 
int sm_pins[4] = {34,35,33,32};
float max_moisture = 1000;
float max_moisture_read = 4095;
float moist_ratio = max_moisture/max_moisture_read;

void setup() {
  Serial.begin(115200);
  delay(100);
  pinMode(Relay1pin, OUTPUT);
  pinMode(Relay2pin, OUTPUT);

  Serial.println("Connecting to ");
  Serial.println(ssid);

  //connect to your local wi-fi network
  WiFi.begin(ssid, password);

  //check wi-fi is connected to wi-fi network
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected..!");
  Serial.print("Got IP: ");  Serial.println(WiFi.localIP());
  //
  if (!bmp.begin(i2caddr)) {
    Serial.println(F("Could not find a valid BMP280 sensor, check wiring!"));
  }

  server.on("/", handle_OnConnect);
  server.on("/getData", handle_getData);
  server.on("/relay1", handle_relay1);  
  server.on("/relay2", handle_relay2);
  server.on("/dht11", handle_dht);
  server.on("/bmp", handle_bmp);
  server.on("/anen", handle_anen);
  server.on("/sm", handle_sm);
  server.onNotFound(handle_NotFound);

  server.begin();
  Serial.println("HTTP server started");

  Serial.println(moist_ratio);
}
void loop() {
  server.handleClient();
//    if (!Relay1status) {
//    digitalWrite(Relay1pin, HIGH);
//    Relay1status = HIGH;
//    
//    digitalWrite(Relay2pin, HIGH);
//    Relay2status = HIGH;
//  }else{
//    digitalWrite(Relay1pin, LOW);
//    Relay1status = LOW;
//    
//    digitalWrite(Relay2pin, LOW);
//    Relay2status = LOW; 
//  }
  
//  Serial.println(Relay1status);
//  DHT.read11(dht_pin);
//  float hum = DHT.humidity;
//  float temp = DHT.temperature + 273.15; //to Kelvin
//  
//  //Pressure Data
//  float pressure = bmp.readPressure();
//
//  float sm0 = get_soil_moisture(sm_pins[0]);
//  float sm1 = get_soil_moisture(sm_pins[1]);
//  float sm2 = get_soil_moisture(sm_pins[2]);
//  float sm3 = get_soil_moisture(sm_pins[3]);
//
//  Serial.println("Hum" + String(hum));
//  Serial.println("Temp" + String(temp));
//  Serial.println("Pressure" + String(pressure));
//  Serial.println("Soil Moisture" + String(sm0)+ String(sm1)+ String(sm2)+ String(sm3));
//  delay(5000);
}

void handle_OnConnect() {
  server.send(200, "text/plain", "Connection Okay");
}

float get_avg_windspd(){
  float total_windspd = 0;
  for(int i =0; i<num_windspd_readings; i++){
    float anen_read = analogRead(anen_pin);//read pin
    float anen_volt = anen_read*analog_to_actual_volt_conversion; //Convert to voltage
    float windspd = 0;
    if(anen_volt<=min_volt){
      windspd = 0;
    }else{
      windspd = ((anen_volt - min_volt)*wind_speedMax / (max_volt - min_volt))*windspd_conv;
      }
    
    total_windspd += windspd;  //get windspd val
    delay(windspd_interval);
  }

  float avg_wind = total_windspd/num_windspd_readings;
  return avg_wind;   
}

float get_soil_moisture(int pin){
  float an_read= analogRead(pin);
  float ratio = max_moisture/max_moisture_read;
  Serial.println("Soil_read"+ String(an_read));
  return an_read*ratio;
  }

//Alpha method of dewpoint
float alpha(float hum, float temp, float a, float b){
    return log(hum/100)+ (a*temp/(b+temp));
}

float calculate_dewpoint(float hum, float temp){
  //constants
  float a = 17.62;
  float b = 243.12;  

  float dp = b*(alpha(hum, temp, a, b)) / (a - alpha(hum, temp, a, b));
  float dp_to_kelvin = dp + 273.15;
  return dp_to_kelvin;
  }


String to_csv(float dat[], int len_arr){
  String csv = "";   
  for(int i=0; i<len_arr; i++){
    csv = csv + "," + String(dat[i]); 
    }
  return csv;   
}   


void handle_getData() {
    //Turn Relays on
  if (!Relay1status) {
    digitalWrite(Relay1pin, HIGH);
    Relay1status = HIGH;
    
    digitalWrite(Relay2pin, HIGH);
    Relay2status = HIGH;
  } 
  //Wait 1 sec to ensure elec flow
  delay(1000);

  //DHT Data
  DHT.read11(dht_pin);
  float hum = DHT.humidity;
  float temp = DHT.temperature + 273.15; //to Kelvin
  
  //Pressure Data
  float pressure = bmp.readPressure();

  //Calculate dew point
  float dew_point = calculate_dewpoint(hum, temp);

  //TODO Get anenometer data
  //float windspd = get_avg_windspd();
  float windspd_placeholder = -20;
  //TODO Get Soil Moisture Data
  float sm0 = get_soil_moisture(sm_pins[0]);
  float sm1 = get_soil_moisture(sm_pins[1]);
  float sm2 = get_soil_moisture(sm_pins[2]);
  float sm3 = get_soil_moisture(sm_pins[3]);
  int len_arr = 9;
  float vals[len_arr] = {hum, temp, dew_point, pressure, windspd_placeholder, sm0, sm1, sm2, sm3};
  String csv_data = to_csv(vals, len_arr);
  server.send(200, "text/plain", csv_data);

  //Turn Relays off
  digitalWrite(Relay1pin, LOW);
  Relay1status = LOW;
  digitalWrite(Relay2pin, LOW);
  Relay2status = LOW;  
}

void handle_relay1() {
  if (!Relay1status) {
    digitalWrite(Relay1pin, HIGH);
    Relay1status = HIGH;
  }else{
      digitalWrite(Relay1pin, LOW);
    Relay1status = LOW;
  }
  server.send(200, "text/plain", "Relay 1: " + String(Relay1status));
}

void handle_relay2() {
  if (!Relay2status) {
    digitalWrite(Relay2pin, HIGH);
    Relay2status = HIGH;
  }else{
      digitalWrite(Relay2pin, LOW);
    Relay2status = LOW;
  }
  server.send(200, "text/plain", "Relay 2: " + String(Relay2status));
}

void handle_sm(){
  String sen = "SM Values: ";
  for(int i=0; i<4; i++){
    sen+= String(get_soil_moisture(sm_pins[i])) + "  ";
  }
  server.send(200, "text/plain",sen);
}

void handle_dht() {
  DHT.read11(dht_pin);
  
  float hum = DHT.humidity;
  float temp = DHT.temperature;
  
  server.send(200, "text/plain", "DHT Hum(%) and Temp(C): " + String(hum) + " " + String(temp));
}

void handle_anen(){
  float total_windspd = 0;
  String readlist = "";
  int num_read = 5;
  for(int i =0; i<num_read; i++){
    float anen_read = analogRead(anen_pin);//read pin
    readlist += String(anen_read);
    float anen_volt = anen_read*analog_to_actual_volt_conversion; //Convert to voltage
    float windspd = 0;
    if(anen_volt<=min_volt){
      windspd = 0;
    }else{
      windspd = ((anen_volt - min_volt)*wind_speedMax / (max_volt - min_volt))*windspd_conv;
      }
    
    total_windspd += windspd;  //get windspd val
    delay(windspd_interval);
  }

  float avg_wind = total_windspd/num_read;
  
  server.send(200, "text/plain", "ANEN avg speed: " + String(avg_wind) + " and " + readlist);
}
void handle_bmp() {
  float pressure = bmp.readPressure();
  Serial.println(String(pressure));
  server.send(200, "text/plain", "BMP Pressure (PA): " + String(pressure));
}

void handle_NotFound() {
  server.send(404, "text/plain", "Not found");
}
