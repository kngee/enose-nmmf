// Auto-generated Arduino Sketch
// CEFIM E-NOSE AUTOMATION v0.1
// Date: 2025/07/10
// Author: Kenna Geleta (u23575035@tuks.co.za)
// (C) 2025 University of Pretoria


int measurementnum = 0; // number of cycles
int quicksamplenum = 0; // step inside measurement cycle
const int oneWirePin = 2;

float vref = 5;         // Reference voltage Vref
int resolution = 1023;  // Voltage resolution 

void ReadVals(int duration, boolean state) {
  quicksamplenum = 0;
  float starttime = millis();
  float endtime = starttime;

  String stateInfo = "<WRITE>" + String(state) + "</WRITE>";
  Serial.write(stateInfo.c_str());

  while ((endtime - starttime)/1000 <=(duration)) {
    int sensorValue = analogRead(A0);   // MQ135 
    int sensorValue1 = analogRead(A1);  // MQ3
    int sensorValue3 = analogRead(A3);  // MQ6
    int sensorValue4 = analogRead(A4);  // MQ9
    int sensorValue5 = analogRead(A5);  // MQ5
    int sensorValue6 = analogRead(A6);  // MQ8
    int sensorValue7 = analogRead(A7);  // MQ4

    String stateInfo = "State: " + String(state) + "  Measurement No. : " + String(measurementnum) + "  Sample No. : " + String(quicksamplenum);
    String sensorResults = "S1 (MQ135): " +   String(sensorValue) + " S2 (MQ3): " + String(sensorValue1) + " S4 (MQ6): " +  String(sensorValue3) + " S5 (MQ9): " +  String(sensorValue4) + " S6 (MQ5): " +  String(sensorValue5) + " S7 (MQ8): " +  String(sensorValue6) + " S8 (MQ4): " +  String(sensorValue7);
    String data = "<WRITE>" + String(sensorValue) + "," + String(sensorValue1) + "," + String(sensorValue3) + "," + String(sensorValue4) + "," + String(sensorValue5) + "," + String(sensorValue6) + "," + String(sensorValue7) + "</WRITE>";
  
    Serial.println(stateInfo);
    Serial.println(sensorResults);
    Serial.write(data.c_str());

    delay(300);
    endtime = millis();
    quicksamplenum++;
  }
}

void CleanSetup() {
  digitalWrite(10, LOW); // pump on
  digitalWrite(9, HIGH); // valve clean
}

void DirtySetup() {
  digitalWrite(10, LOW); // pump on
  digitalWrite(9, LOW);  // valve dirty
}

void OffSetup() {
  digitalWrite(10, HIGH); // pump off
  digitalWrite(9, HIGH);  // valve off
}

void setup() {
  Serial.begin(9600);
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(3, OUTPUT);
  digitalWrite(3, HIGH);

  OffSetup();
  Serial.println("Device is in off state");
  delay(10000);

  CleanSetup();
  Serial.println("Device is self cleaning");
  delay(30000);
}

void loop() {
  DirtySetup();
  Serial.println("Initialise Dirty Read"); 

  ReadVals(100, 1);

  CleanSetup();
  Serial.println("Initialise Clean Read");
  Serial.println("Complete");

  delay(30000);
  ReadVals(200, 0);
}
