 
int measurementnum = 0; // number of cycles
int quicksamplenum = 0; // step inside measurement cycle
const int oneWirePin = 2;

float vref = 5;               // Reference voltage Vref
int resolution = 1023;      // Voltage resolution 

void ReadVals(int duration, boolean state) {
  quicksamplenum = 0;
  float starttime = millis();
  float endtime = starttime;
  while ((endtime - starttime)/1000 <=(duration)) // do this loop for up to 15s
  {  
    int sensorValue = analogRead(A0);   // MQ135 
    int sensorValue1 = analogRead(A1);  // MQ3
    int sensorValue2 = analogRead(A2);  // MQ2
    int sensorValue3 = analogRead(A3);  // MQ6
    int sensorValue4 = analogRead(A4);  // MQ9
    int sensorValue5 = analogRead(A5);  // MQ5
    int sensorValue6 = analogRead(A6);  // MQ8
    int sensorValue7 = analogRead(A7);  // MQ4

    String stateInfo = "State: " + String(state) + "  Measurement No. : " + String(measurementnum) + "  Sample No. : " + String(quicksamplenum);
    String sensorResults = "S1 (MQ135): " +   String(sensorValue) + " S2 (MQ3): " + String(sensorValue1) + " S3 (MQ2): " +  String(sensorValue2) + " S4 (MQ6): " +  String(sensorValue3) + " S5 (MQ9): " +  String(sensorValue4) + " S6 (MQ5): " +  String(sensorValue5) + " S7 (MQ8): " +  String(sensorValue6) + " S8 (MQ4): " +  String(sensorValue7);
    String data = "<WRITE>" + String(sensorValue) + "," + String(sensorValue1) + "," + String(sensorValue2) + "," + String(sensorValue3) + "," + String(sensorValue4) + "," + String(sensorValue5) + "," + String(sensorValue6) + "," + String(sensorValue7) + "</WRITE>";
  
    Serial.println(stateInfo);
    Serial.println(sensorResults);

    // Send data as CSV with newline
    Serial.write(data.c_str());

    delay(300);        // delay in between reads for stability

    endtime = millis();
    quicksamplenum++;
  }
 
 
} 

//NOTE: Relay is an active-low (Inverted logics)
void CleanSetup() {
  //set blow pin
  digitalWrite(10, LOW); //pump on
  //set valve pin
  digitalWrite(9, HIGH); //valve clean
}

void DirtySetup() {

  //set blow pin
  digitalWrite(10, LOW); //pump on
  //set valve pin
  digitalWrite(9, LOW); //valve clean
 
}

void OffSetup() {
  //set blow pin
  digitalWrite(10, HIGH); //pump off
  //set valve pin
  digitalWrite(9, HIGH); //valve off
}
 

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  // sensor.begin(); 
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(3, OUTPUT);
  digitalWrite(3, HIGH); //PWM PIN STUFF
    
  OffSetup();
  Serial.println("Device is in off state");
  delay(10000);
  
  CleanSetup();
  Serial.println("Device is self cleaning");

  delay(30000);

}

// the loop routine runs over and over again forever:
void loop() {

  DirtySetup();
  Serial.println("Initialise Dirty Read"); 

  ReadVals(1500, 1);

  // CleanSetup();
  // Serial.println("Initialise Clean Read");
  // Serial.println("Complete");

  // delay(30000);
  // ReadVals(150, 0);

  measurementnum++;
}
