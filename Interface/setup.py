import sys
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt

CONFIG_FILE = "config.json"
SKETCH_FILE = "ENOSE_AUTO.ino"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"flood": "", "purge": "", "stageDelay": ""}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def print_config(config):
    print("\nCurrent Configuration:")
    for key, value in config.items():
        print(f" - {key}: {value}")

def ask_user_yes_no(prompt="Would you like to update the config? (y/n): "):
    while True:
        response = input(prompt).strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'.")

class ConfigForm(QWidget):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("E-NOSE Config Editor")
        self.setFixedSize(300, 260)
        self.updated_config = config.copy()

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                font-family: 'Segoe UI';
                font-size: 12pt;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #f85a3e;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e14b2d;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(20, 10, 20, 10)
        form_layout.setSpacing(15)

        self.flood_input = QLineEdit(str(config.get("flood", "")))
        self.purge_input = QLineEdit(str(config.get("purge", "")))
        self.stage_delay_input = QLineEdit(str(config.get("stageDelay", "")))


        form_layout.addRow("Flood Duration (s):", self.flood_input)
        form_layout.addRow("Purge Duration (s):", self.purge_input)
        form_layout.addRow("Stage Delay (ms):", self.stage_delay_input)

        layout.addLayout(form_layout)

        self.register_button = QPushButton("Update Settings")
        self.register_button.clicked.connect(self.save_config_and_close)
        layout.addWidget(self.register_button, alignment=Qt.AlignCenter) # type: ignore

        self.setLayout(layout)

    def save_config_and_close(self):
        self.updated_config["flood"] = self.flood_input.text()
        self.updated_config["purge"] = self.purge_input.text()
        self.updated_config["stageDelay"] = self.stage_delay_input.text()

        save_config(self.updated_config)
        generate_sketch(self.updated_config)
        QMessageBox.information(self, "Saved", "Configuration updated.")
        print_config(self.updated_config)
        self.close()

def run_gui(config):
    app = QApplication(sys.argv)
    window = ConfigForm(config)
    window.show()
    app.exec_()

def generate_sketch(params):
    header_snippet = """\
// Auto-generated Arduino Sketch
// CEFIM E-NOSE AUTOMATION v0.1
// Date: 2025/07/10
// Author: Kenna Geleta (u23575035@tuks.co.za)
// (C) 2025 University of Pretoria

"""

    constant_snippet = """
int measurementnum = 0; // number of cycles
int quicksamplenum = 0; // step inside measurement cycle
const int oneWirePin = 2;

float vref = 5;         // Reference voltage Vref
int resolution = 1023;  // Voltage resolution 
"""

    readval_snippet = """
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
"""

    cleanval_snippet = """
void CleanSetup() {
  digitalWrite(10, LOW); // pump on
  digitalWrite(9, HIGH); // valve clean
}
"""

    dirtysetup_snippet = """
void DirtySetup() {
  digitalWrite(10, LOW); // pump on
  digitalWrite(9, LOW);  // valve dirty
}
"""

    offsetup_snippet = """
void OffSetup() {
  digitalWrite(10, HIGH); // pump off
  digitalWrite(9, HIGH);  // valve off
}
"""

    setup_snippet = """
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
"""

    loop_snippet = f"""
void loop() {{
  DirtySetup();
  Serial.println("Initialise Dirty Read"); 

  ReadVals({params['flood']}, 1);

  CleanSetup();
  Serial.println("Initialise Clean Read");
  Serial.println("Complete");

  delay({params['stageDelay']});
  ReadVals({params['purge']}, 0);
}}
"""

    full_sketch = (
        header_snippet
        + constant_snippet
        + readval_snippet
        + cleanval_snippet
        + dirtysetup_snippet
        + offsetup_snippet
        + setup_snippet
        + loop_snippet
    )

    with open(SKETCH_FILE, "w") as file:
        file.write(full_sketch)
    print(f"\n📄 Arduino sketch saved to: {SKETCH_FILE}")

# ======================= RUN MAIN LOOP =======================

if __name__ == "__main__":
    print("CEFIM E-NOSE AUTOMATION CONFIG TOOL")
    while True:
        config = load_config()
        print_config(config)

        if not ask_user_yes_no("\nWould you like to update the config? (y/n): "):
            print("\n✅ Exiting. Config unchanged.")
            break

        run_gui(config)
