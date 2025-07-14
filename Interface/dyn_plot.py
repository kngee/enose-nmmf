import subprocess
import json
import os
import time
import csv
import random # For dummy data if serial not connected

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

# --- Helper function to run arduino-cli commands ---
def run_arduino_cli_command(command_args):
    """
    Executes an arduino-cli command using subprocess.
    Args:
        command_args (list): A list of strings representing the arduino-cli command
                             and its arguments, e.g., ["compile", "--fqbn", "arduino:avr:uno", "my_sketch"].
    Returns:
        tuple: (stdout, stderr, returncode)
    Raises:
        Exception: If the arduino-cli command fails to execute.
    """
    full_command = ["arduino-cli"] + command_args
    print(f"Executing command: {' '.join(full_command)}")
    try:
        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False # Do not raise CalledProcessError for non-zero exit codes immediately
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        returncode = process.returncode

        if returncode != 0:
            print(f"Arduino CLI command failed with exit code {returncode}")
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            raise Exception(f"Arduino CLI command failed: {stderr if stderr else stdout}")

        return stdout, stderr, returncode

    except FileNotFoundError:
        raise Exception("arduino-cli not found. Please ensure it's installed and in your system's PATH.")
    except Exception as e:
        raise Exception(f"Error executing arduino-cli command: {e}")

# --- SensorMonitorApp Class (modified for stage display) ---

class SensorMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time MQ Sensor Monitor")
        self.setGeometry(100, 100, 1200, 800)

        self.sensor_labels = ["MQ135", "MQ3", "MQ6", "MQ9", "MQ5", "MQ8", "MQ4"]
        self.sensor_data = {label: [] for label in self.sensor_labels}
        self.xdata = []
        self.ani = None
        self.csv_fh = None
        self.csv_writer = None
        self.folder_name = None

        self.data_started = False
        self.start_time = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)

        self.gas_name = "DefaultGas"
        self.gas_concentration = "N/A"
        self.run_type = "Response"
        self.current_stage = "Unknown" # New attribute to hold the current stage

        self.configs = self.load_configs()
        # Initial values for serial port and FQBN (will be updated after board list)
        self.serial_port = self.configs.get("serial_port", "COM11")
        self.baud_rate = self.configs.get("baud_rate", 9600)
        self.arduino_fqbn = None # This will be detected
        self.arduino_sketch_path = self.configs.get("arduino_sketch_path", "ENOSE_AUTO.ino") # Path to your sketch folder

        self.ser = None

        self.setup_ui()
        self.setup_plot()

    def load_configs(self):
        """Loads configuration settings from config.json."""
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    configs = json.load(f)
                    # Ensure essential keys are present, or add defaults
                    if "serial_port" not in configs: configs["serial_port"] = "COM11"
                    if "baud_rate" not in configs: configs["baud_rate"] = 9600
                    if "arduino_sketch_path" not in configs: configs["arduino_sketch_path"] = "ENOSE_AUTO.ino"
                    if "animation_interval_ms" not in configs: configs["animation_interval_ms"] = 200
                    return configs
            except json.JSONDecodeError as e:
                print(f"Error decoding config.json: {e}. Using default settings.")
                return {"serial_port": "COM11", "baud_rate": 9600, "arduino_sketch_path": "ENOSE_AUTO.ino", "animation_interval_ms": 200}
        print(f"config.json not found at {config_path}. Using default settings.")
        return {"serial_port": "COM11", "baud_rate": 9600, "arduino_sketch_path": "ENOSE_AUTO.ino", "animation_interval_ms": 200}

    def setup_ui(self):
        """Sets up the main user interface layout including sidebar and plot area."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Sidebar Layout (Vertical)
        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.setAlignment(QtCore.Qt.AlignTop)
        sidebar_layout.setSpacing(15)

        # Sidebar Title
        title_label = QtWidgets.QLabel("Monitoring Controls")
        title_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        sidebar_layout.addWidget(line)

        # Start/Stop Button
        self.start_button = QtWidgets.QPushButton("Start Data")
        self.start_button.setFont(QtGui.QFont("Arial", 12))
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 5px;")
        self.start_button.clicked.connect(self.toggle_data_collection)
        sidebar_layout.addWidget(self.start_button)

        # Timer Display Section
        timer_label_title = QtWidgets.QLabel("Run Duration:")
        timer_label_title.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        sidebar_layout.addWidget(timer_label_title)

        self.timer_display = QtWidgets.QLabel("00:00:00")
        self.timer_display.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.timer_display.setAlignment(QtCore.Qt.AlignCenter)
        self.timer_display.setStyleSheet("color: #333333; padding: 5px; border: 1px solid #ccc; border-radius: 5px; background-color: #f0f0f0;")
        sidebar_layout.addWidget(self.timer_display)

        # --- NEW: Current Stage Display ---
        stage_label_title = QtWidgets.QLabel("Current Stage:")
        stage_label_title.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        sidebar_layout.addWidget(stage_label_title)

        self.current_stage_label = QtWidgets.QLabel(self.current_stage)
        self.current_stage_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.current_stage_label.setAlignment(QtCore.Qt.AlignCenter)
        # Dynamic styling for current stage
        self.current_stage_label.setStyleSheet(
            "QLabel { "
            "   color: white; "
            "   background-color: #607D8B; " # Default grey
            "   border-radius: 5px; "
            "   padding: 5px; "
            "   border: 1px solid #455A64; "
            "}"
        )
        sidebar_layout.addWidget(self.current_stage_label)
        # --- END NEW ---

        # Run Information Display (Gas Name and Concentration)
        info_group_box = QtWidgets.QGroupBox("Current Run Details")
        info_group_box.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        info_group_box.setStyleSheet("QGroupBox { border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; }"
                                     "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        info_layout = QtWidgets.QVBoxLayout(info_group_box)

        self.gas_name_label = QtWidgets.QLabel(f"Gas Name: {self.gas_name}")
        self.gas_name_label.setFont(QtGui.QFont("Arial", 10))
        self.gas_concentration_label = QtWidgets.QLabel(f"Concentration: {self.gas_concentration}")
        self.gas_concentration_label.setFont(QtGui.QFont("Arial", 10))
        self.run_type_label = QtWidgets.QLabel(f"Run Type: {self.run_type}")
        self.run_type_label.setFont(QtGui.QFont("Arial", 10))


        info_layout.addWidget(self.gas_name_label)
        info_layout.addWidget(self.gas_concentration_label)
        info_layout.addWidget(self.run_type_label)
        sidebar_layout.addWidget(info_group_box)

        # Active Run Settings from config.json
        configs_group_box = QtWidgets.QGroupBox("Active Sensor Settings")
        configs_group_box.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        configs_group_box.setStyleSheet("QGroupBox { border: 1px solid #a0a0a0; border-radius: 5px; margin-top: 10px; background-color: #e6e6e6; }"
                                        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        configs_layout = QtWidgets.QVBoxLayout(configs_group_box)

        if self.configs:
            for key, value in self.configs.items():
                setting_label = QtWidgets.QLabel(f"  • {key.replace('_', ' ').title()}: <b>{value}</b>")
                setting_label.setFont(QtGui.QFont("Arial", 9))
                setting_label.setStyleSheet("color: #444; padding-bottom: 2px;")
                configs_layout.addWidget(setting_label)
        else:
            configs_layout.addWidget(QtWidgets.QLabel("No settings loaded from config.json"))

        sidebar_layout.addWidget(configs_group_box)


        sidebar_layout.addStretch()

        main_layout.addLayout(sidebar_layout, 1)

        # Plot Area
        self.plot_widget = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        main_layout.addWidget(self.plot_widget, 4)

    def setup_plot(self):
        """Initializes the Matplotlib figure, axes, and canvas for plotting."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout.addWidget(self.canvas)

        self.lines = {}
        for label in self.sensor_labels:
            (line,) = self.ax.plot([], [], label=label)
            self.lines[label] = line

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Combined MQ Sensor Response")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Sensor Output Normalized to Air (V/V)")
        self.ax.grid(True)
        self.ax.legend(loc='upper right')
        self.fig.tight_layout()

    def setup_csv(self):
        """Sets up the CSV file for data logging, creating a new file for each run."""
        if self.csv_fh and not self.csv_fh.closed:
            self.csv_fh.close()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.folder_name = f"Results/{self.run_type}/{self.gas_name}-{timestr}"
        os.makedirs(self.folder_name, exist_ok=True)
        self.csv_file = f"{self.folder_name}/{self.gas_name}-{timestr}.csv"

        try:
            self.csv_fh = open(self.csv_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_fh)
            self.csv_writer.writerow(self.sensor_labels + ["Stage"]) # Add Stage column to CSV
            print(f"CSV data will be saved to: {self.csv_file}")
        except IOError as e:
            QtWidgets.QMessageBox.critical(self, "CSV File Error", f"Error opening CSV file for writing: {e}\nData will not be saved.")
            self.csv_fh = None

    def prompt_user_for_run_details(self):
        """Displays a dialog to prompt the user for gas name, concentration, and run type."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Enter Run Details")
        dialog.setModal(True)
        dialog.setFixedSize(350, 280)

        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        dialog_layout.setSpacing(15)

        dialog_title = QtWidgets.QLabel("New Experiment Details")
        dialog_title.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        dialog_title.setAlignment(QtCore.Qt.AlignCenter)
        dialog_layout.addWidget(dialog_title)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)

        # Gas Name Input
        gas_name_label = QtWidgets.QLabel("Gas Name:")
        gas_name_label.setFont(QtGui.QFont("Arial", 10))
        gas_name_input = QtWidgets.QLineEdit()
        gas_name_input.setText(self.gas_name)
        gas_name_input.setFont(QtGui.QFont("Arial", 10))
        gas_name_input.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
        form_layout.addRow(gas_name_label, gas_name_input)

        # Concentration Input
        concentration_label = QtWidgets.QLabel("Concentration:")
        concentration_label.setFont(QtGui.QFont("Arial", 10))
        concentration_input = QtWidgets.QLineEdit()
        concentration_input.setText(self.gas_concentration)
        concentration_input.setFont(QtGui.QFont("Arial", 10))
        concentration_input.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
        form_layout.addRow(concentration_label, concentration_input)

        # Run Type Selection
        run_type_label = QtWidgets.QLabel("Run Type:")
        run_type_label.setFont(QtGui.QFont("Arial", 10))
        run_type_combo = QtWidgets.QComboBox()
        run_type_combo.setFont(QtGui.QFont("Arial", 10))
        run_type_combo.addItem("Response")
        run_type_combo.addItem("Training")
        run_type_combo.setCurrentText(self.run_type)
        run_type_combo.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px; combobox-popup: 0;")
        form_layout.addRow(run_type_label, run_type_combo)

        dialog_layout.addLayout(form_layout)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        button_box.setStyleSheet(
            "QPushButton { padding: 8px 15px; border-radius: 5px; font-size: 10pt; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
            "QPushButton#OK { background-color: #4CAF50; color: white; }"
            "QPushButton#Cancel { background-color: #f44336; color: white; }"
        )
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setObjectName("OK")
        button_box.button(QtWidgets.QDialogButtonBox.Cancel).setObjectName("Cancel")

        dialog_layout.addWidget(button_box)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.gas_name = gas_name_input.text().strip()
            self.gas_concentration = concentration_input.text().strip()
            self.run_type = run_type_combo.currentText()

            self.gas_name_label.setText(f"Gas Name: {self.gas_name}")
            self.gas_concentration_label.setText(f"Concentration: {self.gas_concentration}")
            self.run_type_label.setText(f"Run Type: {self.run_type}")

            self.ax.set_title(f"{self.gas_name} ({self.gas_concentration}) - {self.run_type} Run")
            self.canvas.draw_idle()
            return True
        return False

    def clear_plot(self):
        """Clears all data from the plot lines and resets axis limits."""
        self.xdata = []
        for label in self.sensor_labels:
            self.sensor_data[label] = []
            self.lines[label].set_data([], [])

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.canvas.draw_idle()

    def detect_arduino_board(self):
        """
        Detects the connected Arduino board's FQBN and port using arduino-cli.
        Assumes only one Arduino board is connected.
        """
        try:
            stdout, _, _ = run_arduino_cli_command(["board", "list", "--format", "json"])
            brds = json.loads(stdout)

            self.arduino_fqbn = None
            self.serial_port = None

            for port_info in brds.get('detected_ports', []):
                if 'matching_boards' in port_info and port_info['matching_boards']:
                    # Assuming only one Arduino is connected, take the first match
                    self.arduino_fqbn = port_info['matching_boards'][0]['fqbn']
                    self.serial_port = port_info['port']['address']
                    print(f"Detected Arduino FQBN: {self.arduino_fqbn}")
                    print(f"Detected Arduino Port: {self.serial_port}")
                    return True # Board found
            
            QtWidgets.QMessageBox.warning(self, "Arduino Detection",
                                          "No Arduino board detected. Please ensure it's connected and drivers are installed.")
            return False # Board not found

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Arduino CLI Error",
                                           f"Failed to list Arduino boards: {e}\n"
                                           "Please ensure arduino-cli is installed and configured correctly.")
            return False

    def update_stage_display(self, stage_value):
        """Updates the stage label text and styling based on the serial input."""
        if stage_value == "0":
            self.current_stage = "Exposure Stage 🌬️"
            self.current_stage_label.setStyleSheet(
                "QLabel { "
                "   color: white; "
                "   background-color: #2196F3; " # Blue for Exposure
                "   border-radius: 5px; "
                "   padding: 5px; "
                "   border: 1px solid #1976D2; "
                "}"
            )
        elif stage_value == "1":
            self.current_stage = "Cleaning Stage 💨"
            self.current_stage_label.setStyleSheet(
                "QLabel { "
                "   color: white; "
                "   background-color: #FFC107; " # Amber for Cleaning
                "   border-radius: 5px; "
                "   padding: 5px; "
                "   border: 1px solid #FFA000; "
                "}"
            )
        else:
            self.current_stage = "Unknown Stage ❓"
            self.current_stage_label.setStyleSheet(
                "QLabel { "
                "   color: white; "
                "   background-color: #607D8B; " # Default grey
                "   border-radius: 5px; "
                "   padding: 5px; "
                "   border: 1px solid #455A64; "
                "}"
            )
        self.current_stage_label.setText(self.current_stage)


    def toggle_data_collection(self):
        """Starts or stops the data collection and plot animation, including serial communication."""
        if not self.data_started:
            if not self.prompt_user_for_run_details():
                return

            # --- Arduino Firmware Upload ---
            if not self.detect_arduino_board():
                return # Stop if Arduino not detected

            if not self.arduino_fqbn or not self.serial_port:
                QtWidgets.QMessageBox.critical(self, "Arduino Error", "Could not determine Arduino FQBN or Port.")
                return

            try:
                # Compile the sketch
                # Get the directory of the sketch file
                sketch_dir = os.path.dirname(self.arduino_sketch_path)
                if not sketch_dir: # If path is just filename, assume current directory
                    sketch_dir = "."
                
                compile_args = ["compile", "--fqbn", self.arduino_fqbn, sketch_dir] # Pass the directory, not the .ino file
                run_arduino_cli_command(compile_args)
                QtWidgets.QMessageBox.information(self, "Arduino Compile", "Sketch compiled successfully!")

                # Upload the sketch
                upload_args = ["upload", "--fqbn", self.arduino_fqbn, "--port", self.serial_port, sketch_dir] # Pass the directory
                run_arduino_cli_command(upload_args)
                QtWidgets.QMessageBox.information(self, "Arduino Upload", "Sketch uploaded successfully!")

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Arduino CLI Error", f"Arduino firmware upload failed: {e}")
                return # Stop if upload fails

            # --- Serial Port Initialization ---
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = serial.Serial(port=self.serial_port, baudrate=self.baud_rate, timeout=1)
                time.sleep(2) # Give Arduino time to reset after connection
                print(f"Serial port {self.serial_port} opened successfully.")
                self.update_stage_display("?") # Initialize stage display
            except serial.SerialException as e:
                QtWidgets.QMessageBox.critical(self, "Serial Port Error",
                                               f"Could not open serial port {self.serial_port}: {e}\n"
                                               "Please check the port name, connection, and ensure Arduino IDE Serial Monitor is closed.")
                self.ser = None
                return

            self.clear_plot()
            self.setup_csv()

            self.start_button.setText("Stop Data")
            self.start_button.setStyleSheet("background-color: #f44336; color: white; border-radius: 5px; padding: 5px;")
            self.data_started = True
            self.start_time = time.time()
            self.timer.start(1000)

            animation_interval = self.configs.get("animation_interval_ms", 200)
            self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=animation_interval, blit=False)
            self.canvas.draw_idle()
        else:
            self.start_button.setText("Start Data")
            self.start_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 5px;")
            self.data_started = False
            self.timer.stop()

            if self.ani:
                self.ani.event_source.stop()
                self.ani = None

            if self.ser and self.ser.is_open:
                self.ser.close()
                print(f"Serial port {self.serial_port} closed.")

            if self.csv_fh and not self.csv_fh.closed:
                self.csv_fh.close()
                print(f"Data saved to {self.csv_file}")

            self.save_plot()
            self.save_current_config()
            print("Live plot interrupted.")

    def update_plot(self, frame):
        """Updates the plot with new serial sensor data."""
        if not self.data_started:
            return []

        new_row = []
        if self.ser and self.ser.is_open:
            try:
                data_in_waiting = self.ser.in_waiting
                if data_in_waiting > 0:
                    raw_data = self.ser.read(data_in_waiting).decode('utf-8', errors='ignore')
                    # print(f"Raw serial data: {raw_data.strip()}") # For debugging raw input

                    # --- Parse stage information and sensor data ---
                    # Split by the closing tag to process complete messages
                    messages = raw_data.split('</WRITE>')
                    for msg_part in messages:
                        if "<WRITE>" in msg_part:
                            start_index = msg_part.find("<WRITE>") + len("<WRITE>")
                            content = msg_part[start_index:].strip() # Get content between tags

                            # --- CRITICAL CHANGE: Check if content is a single digit '0' or '1' ---
                            if len(content) == 1 and content in ['0', '1']:
                                self.update_stage_display(content)
                                print(f"Stage changed to: {self.current_stage}") # Debugging stage changes
                                # No sensor data expected when stage is sent alone
                                new_row = [] # Ensure new_row is reset
                            else: # Assume it's sensor data if not a single digit stage command
                                try:
                                    # Attempt to parse as sensor data (comma-separated floats)
                                    parsed_data = list(map(float, content.split(",")))
                                    if len(parsed_data) == len(self.sensor_labels):
                                        new_row = parsed_data
                                    else:
                                        print(f"Warning: Sensor data length mismatch ({len(parsed_data)} != {len(self.sensor_labels)}). Skipping: {content}")
                                        new_row = []
                                except ValueError as ve:
                                    print(f"Warning: Could not convert data to float. Skipping: '{content}' Error: {ve}")
                                    new_row = []

                            # Process sensor data if a valid row was obtained
                            if new_row:
                                self.xdata.append(len(self.xdata))
                                if self.csv_writer:
                                    # Ensure 'current_stage' is always defined for CSV
                                    stage_for_csv = self.current_stage.split(' ')[0] if self.current_stage != "Unknown" else "Unknown"
                                    self.csv_writer.writerow(new_row + [stage_for_csv])
                                    self.csv_fh.flush()

                                for i, label in enumerate(self.sensor_labels):
                                    self.sensor_data[label].append(new_row[i])

            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
                self.data_started = False
                QtWidgets.QMessageBox.critical(self, "Serial Error", f"Serial communication interrupted: {e}\nData collection stopped.")
                self.toggle_data_collection()
                return []
            except Exception as e:
                print(f"Unexpected error reading serial: {e}")
                new_row = [] # Reset to avoid processing incomplete data
        else:
            # Fallback to random data if serial is not connected or failed
            if self.data_started:
                self.xdata.append(len(self.xdata))
                new_row = [random.uniform(0.1, 1.0) for _ in self.sensor_labels]
                # Simulate stage change for dummy data every 50 samples
                if len(self.xdata) > 0 and len(self.xdata) % 50 == 0:
                    self.update_stage_display("0" if self.current_stage == "Cleaning Stage 💨" else "1")
                
                if self.csv_writer:
                    stage_for_csv = self.current_stage.split(' ')[0] if self.current_stage != "Unknown" else "Unknown"
                    self.csv_writer.writerow(new_row + [stage_for_csv])
                    self.csv_fh.flush()
                for i, label in enumerate(self.sensor_labels):
                    self.sensor_data[label].append(new_row[i])
            else:
                return []

        # Only update plot if a valid new_row was processed (can be empty if only stage command was received)
        if self.xdata and new_row: # new_row ensures we only plot if sensor data was just added
            for i, label in enumerate(self.sensor_labels):
                data = self.sensor_data[label]

                min_val = min(data) if data else 0
                max_val = max(data) if data else 1
                
                if max_val != min_val:
                    norm_data = [(x - min_val) / (max_val - min_val) for x in data]
                else:
                    norm_data = [1.0 for _ in data] # If all values are same, normalize to 1.0

                self.lines[label].set_data(range(len(norm_data)), norm_data)

            self.ax.set_xlim(0, len(self.xdata) + 10)
            self.canvas.draw_idle()
        elif not self.xdata and not new_row: # If no data yet and no new row, still return empty list
            return []
        
        return list(self.lines.values())

    def update_timer_display(self):
        """Updates the timer display label in the sidebar."""
        if self.data_started:
            elapsed_time = int(time.time() - self.start_time)
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.timer_display.setText(f"{hours:02}:{minutes:02}:{seconds:02}")
        else:
            self.timer_display.setText("00:00:00")

    def save_plot(self):
        """Saves the current Matplotlib plot to a PDF file."""
        if not self.fig or not self.ax:
            return

        plot_title = f"{self.gas_name} ({self.gas_concentration}) - {self.run_type} Run"
        self.ax.set_title(plot_title)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not self.folder_name:
            self.folder_name = f"Results/{self.run_type}/{self.gas_name}-{timestr}"
            os.makedirs(self.folder_name, exist_ok=True)

        plot_filename = f"{self.folder_name}/{self.gas_name}-{self.gas_concentration}-{timestr}.pdf"

        try:
            self.fig.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Plot Error", f"Could not save plot: {e}")

    def save_current_config(self):
        """Saves the currently loaded config settings to a JSON file in the run's folder."""
        if not self.configs:
            print("No configs loaded, skipping saving current config.")
            return

        if not self.folder_name:
            print("Run folder not defined, cannot save config file.")
            return

        timestr = time.strftime("%Y%m%d-%H%M%S")
        config_filename = f"{self.folder_name}/conf-{timestr}.json"

        try:
            with open(config_filename, 'w') as f:
                json.dump(self.configs, f, indent=4)
            print(f"Current config saved to {config_filename}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Config Error", f"Could not save config file: {e}")


    def closeEvent(self, event):
        """Ensures proper cleanup on application close."""
        if self.ani:
            self.ani.event_source.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.csv_fh and not self.csv_fh.closed:
            self.csv_fh.close()
            print(f"Final data saved to {self.csv_file}")
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.serial_port} closed.")
        super().closeEvent(event)

if __name__ == '__main__':
    # Create a dummy config.json if it doesn't exist for testing purposes
    config_file_path = "config.json"
    if not os.path.exists(config_file_path):
        dummy_configs = {
            "serial_port": "COM11", # <<< IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S PORT (e.g., COM3, /dev/ttyACM0)
            "baud_rate": 9600,
            "animation_interval_ms": 200, # Controls how fast the plot updates (ms)
            "arduino_sketch_path": "./ENOSE_AUTO", # <<< IMPORTANT: PATH TO YOUR ARDUINO SKETCH FOLDER (e.g., "./my_sketch")
            "sensor_model": "MQ-Series v2.1",
            "calibration_date": "2024-01-15",
            "firmware_version": "1.2.0",
            "adc_resolution": "10-bit",
            "gain_setting": "Low"
        }
        with open(config_file_path, 'w') as f:
            json.dump(dummy_configs, f, indent=4)
        print(f"Created a dummy {config_file_path} for demonstration. "
              "Please edit it to set your Arduino's COM port and sketch path.")

    # Create Results directory if it doesn't exist
    os.makedirs("Results", exist_ok=True)

    app = QtWidgets.QApplication([])
    window = SensorMonitorApp()
    window.show()
    app.exec_()