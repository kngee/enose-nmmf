import serial
import csv


# Read data boolean variable
read : bool = True

# Set up the serial connection
ser = serial.Serial(
    port='COM11',       # Replace with your Arduino's serial port
    baudrate=9600,     # Match the baud rate in your Arduino code
)

# File name for the CSV
csv_file = "air_07_07_25.csv"

try:
    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        # Write a header row to the CSV file
        # csv_writer.writerow(["Timestamp", "Filtered Data"])

        print("Listening for Serial.write data and saving to CSV...")
        while read:
            if ser.in_waiting > 0:
                
                data = ser.read(ser.in_waiting).decode('utf-8')  # Read and decode
                
                # Stop reading data
                if data == "Complete":
                    read = False
            
                if "<WRITE>" in data:  # Only process Serial.write tagged data
                    start = data.find("<WRITE>") + len("<WRITE>")
                    end = data.find("</WRITE>")
                    filtered_data = data[start:end].strip()

                    # Log data to CSV
                    csv_writer.writerow([filtered_data])

                    # Print filtered data for debugging
                    print(f"Logged: {filtered_data}")

except KeyboardInterrupt:
    print("\nExiting and saving the CSV file...")
finally:
    ser.close()