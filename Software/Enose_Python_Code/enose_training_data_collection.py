import serial
import csv
import time

# Read data boolean variable
read : bool = True

# Set up the serial connection
ser = serial.Serial(
    port='COM11',       # Replace with your Arduino's serial port
    baudrate=9600,      # Match the baud rate in your Arduino code
)

# File name for the CSV
timestr = time.strftime("%Y%m%d-%H%M%S")
gas_name = input("What is the name of your gas?\n").capitalize()
csv_file = gas_name + "-training-" + timestr + ".csv"
print("The file will be saved to: " + csv_file)

try:
    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["MQ135", "MQ3", "MQ6", "MQ9", "MQ5", "MQ8", "MQ4"])

        # Write a header row to the CSV file
        # csv_writer.writerow(["Timestamp", "Filtered Data"])

        print("Listening for Serial.write data and saving to CSV...")
        while read:
            if ser.in_waiting > 0:
                
                data = ser.read(ser.in_waiting).decode('utf-8')  # Read and decode
                state = 0 # EXPOSURE - Write to CSV
            
                if "<WRITE>" in data:  # Only process Serial.write tagged data
                    start = data.find("<WRITE>") + len("<WRITE>")
                    end = data.find("</WRITE>")
                    filtered_data = data[start:end].strip()

                    # If the state is related to state change, update the state variable and write only in the flooding state.
                    if len(filtered_data) == 1:
                        state = filtered_data

                    # Log data to CSV - IF IN FLOODING STATE
                    if (state == 0):
                        csv_writer.writerow([filtered_data])

                    # Print filtered data for debugging
                    print(f"Logged: {data}")

except KeyboardInterrupt:
    print("\nExiting and saving the CSV file...")
finally:
    ser.close()