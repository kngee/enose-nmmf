import pyduinocli

arduino = pyduinocli.Arduino("./arduino-cli")
brds = arduino.board.list()

print(brds)

# assuming we are using the first board listed
port = brds['result'][0]['port']['address']
fqbn = brds['result'][0]['matching_boards'][0]['fqbn']

arduino.compile(fqbn=fqbn, sketch="upload_me")
arduino.upload(fqbn=fqbn, sketch="upload_me", port=port)