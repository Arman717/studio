const { SerialPort } = require('serialport');

const port = new SerialPort({
  // Use ARDUINO_PORT if set, otherwise default to COM6
  path: process.env.ARDUINO_PORT || 'COM6',
  // Match the ESP8266 sketch's faster serial speed
  baudRate: 115200,
});

port.on('open', () => {
  const cmd = process.argv[2] || 'A0';
  console.log('Sende:', cmd);
  port.write(cmd + '\n', (err) => {
    if (err) {
      console.error('Fehler beim Schreiben:', err.message);
    } else {
      console.log('Befehl gesendet!');
    }
    port.close();
  });
});
