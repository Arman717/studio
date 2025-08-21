const { SerialPort } = require('serialport');

const port = new SerialPort({
  // Use STM32_PORT if set, otherwise default to COM7
  path: process.env.STM32_PORT || 'COM7',
  // Match the STM32 firmware's serial speed
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
