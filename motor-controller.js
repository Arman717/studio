const { SerialPort } = require('serialport');

const port = new SerialPort({
  path: 'COM6',
  baudRate: 9600,
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
