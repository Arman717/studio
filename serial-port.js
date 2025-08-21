const { SerialPort } = require('serialport');

// Default port path (COM7 on Windows) can be overridden via STM32_PORT env var
const path = process.env.STM32_PORT || '/dev/ttyACM0';
// Match STM32 firmware baud rate
const baudRate = 115200;

function openPort() {
  const port = new SerialPort({ path, baudRate });
  port.on('error', (err) => console.error('Serial port error:', err));
  return port;
}

// If run directly, optionally send a command provided as CLI arg
if (require.main === module) {
  const cmd = process.argv[2];
  const port = openPort();
  port.on('open', () => {
    if (cmd) {
      console.log('Sending:', cmd);
      port.write(cmd + '\n', (err) => {
        if (err) {
          console.error('Write error:', err.message);
        } else {
          console.log('Command sent');
        }
        port.close();
      });
    } else {
      console.log(`Serial port ${path} opened at ${baudRate} baud`);
    }
  });
  port.on('data', (data) => process.stdout.write(data.toString()));
}

module.exports = { openPort };