// bridge-server.js
const express = require('express');
const bodyParser = require('body-parser');
const { openPort } = require('./serial-port');
 // your module
const app = express();

app.use(bodyParser.json());

const port = openPort();

app.post('/send', (req, res) => {
  const command = req.body.command;
  if (!command) return res.status(400).send('No command provided');

  port.write(command + '\n', (err) => {
    if (err) {
      console.error('Write error:', err.message);
      return res.status(500).send('Serial write failed');
    }
    res.send('Command sent: ' + command);
  });
});

app.listen(3030, () => {
  console.log('Serial bridge listening on http://localhost:3030');
});
