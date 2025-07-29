'use server'

import {SerialPort} from 'serialport'

let port: SerialPort | null = null

function createPort() {
  const path = process.env.ARDUINO_PORT ?? '/dev/ttyACM0'
  const baudRate = 9600
  port = new SerialPort({path, baudRate})
  port.on('error', err => console.error('Serial port error', err))
}

async function ensurePort() {
  if (!port) createPort()
  return port!
}

export async function sendMotorCommand(command: string) {
  const p = await ensurePort()
  await new Promise<void>((resolve, reject) => {
    p.write(command + '\n', err => (err ? reject(err) : resolve()))
  })
}

