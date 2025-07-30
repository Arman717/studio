'use server'

import type {SerialPort as SerialPortType} from 'serialport'

let SerialPort: typeof SerialPortType | null = null

async function loadSerialPort() {
  if (!SerialPort) {
    try {
      SerialPort = (await import('serialport')).SerialPort
    } catch (err) {
      console.error('Failed to load serialport module', err)
      throw new Error('serialport module not available. Did you run `npm install`?')
    }
  }
  return SerialPort!
}

/**
 * Resolve the serial port for the Arduino. If the `ARDUINO_PORT`
 * environment variable is set it is used directly. Otherwise the first
 * available port that looks like an Arduino (/dev/ttyACM* or COM*) is used.
 */
async function resolvePortPath(): Promise<string> {
  const env = process.env.ARDUINO_PORT
  if (env) return env
  try {
    const Serial = await loadSerialPort()
    const ports = await Serial.list()
    const match = ports.find(p => p.path && (p.path.startsWith('COM') || p.path.includes('ttyACM')))
    if (match?.path) return match.path
  } catch (err) {
    console.error('Failed to list serial ports', err)
  }
  return '/dev/ttyACM0'
}

let port: SerialPortType | null = null

async function createPort() {
  const Serial = await loadSerialPort()
  const path = await resolvePortPath()
  const baudRate = 9600
  port = new Serial({path, baudRate})
  port.on('error', err => console.error('Serial port error', err))
}

async function ensurePort() {
  if (!port) await createPort()
  return port!
}

export async function sendMotorCommand(command: string) {
  const p = await ensurePort()
  await new Promise<void>((resolve, reject) => {
    p.write(command + '\n', err => (err ? reject(err) : resolve()))
  })
}

