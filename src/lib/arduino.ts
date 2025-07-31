'use server'

import { SerialPort } from 'serialport'

let port: SerialPort | null = null
const baudRate = 9600

/**
 * Gibt immer 'COM6' zurück – der Arduino-Port.
 */
async function resolvePortPath(): Promise<string> {
  return 'COM6'
}

/**
 * Öffnet den seriellen Port mit 9600 Baud.
 */
async function createPort() {
  const path = await resolvePortPath()
  port = new SerialPort({ path, baudRate })
  port.on('error', err => console.error('Serial port error:', err))
  port.on('open', () => console.log(`Serial port ${path} opened.`))
}

/**
 * Stellt sicher, dass der Port offen ist.
 */
async function ensurePort() {
  if (!port) await createPort()
  return port!
}

/**
 * Sendet einen Befehl (z. B. JSON-String) über den seriellen Port.
 */
export async function sendMotorCommand(command: string) {
  const p = await ensurePort()
  await new Promise<void>((resolve, reject) => {
    p.write(command + '\n', err => (err ? reject(err) : resolve()))
  })
}
