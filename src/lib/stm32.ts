'use server'

import { SerialPort } from 'serialport'

let port: SerialPort | null = null
// STM32 firmware uses a 115200 baud serial connection
const baudRate = 115200

/**
 * Resolves the serial port to use for the STM32 board.
 * Uses the STM32_PORT env variable or falls back to `COM7`.
 */
async function resolvePortPath(): Promise<string> {
  return process.env.STM32_PORT || 'COM7'
}

/**
 * Öffnet den seriellen Port mit 115200 Baud.
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
