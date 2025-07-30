import {NextRequest, NextResponse} from 'next/server'
import {sendMotorCommand} from '@/lib/arduino'

export async function POST(req: NextRequest) {
  try {
    const {command} = await req.json()
    await sendMotorCommand(String(command))
    return NextResponse.json({status: 'ok'})
  } catch (err: any) {
    console.error(err)
    return NextResponse.json({status: 'error', message: err.message}, {status: 500})
  }
}

