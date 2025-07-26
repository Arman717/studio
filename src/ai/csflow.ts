import {spawn} from 'child_process';
import {tmpdir} from 'os';
import {join} from 'path';
import {writeFile} from 'fs/promises';


const PYTHON_CMD = process.env.PYTHON ?? 'python3';

function runPython(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_CMD, args, {stdio: ['ignore', 'pipe', 'inherit']});
    let output = '';
    child.stdout.on('data', (d) => {
      process.stdout.write(d);
      output += d.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Python exited with code ${code}`));
      }
    });
  });
}

export interface CsFlowResult {
  defectDetected: boolean;
  defectVisualizationDataUri?: string;
  screwStatus: string;
}

export async function analyzeWithCsFlow(
  cameraFeedDataUri: string,
  modelPath: string,
): Promise<CsFlowResult> {
  const [, data] = cameraFeedDataUri.split(',');
  const buffer = Buffer.from(data, 'base64');
  const imagePath = join(tmpdir(), `csflow-${Date.now()}.png`);
  await writeFile(imagePath, buffer);
  const stdout = await runPython([
    'src/python/analyze_cs_flow.py',
    '--image',
    imagePath,
    '--model',
    modelPath,
  ]);
  const lines = stdout.trim().split(/\r?\n/);
  return JSON.parse(lines[lines.length - 1]);
}

export async function trainCsFlow(referenceImages: string[]): Promise<string> {
  const imagePaths: string[] = [];
  for (const img of referenceImages) {
    const [, data] = img.split(',');
    const buffer = Buffer.from(data, 'base64');
    const path = join(tmpdir(), `csflow-ref-${Date.now()}-${Math.random()}.png`);
    await writeFile(path, buffer);
    imagePaths.push(path);
  }
  const modelPath = join(tmpdir(), `csflow-model-${Date.now()}.pth`);
  const args = [
    'src/python/train_cs_flow.py',
    '--output',
    modelPath,
    ...imagePaths,
  ];
  const stdout = await runPython(args);
  const lines = stdout.trim().split(/\r?\n/);
  const result = JSON.parse(lines[lines.length - 1]);
  return result.modelId;
}
