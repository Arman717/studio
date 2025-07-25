import {execFile} from 'child_process';
import {promisify} from 'util';
import {tmpdir} from 'os';
import {join} from 'path';
import {writeFile} from 'fs/promises';

const execFileAsync = promisify(execFile);

export interface CsFlowResult {
  defectDetected: boolean;
  defectVisualizationDataUri?: string;
  screwStatus: string;
}

export async function analyzeWithCsFlow(cameraFeedDataUri: string): Promise<CsFlowResult> {
  const [, data] = cameraFeedDataUri.split(',');
  const buffer = Buffer.from(data, 'base64');
  const imagePath = join(tmpdir(), `csflow-${Date.now()}.png`);
  await writeFile(imagePath, buffer);
  const {stdout} = await execFileAsync('python3', [
    'src/python/analyze_cs_flow.py',
    '--image',
    imagePath,
    '--model',
    'model.pth',
  ]);
  return JSON.parse(stdout.trim());
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
  const {stdout} = await execFileAsync('python3', args);
  const result = JSON.parse(stdout.trim());
  return result.modelId;
}
