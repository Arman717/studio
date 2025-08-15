import {spawn} from 'child_process';
import {tmpdir} from 'os';
import {basename, join} from 'path';
import {writeFile} from 'fs/promises';

const PYTHON_CMD = process.env.PYTHON ?? 'python3';

function run(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {stdio: ['ignore', 'pipe', 'inherit']});
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
        reject(new Error(`${cmd} exited with code ${code}`));
      }
    });
  });
}

function runPython(args: string[]): Promise<string> {
  return run(PYTHON_CMD, args);
}

export interface GlassResult {
  defectDetected: boolean;
  defectVisualizationDataUri?: string;
  screwStatus: string;
}

export async function analyzeWithGlass(
  cameraFeedDataUri: string,
  modelPath: string,
): Promise<GlassResult> {
  const [, data] = cameraFeedDataUri.split(',');
  const buffer = Buffer.from(data, 'base64');
  const imagePath = join(tmpdir(), `glass-${Date.now()}.png`);
  await writeFile(imagePath, buffer);
  const outputPath = join(tmpdir(), `glass-overlay-${Date.now()}.png`);
  const stdout = await runPython([
    'src/python/analyze_glass.py',
    '--image',
    imagePath,
    '--model',
    modelPath,
    '--output',
    outputPath,
  ]);
  const lines = stdout.trim().split(/\r?\n/);
  return JSON.parse(lines[lines.length - 1]);
}

export async function trainGlass(
  referenceImages: string[],
  backgroundImages?: string[],
): Promise<string> {
  const imagePaths: string[] = [];
  for (const img of referenceImages) {
    const [, data] = img.split(',');
    const buffer = Buffer.from(data, 'base64');
    const path = join(tmpdir(), `glass-ref-${Date.now()}-${Math.random()}.png`);
    await writeFile(path, buffer);
    imagePaths.push(path);
  }
  const backgroundPaths: string[] = [];
  if (backgroundImages) {
    for (const img of backgroundImages) {
      const [, data] = img.split(',');
      const buffer = Buffer.from(data, 'base64');
      const path = join(tmpdir(), `glass-bg-${Date.now()}-${Math.random()}.png`);
      await writeFile(path, buffer);
      backgroundPaths.push(path);
    }
  }
  if (process.env.GCLOUD_PROJECT && process.env.GCS_BUCKET) {
    return await trainGlassGCloud(imagePaths, backgroundPaths);
  }
  const modelPath = join(tmpdir(), `glass-model-${Date.now()}.pth`);
  const args = ['src/python/train_glass.py', '--output', modelPath];
  for (const bg of backgroundPaths) {
    args.push('--background', bg);
  }
  args.push(...imagePaths);
  const stdout = await runPython(args);
  const lines = stdout.trim().split(/\r?\n/);
  const result = JSON.parse(lines[lines.length - 1]);
  return result.modelId;
}

async function trainGlassGCloud(imagePaths: string[], backgroundPaths: string[]): Promise<string> {
  const project = process.env.GCLOUD_PROJECT!;
  const bucket = process.env.GCS_BUCKET!;
  const region = process.env.GCLOUD_REGION ?? 'us-central1';
  const repo = process.env.GCLOUD_REPO;
  if (!repo) {
    throw new Error('GCLOUD_REPO must be set to a Git repository containing train_glass.py');
  }
  const jobId = `glass-${Date.now()}`;
  // Upload training images to the bucket
  for (const p of imagePaths) {
    await run('gsutil', ['cp', p, `gs://${bucket}/${jobId}/${basename(p)}`]);
  }
  for (const p of backgroundPaths) {
    await run('gsutil', ['cp', p, `gs://${bucket}/${jobId}/${basename(p)}`]);
  }
  const gcsImages = imagePaths.map((p) => `gs://${bucket}/${jobId}/${basename(p)}`);
  const gcsBackgrounds = backgroundPaths.map((p) => `gs://${bucket}/${jobId}/${basename(p)}`);
  const outputGcs = `gs://${bucket}/${jobId}/model.pth`;
  const cmdParts = [
    `git clone ${repo} repo`,
    'cd repo',
    'pip install torch torchvision numpy Pillow >/dev/null',
    `python3 src/python/train_glass.py --output ${outputGcs}`,
  ];
  for (const bg of gcsBackgrounds) {
    cmdParts.push('--background', bg);
  }
  cmdParts.push(...gcsImages);
  const containerCmd = cmdParts.join(' ');
  const yaml = `workerPoolSpecs:\n- machineSpec:\n    machineType: n1-standard-8\n    acceleratorType: NVIDIA_TESLA_T4\n    acceleratorCount: 1\n  replicaCount: 1\n  containerSpec:\n    imageUri: gcr.io/deeplearning-platform-release/pytorch-gpu\n    command:\n    - /bin/bash\n    - -c\n    - |\n      ${containerCmd}\n`;
  const yamlPath = join(tmpdir(), `${jobId}.yaml`);
  await writeFile(yamlPath, yaml);
  const jobName = (
    await run('gcloud', [
      'ai',
      'custom-jobs',
      'create',
      '--region',
      region,
      '--project',
      project,
      '--display-name',
      jobId,
      '--config',
      yamlPath,
      '--format=value(name)',
    ])
  ).trim();
  // Poll job status
  while (true) {
    const state = (
      await run('gcloud', [
        'ai',
        'custom-jobs',
        'describe',
        jobName,
        '--region',
        region,
        '--project',
        project,
        '--format=value(state)',
      ])
    ).trim();
    if (state === 'JOB_STATE_SUCCEEDED') break;
    if (state === 'JOB_STATE_FAILED' || state === 'JOB_STATE_CANCELLED') {
      throw new Error(`Training job ${jobName} ended with state ${state}`);
    }
    await new Promise((r) => setTimeout(r, 10000));
  }
  const localModel = join(tmpdir(), `glass-model-${Date.now()}.pth`);
  await run('gsutil', ['cp', outputGcs, localModel]);
  return localModel;
}
