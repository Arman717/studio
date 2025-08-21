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
    child.on('close', (code, signal) => {
      if (code === 0) {
        resolve(output);
      } else {
        const reason = code !== null ? `code ${code}` : `signal ${signal}`;
        reject(new Error(`${cmd} exited with ${reason}`));
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
  if (
    process.env.RUNPOD_HOST ||
    process.env.RUNPOD_REPO ||
    process.env.RUNPOD_USER ||
    process.env.RUNPOD_KEY ||
    process.env.RUNPOD_PORT
  ) {
    if (process.env.RUNPOD_HOST && process.env.RUNPOD_REPO) {
      return await trainGlassRunpod(imagePaths, backgroundPaths);
    }
    throw new Error('RUNPOD_HOST and RUNPOD_REPO must be set for RunPod training');
  }
  if (
    process.env.GCLOUD_PROJECT ||
    process.env.GCS_BUCKET ||
    process.env.GCLOUD_REPO
  ) {
    if (
      process.env.GCLOUD_PROJECT &&
      process.env.GCS_BUCKET &&
      process.env.GCLOUD_REPO
    ) {
      return await trainGlassGCloud(imagePaths, backgroundPaths);
    }
    throw new Error(
      'GCLOUD_PROJECT, GCS_BUCKET, and GCLOUD_REPO must all be set for Google Cloud training',
    );
  }
  throw new Error(
    'Cloud training required: configure RunPod or Google Cloud environment variables',
  );
}

async function trainGlassRunpod(imagePaths: string[], backgroundPaths: string[]): Promise<string> {
  const host = process.env.RUNPOD_HOST!;
  const port = process.env.RUNPOD_PORT ?? '22';
  const user = process.env.RUNPOD_USER ?? 'root';
  const key = process.env.RUNPOD_KEY ?? `${process.env.HOME ?? ''}/.ssh/id_ed25519`;
  const repo = process.env.RUNPOD_REPO!;
  const jobId = `glass-${Date.now()}`;
  const remoteDir = `/tmp/${jobId}`;
  const sshBase = ['-i', key, '-p', port, `${user}@${host}`];
  await run('ssh', [...sshBase, 'mkdir', '-p', remoteDir]);
  for (const p of imagePaths) {
    await run('scp', ['-i', key, '-P', port, p, `${user}@${host}:${remoteDir}/${basename(p)}`]);
  }
  for (const p of backgroundPaths) {
    await run('scp', ['-i', key, '-P', port, p, `${user}@${host}:${remoteDir}/${basename(p)}`]);
  }
  const remoteModel = `${remoteDir}/model.pth`;
  const trainArgs = ['python3', 'src/python/train_glass.py', '--output', remoteModel];
  for (const bg of backgroundPaths) {
    trainArgs.push('--background', `${remoteDir}/${basename(bg)}`);
  }
  trainArgs.push(...imagePaths.map((p) => `${remoteDir}/${basename(p)}`));
  const remoteCmd = [
    `git clone ${repo} repo`,
    'cd repo',
    'pip install torch torchvision numpy Pillow >/dev/null',
    trainArgs.join(' '),
  ].join(' && ');
  await run('ssh', [...sshBase, 'bash', '-lc', remoteCmd]);
  const localModel = join(tmpdir(), `glass-model-${Date.now()}.pth`);
  await run('scp', ['-i', key, '-P', port, `${user}@${host}:${remoteModel}`, localModel]);
  const remoteBg = `${remoteModel}.background.png`;
  const localBg = `${localModel}.background.png`;
  await run('scp', ['-i', key, '-P', port, `${user}@${host}:${remoteBg}`, localBg]).catch(() => {
    /* ignore missing background */
  });
  return localModel;
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
}'use server';

/**
 * @fileOverview AI flow for generating a defect profile from reference screw images.
 *
 * - generateDefectProfile - A function that handles the defect profile generation process.
 * - GenerateDefectProfileInput - The input type for the generateDefectProfile function.
 * - GenerateDefectProfileOutput - The return type for the generateDefectProfile function.
 */

import {trainGlass} from '@/ai/glass';
import {z} from 'genkit';

const GenerateDefectProfileInputSchema = z.object({
  referenceImages: z
    .array(z.string())
    .describe(
      'An array of images of defect-free screws, as data URIs that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.'
    ),
  backgroundImages: z
    .array(z.string())
    .optional()
    .describe(
      'Optional background photos (30-100 frames) of the empty rig used to build the background model.'
    ),
});
export type GenerateDefectProfileInput = z.infer<typeof GenerateDefectProfileInputSchema>;

const GenerateDefectProfileOutputSchema = z.object({
  modelId: z.string().describe('The ID of the trained AI model.'),
});
export type GenerateDefectProfileOutput = z.infer<typeof GenerateDefectProfileOutputSchema>;

export async function generateDefectProfile(input: GenerateDefectProfileInput): Promise<GenerateDefectProfileOutput> {
  const modelId = await trainGlass(input.referenceImages, input.backgroundImages);
  return {modelId};
}