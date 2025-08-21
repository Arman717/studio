'use server';

import { spawn } from 'child_process';
import { tmpdir } from 'os';
import { basename, join } from 'path';
import { writeFile } from 'fs/promises';
import { z } from 'genkit';

const PYTHON_CMD = process.env.PYTHON ?? 'python3';

/* ------------------------------- helpers -------------------------------- */

function run(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'inherit'] });
    let output = '';
    child.stdout.on('data', (d) => {
      process.stdout.write(d);
      output += d.toString();
    });
    child.on('error', reject);
    child.on('close', (code, signal) => {
      if (code === 0) resolve(output);
      else reject(new Error(`${cmd} exited with ${code !== null ? `code ${code}` : `signal ${signal}`}`));
    });
  });
}

function runPython(args: string[]): Promise<string> {
  return run(PYTHON_CMD, args);
}

/** Parse repo env that may include a branch suffix, e.g. "...git#runpod-setup". */
function parseRepoAndBranch(repoEnv?: string): { repo: string; branch?: string } {
  if (!repoEnv) return { repo: '' };
  const idx = repoEnv.indexOf('#');
  if (idx === -1) return { repo: repoEnv };
  return { repo: repoEnv.slice(0, idx), branch: repoEnv.slice(idx + 1) };
}

/* ----------------------------- inference path --------------------------- */

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

/* ----------------------------- training path ---------------------------- */

export async function trainGlass(
  referenceImages: string[],
  backgroundImages?: string[],
): Promise<string> {
  // Save incoming data URIs to temp files
  const imagePaths: string[] = [];
  for (const img of referenceImages) {
    const [, data] = img.split(',');
    const buffer = Buffer.from(data, 'base64');
    const p = join(tmpdir(), `glass-ref-${Date.now()}-${Math.random()}.png`);
    await writeFile(p, buffer);
    imagePaths.push(p);
  }

  const backgroundPaths: string[] = [];
  if (backgroundImages?.length) {
    for (const img of backgroundImages) {
      const [, data] = img.split(',');
      const buffer = Buffer.from(data, 'base64');
      const p = join(tmpdir(), `glass-bg-${Date.now()}-${Math.random()}.png`);
      await writeFile(p, buffer);
      backgroundPaths.push(p);
    }
  }

  // Prefer RunPod if any hint exists
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

  // Otherwise try Google Cloud
  if (process.env.GCLOUD_PROJECT || process.env.GCS_BUCKET || process.env.GCLOUD_REPO) {
    if (process.env.GCLOUD_PROJECT && process.env.GCS_BUCKET && process.env.GCLOUD_REPO) {
      return await trainGlassGCloud(imagePaths, backgroundPaths);
    }
    throw new Error('GCLOUD_PROJECT, GCS_BUCKET, and GCLOUD_REPO must all be set for Google Cloud training');
  }

  throw new Error('Cloud training required: configure RunPod or Google Cloud environment variables');
}

async function trainGlassRunpod(imagePaths: string[], backgroundPaths: string[]): Promise<string> {
  const host = process.env.RUNPOD_HOST!;
  const port = process.env.RUNPOD_PORT ?? '22';
  const user = process.env.RUNPOD_USER ?? 'root';
  const key = process.env.RUNPOD_KEY ?? `${process.env.HOME ?? ''}/.ssh/id_ed25519`;
  const { repo, branch } = parseRepoAndBranch(process.env.RUNPOD_REPO);

  const jobId = `glass-${Date.now()}`;
  const remoteDir = `/tmp/${jobId}`;
  const sshBase = ['-i', key, '-p', port, `${user}@${host}`];

  // 1) Prepare remote workspace
  await run('ssh', [...sshBase, 'mkdir', '-p', remoteDir]);

  // 2) Upload images
  for (const p of imagePaths) {
    await run('scp', ['-i', key, '-P', port, p, `${user}@${host}:${remoteDir}/${basename(p)}`]);
  }
  for (const p of backgroundPaths) {
    await run('scp', ['-i', key, '-P', port, p, `${user}@${host}:${remoteDir}/${basename(p)}`]);
  }

  // 3) Build and run remote training command in isolated dir
  const remoteModel = `${remoteDir}/model.pth`;
  const trainArgs = ['python3', 'src/python/train_glass.py', '--output', remoteModel];

  for (const bg of backgroundPaths) {
    trainArgs.push('--background', `${remoteDir}/${basename(bg)}`);
  }
  trainArgs.push(...imagePaths.map((p) => `${remoteDir}/${basename(p)}`));

  const cloneOrUpdate =
    branch && branch.length > 0
      ? `if [ -d repo/.git ]; then git -C repo fetch --all && git -C repo checkout ${branch} && git -C repo reset --hard origin/${branch}; else git clone --single-branch -b ${branch} ${repo} repo; fi`
      : `if [ -d repo/.git ]; then git -C repo fetch --all && git -C repo checkout -f && git -C repo reset --hard origin/$(git -C repo rev-parse --abbrev-ref HEAD); else git clone ${repo} repo; fi`;

  const remoteCmd = [
    'set -e',
    `cd ${remoteDir}`,
    'command -v git >/dev/null 2>&1 || (apt-get update -y && apt-get install -y git)',
    'command -v python3 >/dev/null 2>&1 || apt-get install -y python3 python3-pip',
    cloneOrUpdate,
    'cd repo',
    'pip install -q --upgrade pip',
    'pip install -q torch torchvision numpy Pillow',
    trainArgs.join(' '),
  ].join(' && ');

  await run('ssh', [...sshBase, 'bash', '-lc', remoteCmd]);

  // 4) Download outputs
  const localModel = join(tmpdir(), `glass-model-${Date.now()}.pth`);
  await run('scp', ['-i', key, '-P', port, `${user}@${host}:${remoteModel}`, localModel]);

  const remoteBg = `${remoteModel}.background.png`;
  const localBg = `${localModel}.background.png`;
  await run('scp', ['-i', key, '-P', port, `${user}@${host}:${remoteBg}`, localBg]).catch(() => {
    /* optional */
  });

  return localModel;
}

async function trainGlassGCloud(imagePaths: string[], backgroundPaths: string[]): Promise<string> {
  const project = process.env.GCLOUD_PROJECT!;
  const bucket = process.env.GCS_BUCKET!;
  const region = process.env.GCLOUD_REGION ?? 'us-central1';
  const { repo, branch } = parseRepoAndBranch(process.env.GCLOUD_REPO);

  const jobId = `glass-${Date.now()}`;

  // Upload data
  for (const p of imagePaths) {
    await run('gsutil', ['cp', p, `gs://${bucket}/${jobId}/${basename(p)}`]);
  }
  for (const p of backgroundPaths) {
    await run('gsutil', ['cp', p, `gs://${bucket}/${jobId}/${basename(p)}`]);
  }

  const gcsImages = imagePaths.map((p) => `gs://${bucket}/${jobId}/${basename(p)}`);
  const gcsBackgrounds = backgroundPaths.map((p) => `gs://${bucket}/${jobId}/${basename(p)}`);
  const outputGcs = `gs://${bucket}/${jobId}/model.pth`;

  const cloneCmd =
    branch && branch.length > 0
      ? `git clone --single-branch -b ${branch} ${repo} repo`
      : `git clone ${repo} repo`;

  const cmdParts = [
    'set -e',
    cloneCmd,
    'cd repo',
    'pip install -q --upgrade pip',
    'pip install -q torch torchvision numpy Pillow',
    `python3 src/python/train_glass.py --output ${outputGcs}`,
  ];
  for (const bg of gcsBackgrounds) cmdParts.push('--background', bg);
  cmdParts.push(...gcsImages);

  const containerCmd = cmdParts.join(' && ');
  const yaml =
    `workerPoolSpecs:
- machineSpec:
    machineType: n1-standard-8
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: gcr.io/deeplearning-platform-release/pytorch-gpu
    command:
    - /bin/bash
    - -c
    - |
      ${containerCmd}
`;

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

  // Poll
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
    await new Promise((r) => setTimeout(r, 10_000));
  }

  const localModel = join(tmpdir(), `glass-model-${Date.now()}.pth`);
  await run('gsutil', ['cp', outputGcs, localModel]);
  return localModel;
}

/* ---------------------- public server action for UI ---------------------- */

const GenerateDefectProfileInputSchema = z.object({
  referenceImages: z
    .array(z.string())
    .describe("An array of defect-free screw images as data URIs (format: 'data:<mimetype>;base64,<data>')."),
  backgroundImages: z
    .array(z.string())
    .optional()
    .describe('Optional background frames (30–100) of the empty rig.'),
});
export type GenerateDefectProfileInput = z.infer<typeof GenerateDefectProfileInputSchema>;

const GenerateDefectProfileOutputSchema = z.object({
  modelId: z.string().describe('The ID/path of the trained AI model.'),
});
export type GenerateDefectProfileOutput = z.infer<typeof GenerateDefectProfileOutputSchema>;

export async function generateDefectProfile(
  input: GenerateDefectProfileInput,
): Promise<GenerateDefectProfileOutput> {
  const modelId = await trainGlass(input.referenceImages, input.backgroundImages);
  return { modelId };
}
