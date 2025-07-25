'use server';

/**
 * @fileOverview Analyzes screw defects using camera feed and 3D sensor data.
 *
 * - analyzeScrewDefects - A function that analyzes screw defects.
 * - AnalyzeScrewDefectsInput - The input type for the analyzeScrewDefects function.
 * - AnalyzeScrewDefectsOutput - The return type for the analyzeScrewDefects function.
 */

import {analyzeWithCsFlow} from '@/ai/csflow';
import {z} from 'genkit';

const AnalyzeScrewDefectsInputSchema = z.object({
  cameraFeedDataUri: z
    .string()
    .describe(
      "Live camera feed of the screw, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  sensor3dData: z.string().describe('3D sensor data of the screw.'),
  normalAiProfile: z.string().describe('The normal AI profile of a screw.'),
});

export type AnalyzeScrewDefectsInput = z.infer<typeof AnalyzeScrewDefectsInputSchema>;

const AnalyzeScrewDefectsOutputSchema = z.object({
  defectDetected: z.boolean().describe('Whether a defect was detected in the screw.'),
  defectVisualizationDataUri: z
    .string()
    .optional()
    .describe(
      'Data URI for visualizing the defect, if any, as a data URI that must include a MIME type and use Base64 encoding. Expected format: \\`data:<mimetype>;base64,<encoded_data>\\`.'
    ),
  screwStatus: z.string().describe('The status of the screw (OK or NOK).'),
});

export type AnalyzeScrewDefectsOutput = z.infer<typeof AnalyzeScrewDefectsOutputSchema>;

export async function analyzeScrewDefects(input: AnalyzeScrewDefectsInput): Promise<AnalyzeScrewDefectsOutput> {
  return analyzeWithCsFlow(input.cameraFeedDataUri);
}
