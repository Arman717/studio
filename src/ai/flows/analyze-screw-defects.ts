'use server';

/**
 * @fileOverview Analyzes screw defects using camera feed and 3D sensor data.
 *
 * - analyzeScrewDefects - A function that analyzes screw defects.
 * - AnalyzeScrewDefectsInput - The input type for the analyzeScrewDefects function.
 * - AnalyzeScrewDefectsOutput - The return type for the analyzeScrewDefects function.
 */

import {ai} from '@/ai/genkit';
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
      'Data URI for visualizing the defect, if any, as a data URI that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.' // Optional
    ),
  screwStatus: z.string().describe('The status of the screw (OK or NOK).'),
});

export type AnalyzeScrewDefectsOutput = z.infer<typeof AnalyzeScrewDefectsOutputSchema>;

export async function analyzeScrewDefects(input: AnalyzeScrewDefectsInput): Promise<AnalyzeScrewDefectsOutput> {
  return analyzeScrewDefectsFlow(input);
}

const analyzeScrewDefectsPrompt = ai.definePrompt({
  name: 'analyzeScrewDefectsPrompt',
  input: {schema: AnalyzeScrewDefectsInputSchema},
  output: {schema: AnalyzeScrewDefectsOutputSchema},
  prompt: `You are an AI expert in quality control, specializing in screw defect analysis.

You will analyze the provided camera feed and 3D sensor data of a screw against a "normal" AI profile to identify any defects.

Based on your analysis, determine if the screw has any defects (e.g., cracks, dents, form deviations, height errors, thread depth issues).

Provide a defectDetected boolean indicating whether a defect was found.
If a defect is detected, generate a visualization (heatmap or profile overlay) highlighting the defect on the screw image. If a defect is found, the defectVisualizationDataUri should be the data URL of the screw with the defect highlighted; otherwise, leave it blank.
Finally, determine the screwStatus, marking it as "OK" if no defects are found, or "NOK" if defects are present.

Here is the information about the screw:

Camera Feed: {{media url=cameraFeedDataUri}}
3D Sensor Data: {{{sensor3dData}}}
Normal AI Profile: {{{normalAiProfile}}}`,
});

const analyzeScrewDefectsFlow = ai.defineFlow(
  {
    name: 'analyzeScrewDefectsFlow',
    inputSchema: AnalyzeScrewDefectsInputSchema,
    outputSchema: AnalyzeScrewDefectsOutputSchema,
  },
  async input => {
    const {output} = await analyzeScrewDefectsPrompt(input);
    return output!;
  }
);
