'use server';
/**
 * @fileOverview This file defines a Genkit flow for visualizing screw defects using heatmaps or profile overlays.
 *
 * - visualizeScrewDefects - A function that visualizes defects on a screw image.
 * - VisualizeScrewDefectsInput - The input type for the visualizeScrewDefects function.
 * - VisualizeScrewDefectsOutput - The return type for the visualizeScrewDefects function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const VisualizeScrewDefectsInputSchema = z.object({
  screwImage: z
    .string()
    .describe(
      "A photo of the screw, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  defects: z.string().describe('A description of the defects found on the screw.'),
});
export type VisualizeScrewDefectsInput = z.infer<typeof VisualizeScrewDefectsInputSchema>;

const VisualizeScrewDefectsOutputSchema = z.object({
  visualization: z
    .string()
    .describe(
      'A data URI containing the visualized screw defects, using a heatmap or profile overlay.'
    ),
});
export type VisualizeScrewDefectsOutput = z.infer<typeof VisualizeScrewDefectsOutputSchema>;

export async function visualizeScrewDefects(input: VisualizeScrewDefectsInput): Promise<VisualizeScrewDefectsOutput> {
  return visualizeScrewDefectsFlow(input);
}

const visualizeScrewDefectsPrompt = ai.definePrompt({
  name: 'visualizeScrewDefectsPrompt',
  input: {schema: VisualizeScrewDefectsInputSchema},
  output: {schema: VisualizeScrewDefectsOutputSchema},
  prompt: `You are an AI assistant specializing in visualizing defects on screw images.

You will receive a screw image and a description of the defects found on it.  Your task is to generate a visualization of these defects using either a heatmap or a profile overlay on the screw image.

Defect Description: {{{defects}}}
Screw Image: {{media url=screwImage}}

Ensure the output is a data URI containing the visualized image.
`,
});

const visualizeScrewDefectsFlow = ai.defineFlow(
  {
    name: 'visualizeScrewDefectsFlow',
    inputSchema: VisualizeScrewDefectsInputSchema,
    outputSchema: VisualizeScrewDefectsOutputSchema,
  },
  async input => {
    const {media} = await ai.generate({
      model: 'googleai/gemini-2.0-flash-preview-image-generation',
      prompt: [
        {text: visualizeScrewDefectsPrompt(input).prompt},
        {media: {url: input.screwImage}},
      ],
      config: {
        responseModalities: ['TEXT', 'IMAGE'],
      },
    });
    return {visualization: media!.url!};
  }
);
