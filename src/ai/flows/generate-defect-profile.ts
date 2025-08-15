'use server';

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
