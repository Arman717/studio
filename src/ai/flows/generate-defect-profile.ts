'use server';

/**
 * @fileOverview AI flow for generating a defect profile from reference screw images.
 *
 * - generateDefectProfile - A function that handles the defect profile generation process.
 * - GenerateDefectProfileInput - The input type for the generateDefectProfile function.
 * - GenerateDefectProfileOutput - The return type for the generateDefectProfile function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateDefectProfileInputSchema = z.object({
  referenceImages: z
    .array(z.string())
    .describe(
      'An array of images of defect-free screws, as data URIs that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.'      
    ),
});
export type GenerateDefectProfileInput = z.infer<typeof GenerateDefectProfileInputSchema>;

const GenerateDefectProfileOutputSchema = z.object({
  modelId: z.string().describe('The ID of the trained AI model.'),
});
export type GenerateDefectProfileOutput = z.infer<typeof GenerateDefectProfileOutputSchema>;

export async function generateDefectProfile(input: GenerateDefectProfileInput): Promise<GenerateDefectProfileOutput> {
  return generateDefectProfileFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateDefectProfilePrompt',
  input: {schema: GenerateDefectProfileInputSchema},
  output: {schema: GenerateDefectProfileOutputSchema},
  prompt: `You are an AI model trainer. You will receive a set of images of defect-free screws. You will train a model based on these images to create a \"normal\" profile. The model ID will be returned.

Images:{{#each referenceImages}} {{media url=this}} {{/each}}`,
});

const generateDefectProfileFlow = ai.defineFlow(
  {
    name: 'generateDefectProfileFlow',
    inputSchema: GenerateDefectProfileInputSchema,
    outputSchema: GenerateDefectProfileOutputSchema,
  },
  async input => {
    // In a real application, this would involve training an actual AI model.
    // For this example, we'll just return a dummy model ID.
    const {output} = await prompt(input);
    return {
      modelId: 'dummy-model-id-' + Math.random().toString(36).substring(7),
    };
  }
);
