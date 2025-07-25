import { config } from 'dotenv';
config();

import '@/ai/flows/visualize-screw-defects.ts';
import '@/ai/flows/generate-defect-profile.ts';
import '@/ai/flows/analyze-screw-defects.ts';