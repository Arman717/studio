# **App Name**: InspectionAI

## Core Features:

- AI Model Training: Reference Mode: Trains an AI model from uploaded images/scans of reference screws, establishing a 'normal' profile for defect comparison using https://github.com/marco-rudolph/cs-flow.
- Automated Defect Detection: Live Inspection: Compares live camera feed and 3D sensor data against the 'normal' AI profile to identify defects, providing a pass/fail result.
- Defect Visualization: Defect Highlighting: Visualizes identified defects via heatmaps or profile overlays on the screw image.
- Intuitive UI/UX: Dashboard Interface: Presents system status, live camera preview, AI detection results, and 3D analysis in a clear and concise manner.
- Reference Mode Tab: Reference Tab: Guides users through capturing reference images/scans and initiates AI model training.
- Inspection Mode Tab: Inspection Tab: Displays live camera feed, real-time AI detection results, and 3D analysis output.
- Results Management: Results Tab: Provides a historical view of inspection results, including images, status, and export functionality.

## Style Guidelines:

- Primary color: Saturated blue (#4285F4). Choosing this color evokes precision, reliability, and technology - aligning well with the technical focus and accuracy required for screw inspection. The slightly bright color is suitable for highlighting the visuals on a darker surface.
- Background color: Dark gray (#37474F), a desaturated version of the primary color, to reduce eye strain and provide contrast for the visual elements.
- Accent color: Yellow (#FFCA28), analogous to blue. To give visibility to crucial elements like alerts and actionable items, it has contrast in brightness.
- Body and headline font: 'Inter' (sans-serif) for a modern, machined look.
- Code font: 'Source Code Pro' for displaying code snippets.
- Use clear, geometric icons representing inspection status and actions.
- Subtle transitions for mode switching and data loading.