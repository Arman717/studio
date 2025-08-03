# Firebase Studio

This is a NextJS starter in Firebase Studio.

The AI features now use [CS-Flow](https://github.com/Arman717/cs-flow) for
defect detection. The wrapper scripts in `src/python` automatically clone the
CS-Flow repository if needed. Training produces a `*.pth` model file that is
returned by the training flow. Pass this path to the analysis flow. Analysis
returns a heatmap overlay image highlighting detected anomalies.
Ensure your Python environment has the dependencies listed in CS-Flow's
`requirements.txt` installed. In particular, `torch` and `torchvision` are
required. OpenCV (`opencv-python-headless`) and `numpy` are also needed for
automatic screw segmentation during training. Set the `PYTHON` environment
variable if you want to use a custom Python interpreter. The scripts will
automatically use CUDA if available, falling back to the CPU otherwise.

The training wrapper patches CS-Flow's `train.py` so AUROC remains a neutral
0.5 when the dataset only contains one class and noisy warnings are suppressed.

Training and analysis logs from the Python scripts are streamed to your
terminal when running the Next.js server so you can monitor progress.

During training, each captured image is automatically segmented to isolate the
screw before it is used to train the CS-Flow model. This helps ensure the model
focuses on relevant features of the screw.

To get started, take a look at `src/app/page.tsx`.

After pulling new changes, run `npm install` to ensure all dependencies, such as
`react-webcam`, are installed before starting the dev server.

## Arduino Motor Control
The app can control two motors through an Arduino using the commands defined in
`src/lib/arduino.ts`. Set the `ARDUINO_PORT` environment variable to the serial
port where your Arduino is connected (for example `COM6` on Windows). If this
variable is not provided the app attempts to auto-detect the first port that
looks like an Arduino and falls back to `/dev/ttyACM0`. The new **Control** tab
in the UI lets you send direction and speed commands to both motors.
