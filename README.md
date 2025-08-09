# Firebase Studio

This is a NextJS starter in Firebase Studio.

The AI features now use [CS-Flow](https://github.com/Arman717/cs-flow) for
defect detection. The wrapper scripts in `src/python` automatically clone the
CS-Flow repository if needed. Training produces a `*.pth` model file that is
returned by the training flow. Pass this path to the analysis flow. Analysis
returns a heatmap overlay image highlighting detected anomalies.
Ensure your Python environment has the dependencies listed in CS-Flow's
`requirements.txt` installed. In particular, `torch` and `torchvision` are
required. OpenCV (`opencv-python-headless`), `numpy`, and
[`segment-anything`](https://github.com/facebookresearch/segment-anything) are
also needed for automatic screw segmentation during training and inspection.
Download a SAM checkpoint such as `sam_vit_b.pth` and place it in
`src/python/sam` or provide the path via the `--sam-checkpoint` option. Set the
`PYTHON` environment variable if you want to use a custom Python interpreter.
The scripts will automatically use CUDA if available, falling back to the CPU
otherwise.

The training wrapper patches CS-Flow's `train.py` so AUROC remains a neutral
0.5 when the dataset only contains one class and noisy warnings are suppressed.

Training and analysis logs from the Python scripts are streamed to your
terminal when running the Next.js server so you can monitor progress.

During training, each captured image is automatically segmented to isolate the
screw before it is used to train the CS-Flow model. Inspection snapshots are
segmented the same way prior to analysis so the model focuses on relevant
features of the screw.

To get started, take a look at `src/app/page.tsx`.

After pulling new changes, run `npm install` to ensure all dependencies, such as
`react-webcam`, are installed before starting the dev server.

## Arduino Motor Control
The app can control two motors through an Arduino using the commands defined in
`src/lib/arduino.ts`. Set the `ARDUINO_PORT` environment variable to the serial
port where your Arduino is connected (for example `COM7` on Windows). If this
variable is not provided the app attempts to auto-detect the first port that
looks like an Arduino and falls back to `/dev/ttyACM0`. The firmware for an
ESP8266 board is included in `docs/esp8266-motor-control.ino` and communicates
at 115200 baud, which is the speed used by the server utilities. The new
**Control** tab in the UI lets you send direction and speed commands to both
motors.

The sketch accepts PWM speeds from `0`–`1023` for each motor. By default it
starts both channels at a mid-range value so the motors will turn when a
direction command is sent. Adjust the speed with `SA<value>` or `SB<value>`
commands (for example `SA800`), or modify the initial constants in the sketch.
