# Firebase Studio

This is a NextJS starter in Firebase Studio.

The AI features now use [GLASS](https://github.com/cqylunlun/GLASS) for defect
detection. The wrapper scripts in `src/python` automatically clone the GLASS
repository if needed. Training produces a `*.pth` model file that is returned by
the training flow. Pass this path to the analysis flow. Analysis returns a
heatmap overlay image highlighting detected anomalies. Ensure your Python
environment has the dependencies listed in GLASS's `requirements.txt`
installed. In particular, `torch`, `torchvision`, `numpy`, and
`opencv-python-headless` are needed for automatic screw segmentation during
training and inspection. Set the `PYTHON` environment variable if you want to
use a custom Python interpreter. The scripts will automatically use CUDA if
available, falling back to the CPU otherwise.

Training now saves a checkpoint even if interrupted and defaults to the
`wideresnet101` backbone for improved performance. You can choose a different
backbone by passing `--backbone` to `train_glass.py` and `analyze_glass.py`.

Training and analysis logs from the Python scripts are streamed to your
terminal when running the Next.js server so you can monitor progress.

During training, each captured image is automatically segmented to isolate the
screw before it is used to train the CS-Flow model. Inspection snapshots are
segmented the same way prior to analysis so the model focuses on relevant
features of the screw.

To get started, take a look at `src/app/page.tsx`.

After pulling new changes, run `npm install` to ensure all dependencies, such as
`react-webcam`, are installed before starting the dev server.

## STM32 Motor Control
The app can control two motors through an STM32 board using the commands defined in
`src/lib/stm32.ts`. Set the `STM32_PORT` environment variable to the serial
port where your board is connected (for example `COM7` on Windows). If this
variable is not provided, the utilities default to `COM7`. Firmware for an
STM32 Nucleo board is included in `docs/stm32-motor-control.ino` and communicates
at 115200 baud, which is the speed used by the server utilities. The new
**Control** tab in the UI lets you send direction and speed commands to both
motors.

The sketch accepts PWM speeds from `0`–`1023` for each motor. By default it
starts both channels at a mid-range value so the motors will turn when a
direction command is sent. Adjust the speed with `SA<value>` or `SB<value>`
commands (for example `SA800`), or modify the initial constants in the sketch.
