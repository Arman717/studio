# Firebase Studio

This is a NextJS starter in Firebase Studio.

The AI features now use [CS-Flow](https://github.com/Arman717/cs-flow) for
defect detection. The wrapper scripts in `src/python` will automatically clone
the CS-Flow repository if needed. Training produces a `*.pth` model file that
is returned by the training flow. Pass this path to the analysis flow.
Ensure your Python environment has the dependencies listed in CS-Flow's
`requirements.txt` installed. In particular, `torch` and `torchvision` are
required. Set the `PYTHON` environment variable if you want to use a custom
Python interpreter.

To get started, take a look at `src/app/page.tsx`.
