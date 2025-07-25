#!/usr/bin/env python3
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze screw image with cs-flow")
    parser.add_argument('--image', required=True, help='Path to screw image')
    args = parser.parse_args()

    # TODO: integrate real cs-flow model inference
    result = {
        "defectDetected": False,
        "defectVisualizationDataUri": "",
        "screwStatus": "OK"
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
