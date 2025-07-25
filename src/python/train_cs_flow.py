#!/usr/bin/env python3
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cs-flow model")
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('images', nargs='*', help='Training images')
    args = parser.parse_args()

    # TODO: integrate real cs-flow training
    with open(args.output, 'w') as f:
        f.write('placeholder')

    print(json.dumps({'modelId': args.output}))


if __name__ == '__main__':
    main()
