"""CPU-safe command line: inspect retained training, or explicitly launch a fresh GPU run."""
from pathlib import Path
import argparse
import json
from training_port import validate_inputs, execute

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--family', choices=('Mistral','Qwen'), required=True)
    parser.add_argument('--stage', required=True, help='Mistral: participant_article, phase10_qa02_03, full_qa01_04_05; Qwen: staged')
    parser.add_argument('--output-root', type=Path, help='Fresh writable directory; required only for --execute')
    parser.add_argument('--execute', action='store_true', help='Explicitly start GPU training; omission performs CPU file/schema checks only')
    args = parser.parse_args(argv)
    report = validate_inputs(args.package_root, args.family, args.stage)
    print(json.dumps(report, indent=2))
    if args.execute:
        if args.output_root is None:
            parser.error('--execute requires --output-root (a new directory)')
        execute(args.package_root, args.family, args.stage, args.output_root)
    return report

if __name__ == '__main__':
    main()
