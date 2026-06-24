import argparse
import logging
import random
import sys
from pathlib import Path

from data_processor.conversor import Conversor
from data_processor.leak_generator import RealisticLeakGenerator
from data_processor.leak_processor import LeakProcessor

logger = logging.getLogger("aya.processor")


def main():
    parser = argparse.ArgumentParser(description='Process password leak files for AI training')

    parser.add_argument('input_file', nargs='?', help='Input leak file (email:password format)')
    parser.add_argument('output_file', nargs='?', help='Output JSONL file')
    parser.add_argument('--generate', type=int, metavar='N', help='Generate N synthetic leaks')
    parser.add_argument('--convert', nargs='?', const=True, help='JSONL to JSON converter (specify input file)')
    parser.add_argument('--output', '-o', help='Output file for generated leaks or conversions')
    parser.add_argument('--max-entries', type=int, help='Maximum entries to process')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for --generate (reproducible datasets)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Generation mode
    if args.generate:
        if args.seed is not None:
            random.seed(args.seed)
        output_file = args.output or f"generated_leaks_{args.generate}.txt"
        generator = RealisticLeakGenerator()
        generator.generate_leak_file(output_file, args.generate)
        logger.info("Generated %d synthetic leaks to %s", args.generate, output_file)
        return

    # Conversion mode
    if args.convert is not None:
        input_file = args.convert if isinstance(args.convert, str) else args.input_file

        if not input_file:
            parser.error("Input file required for conversion. Use --convert <input_file> or provide positional argument")

        if not Path(input_file).exists():
            logger.error("Input file '%s' not found", input_file)
            sys.exit(1)

        output_file = args.output or input_file.replace('.jsonl', '.json')

        conversor = Conversor()
        conversor.process_file(input_file, output_file)
        logger.info("Conversion complete: %s -> %s", input_file, output_file)
        return

    # Processing mode
    if not args.input_file or not args.output_file:
        parser.error("Input and output files required, or use --generate or --convert")

    if not Path(args.input_file).exists():
        logger.error("Input file '%s' not found", args.input_file)
        sys.exit(1)

    processor = LeakProcessor()
    processor.process_file(args.input_file, args.output_file, args.max_entries)
    logger.info("Processing complete: %s -> %s", args.input_file, args.output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)