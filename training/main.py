import argparse
import logging
import sys
from pathlib import Path
from data_processor.leak_processor import LeakProcessor
from data_processor.leak_generator import RealisticLeakGenerator


def main():
    parser = argparse.ArgumentParser(description='Process password leak files for AI training')
    
    parser.add_argument('input_file', nargs='?', help='Input leak file (email:password format)')
    parser.add_argument('output_file', nargs='?', help='Output JSONL file')
    parser.add_argument('--generate', type=int, metavar='N', help='Generate N synthetic leaks')
    parser.add_argument('--output', '-o', help='Output file for generated leaks')
    parser.add_argument('--max-entries', type=int, help='Maximum entries to process')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generation mode
    if args.generate:
        output_file = args.output or f"generated_leaks_{args.generate}.txt"
        generator = RealisticLeakGenerator()
        generator.generate_leak_file(output_file, args.generate)
        print(f"Generated {args.generate} synthetic leaks to {output_file}")
        return
    
    # Processing mode
    if not args.input_file or not args.output_file:
        parser.error("Input and output files required, or use --generate")
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    processor = LeakProcessor()
    processor.process_file(args.input_file, args.output_file, args.max_entries)
    print(f"Processing complete: {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)