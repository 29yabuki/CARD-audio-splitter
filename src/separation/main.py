#!/usr/bin/env python3
"""
CLI entry point for speech separation using DPRNN-TasNet, Conv-TasNet, or SepFormer.

This script provides a command-line interface for separating
overlapping speech from audio recordings using the Asteroid library
or SpeechBrain. SepFormer is recommended for long audio files
as it supports efficient chunked processing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.separation.separator import SpeechSeparator
from src.separation.utils import extract_speaker_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Speech separation using DPRNN-TasNet, Conv-TasNet, or SepFormer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to input audio file'
    )

    parser.add_argument(
        '--diarization',
        type=str,
        default=None,
        help='Path to diarization JSON file (for speaker count hint)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Output directory for separated audio files'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['dprnn-tasnet', 'conv-tasnet', 'sepformer'],
        default='sepformer',
        help='Model to use: sepformer (best for long audio), dprnn-tasnet, conv-tasnet'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device for inference'
    )

    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=30.0,
        help='Duration of each chunk in seconds for long audio processing'
    )

    parser.add_argument(
        '--overlap',
        type=float,
        default=5.0,
        help='Overlap duration between chunks in seconds'
    )

    parser.add_argument(
        '--no-chunking',
        action='store_true',
        help='Disable automatic chunking for long files'
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    print("=" * 80)
    print("Speech Separation Pipeline")
    print(f"Model: {args.model.upper()}")
    print("=" * 80)

    # Validate input audio file
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"Input audio: {args.audio}")
    print(f"Output directory: {args.output_dir}")
    if not args.no_chunking:
        print(f"Chunk duration: {args.chunk_duration}s")
        print(f"Overlap duration: {args.overlap}s")
    else:
        print("Chunking: Disabled")

    # Try to extract speaker count from diarization
    num_speakers = None
    if args.diarization and os.path.exists(args.diarization):
        try:
            num_speakers = extract_speaker_count(args.diarization)
            print(f"Diarization file: {args.diarization}")
            print(f"Detected speakers from diarization: {num_speakers}")
        except Exception as e:
            logger.warning(f"Could not read diarization file: {e}")
            print("Diarization file: Not available")
    else:
        if args.diarization:
            print(f"Diarization file: Not found ({args.diarization})")
        else:
            print("Diarization file: Not provided")

    print("=" * 80)

    # Initialize separator
    try:
        separator = SpeechSeparator(
            model_name=args.model,
            device=args.device
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load model
    try:
        separator.load_model()
    except RuntimeError as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Run separation
    try:
        saved_paths = separator.process_and_save(
            audio_path=args.audio,
            output_dir=args.output_dir,
            num_speakers=num_speakers,
            use_chunking=not args.no_chunking,
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap
        )
    except Exception as e:
        logger.error(f"Separation failed: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("SEPARATION COMPLETE")
    print("=" * 80)
    print(f"Separated {len(saved_paths)} audio sources:")
    for path in saved_paths:
        print(f"  - {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
