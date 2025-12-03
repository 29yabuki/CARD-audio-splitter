#!/usr/bin/env python3
"""
CLI entry point for speech separation.

This script provides a command-line interface for separating
overlapping speech from audio recordings using either:
1. Diarization-guided extraction (recommended when diarization JSON is available)
2. Blind source separation using DPRNN-TasNet, Conv-TasNet, or SepFormer
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.separation.separator import SpeechSeparator
from src.separation.diarization_separator import DiarizationGuidedSeparator
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
        description='Speech separation using diarization-guided extraction or blind source separation',
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
        help='Path to diarization JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Output directory for separated audio files'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['diarization-guided', 'blind', 'auto'],
        default='auto',
        help='Separation mode: diarization-guided (uses timestamps), blind (ML-based), '
             'or auto (uses diarization-guided if diarization file provided)'
    )

    # Diarization-guided mode options
    parser.add_argument(
        '--handle-overlap',
        type=str,
        choices=['skip', 'mix', 'both'],
        default='skip',
        help='Strategy for overlapping speech: skip (silence), mix (include for all), '
             'both (include audio for all involved speakers)'
    )

    parser.add_argument(
        '--preserve-timing',
        action='store_true',
        default=True,
        help='Maintain original timestamps with silence gaps (default: True)'
    )

    parser.add_argument(
        '--compact',
        action='store_true',
        help='Concatenate speaker segments without gaps (overrides --preserve-timing)'
    )

    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=0.0,
        help='Minimum segment duration in seconds to include'
    )

    # Blind separation mode options
    parser.add_argument(
        '--model',
        type=str,
        choices=['dprnn-tasnet', 'conv-tasnet', 'sepformer'],
        default='sepformer',
        help='Model for blind separation: sepformer (best for long audio), dprnn-tasnet, conv-tasnet'
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
        help='Duration of each chunk in seconds for long audio processing (blind mode)'
    )

    parser.add_argument(
        '--overlap',
        type=float,
        default=5.0,
        help='Overlap duration between chunks in seconds (blind mode)'
    )

    parser.add_argument(
        '--no-chunking',
        action='store_true',
        help='Disable automatic chunking for long files (blind mode)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Validate input audio file
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    # Determine separation mode
    mode = args.mode
    if mode == 'auto':
        if args.diarization and os.path.exists(args.diarization):
            mode = 'diarization-guided'
            logger.info("Auto mode: Using diarization-guided separation (diarization file provided)")
        else:
            mode = 'blind'
            logger.info("Auto mode: Using blind separation (no diarization file)")

    # Check for diarization file when using diarization-guided mode
    if mode == 'diarization-guided':
        if not args.diarization:
            logger.error("Diarization-guided mode requires --diarization file")
            sys.exit(1)
        if not os.path.exists(args.diarization):
            logger.error(f"Diarization file not found: {args.diarization}")
            sys.exit(1)

    print("=" * 80)
    print("Speech Separation Pipeline")
    print(f"Mode: {mode.upper()}")
    print("=" * 80)

    print(f"Input audio: {args.audio}")
    print(f"Output directory: {args.output_dir}")

    if mode == 'diarization-guided':
        # Diarization-guided separation
        print(f"Diarization file: {args.diarization}")
        print(f"Overlap handling: {args.handle_overlap}")

        preserve_timing = args.preserve_timing and not args.compact
        print(f"Preserve timing: {preserve_timing}")
        if args.min_segment_duration > 0:
            print(f"Min segment duration: {args.min_segment_duration}s")

        print("=" * 80)

        try:
            separator = DiarizationGuidedSeparator(
                sample_rate=16000,
                handle_overlap=args.handle_overlap,
                preserve_timing=preserve_timing,
                min_segment_duration=args.min_segment_duration
            )

            saved_paths = separator.process_and_save(
                audio_path=args.audio,
                diarization_path=args.diarization,
                output_dir=args.output_dir
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Invalid diarization file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            sys.exit(1)

    else:
        # Blind source separation
        print(f"Model: {args.model.upper()}")
        if not args.no_chunking:
            print(f"Chunk duration: {args.chunk_duration}s")
            print(f"Overlap duration: {args.overlap}s")
        else:
            print("Chunking: Disabled")

        # Try to extract speaker count from diarization if provided
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
