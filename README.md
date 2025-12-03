# audio-splitter

Audio separation toolkit for extracting individual speaker tracks from multi-speaker audio recordings.

## Features

- **Diarization-Guided Separation** (Recommended): Extracts speaker-specific audio using timestamps from diarization JSON files
- **Blind Source Separation**: ML-based separation using SepFormer, DPRNN-TasNet, or Conv-TasNet models
- Handles overlapping speech with configurable strategies
- Supports timing preservation or compact output modes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Diarization-Guided Separation (Recommended)

When you have diarization data (JSON file with speaker timestamps), use diarization-guided separation for best results:

```bash
# Basic usage - auto-detects mode when diarization file is provided
python src/separation/main.py \
    --audio interview.wav \
    --diarization diarization.json \
    --output-dir outputs/

# Explicit diarization-guided mode with options
python src/separation/main.py \
    --audio interview.wav \
    --diarization diarization.json \
    --output-dir outputs/ \
    --mode diarization-guided \
    --handle-overlap skip

# Compact mode (no silence gaps)
python src/separation/main.py \
    --audio interview.wav \
    --diarization diarization.json \
    --output-dir outputs/ \
    --compact
```

#### Diarization JSON Format

The diarization JSON should be an array of segments with start/end times and speaker IDs:

```json
[
  {
    "start": 2.98,
    "end": 8.84,
    "speaker": "SPEAKER_00",
    "text": "Hello, how are you?"
  },
  {
    "start": 9.12,
    "end": 15.45,
    "speaker": "SPEAKER_01",
    "text": "I'm doing great, thanks!"
  }
]
```

#### Overlap Handling Strategies

- `--handle-overlap skip`: Leave silence in overlapping regions (default)
- `--handle-overlap mix`: Include overlapping audio for all speakers
- `--handle-overlap both`: Same as mix - include audio for all involved speakers

#### Output Modes

- `--preserve-timing`: Maintain original timestamps with silence gaps (default)
- `--compact`: Concatenate speaker segments without gaps

### Blind Source Separation

When diarization data is not available, use ML-based blind separation:

```bash
# Using SepFormer (recommended for long audio)
python src/separation/main.py \
    --audio interview.wav \
    --output-dir outputs/ \
    --mode blind \
    --model sepformer

# Using DPRNN-TasNet
python src/separation/main.py \
    --audio interview.wav \
    --output-dir outputs/ \
    --mode blind \
    --model dprnn-tasnet
```

## Python API

### Diarization-Guided Separation

```python
from src.separation import DiarizationGuidedSeparator, separate_with_diarization

# Using the class directly
separator = DiarizationGuidedSeparator(
    sample_rate=16000,
    handle_overlap='skip',
    preserve_timing=True,
    min_segment_duration=0.5
)

saved_paths = separator.process_and_save(
    audio_path='interview.wav',
    diarization_path='diarization.json',
    output_dir='outputs/'
)

# Using the convenience function
saved_paths = separate_with_diarization(
    audio_path='interview.wav',
    diarization_path='diarization.json',
    output_dir='outputs/',
    handle_overlap='skip',
    preserve_timing=True
)
```

### Blind Source Separation

```python
from src.separation import SpeechSeparator

separator = SpeechSeparator(model_name='sepformer', device='auto')
separator.load_model()

saved_paths = separator.process_and_save(
    audio_path='interview.wav',
    output_dir='outputs/'
)
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to input audio file | Required |
| `--diarization` | Path to diarization JSON file | None |
| `--output-dir` | Output directory for separated audio | `outputs/` |
| `--mode` | Separation mode: `diarization-guided`, `blind`, or `auto` | `auto` |
| `--handle-overlap` | Overlap strategy: `skip`, `mix`, or `both` | `skip` |
| `--preserve-timing` | Maintain original timestamps | `True` |
| `--compact` | Concatenate segments without gaps | `False` |
| `--min-segment-duration` | Minimum segment duration (seconds) | `0.0` |
| `--model` | ML model for blind mode | `sepformer` |
| `--device` | Device: `cpu`, `cuda`, or `auto` | `auto` |

## Output

Output files are saved to `<output-dir>/<audio-name>_separation/`:

- **Diarization-guided mode**: `SPEAKER_00.wav`, `SPEAKER_01.wav`, etc.
- **Blind mode**: `speaker_00.wav`, `speaker_01.wav`, etc.