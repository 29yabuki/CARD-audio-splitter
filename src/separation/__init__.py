"""
Speech Separation Module.

This module provides functionality to separate overlapping speech sources
from mixed audio recordings using:
1. Diarization-guided extraction (recommended when diarization data is available)
2. Blind source separation using DPRNN-TasNet, Conv-TasNet, or SepFormer
"""

from .separator import SpeechSeparator
from .diarization_separator import DiarizationGuidedSeparator, separate_with_diarization
from .utils import load_audio, save_separated_audio, extract_speaker_count

__all__ = [
    'SpeechSeparator',
    'DiarizationGuidedSeparator',
    'separate_with_diarization',
    'load_audio',
    'save_separated_audio',
    'extract_speaker_count'
]
