"""
Speech Separation Module using DPRNN-TasNet from Asteroid library.

This module provides functionality to separate overlapping speech sources
from mixed audio recordings using deep learning models.
"""

from .separator import SpeechSeparator
from .utils import load_audio, save_separated_audio, extract_speaker_count

__all__ = ['SpeechSeparator', 'load_audio', 'save_separated_audio', 'extract_speaker_count']
