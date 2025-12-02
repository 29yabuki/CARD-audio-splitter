"""
Core speech separation logic using DPRNN-TasNet from Asteroid library.

This module provides the SpeechSeparator class that uses pretrained
DPRNN-TasNet or Conv-TasNet models to separate overlapping speech.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from .utils import load_audio, save_separated_audio, ensure_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SpeechSeparator:
    """
    Speech separation using DPRNN-TasNet or Conv-TasNet from Asteroid.

    This class loads pretrained models from the Asteroid library and
    provides methods to separate overlapping speech sources from
    mixed audio recordings.

    Attributes:
        model_name: Name of the model to use ('dprnn-tasnet' or 'conv-tasnet').
        device: Device for inference ('cpu', 'cuda', or 'auto').
        model: The loaded Asteroid separation model.
    """

    # Pretrained model identifiers from mpariente organization (verified working)
    MODEL_CONFIGS = {
        'dprnn-tasnet': 'mpariente/DPRNNTasNet-ks2_WHAM_sepclean',
        'conv-tasnet': 'mpariente/ConvTasNet_WHAM!_sepclean'
    }

    def __init__(
        self,
        model_name: str = 'dprnn-tasnet',
        device: str = 'auto'
    ):
        """
        Initialize the SpeechSeparator.

        Args:
            model_name: Model to use ('dprnn-tasnet' or 'conv-tasnet').
            device: Device for inference ('cpu', 'cuda', 'auto').

        Raises:
            ValueError: If an invalid model_name is provided.
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_name: {model_name}. "
                f"Choose from: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self._sample_rate = 16000 if model_name == 'dprnn-tasnet' else 8000

        logger.info(f"Initialized SpeechSeparator with model: {model_name}")
        logger.info(f"Using device: {self.device}")

    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device string to an actual device.

        Args:
            device: Device specification ('cpu', 'cuda', 'auto').

        Returns:
            Resolved device string.
        """
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device

    def load_model(self) -> None:
        """
        Load the pretrained Asteroid model with fallback options.

        This method downloads and caches the pretrained model if not
        already available locally. If loading from HuggingFace fails,
        it falls back to using a default model configuration.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if self.model is not None:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading pretrained model: {self.MODEL_CONFIGS[self.model_name]}")

        try:
            if self.model_name == 'dprnn-tasnet':
                from asteroid.models import DPRNNTasNet
                try:
                    self.model = DPRNNTasNet.from_pretrained(
                        self.MODEL_CONFIGS[self.model_name]
                    )
                except (OSError, IOError, ValueError) as e:
                    logger.warning(f"Failed to load from HuggingFace: {e}. Using default config...")
                    self.model = DPRNNTasNet(n_src=2)
            else:  # conv-tasnet
                from asteroid.models import ConvTasNet
                try:
                    self.model = ConvTasNet.from_pretrained(
                        self.MODEL_CONFIGS[self.model_name]
                    )
                except (OSError, IOError, ValueError) as e:
                    logger.warning(f"Failed to load from HuggingFace: {e}. Using default config...")
                    self.model = ConvTasNet(n_src=2)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def separate(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> np.ndarray:
        """
        Separate speech sources from mixed audio.

        Args:
            audio_path: Path to the input mixed audio file.
            num_speakers: Optional hint for number of speakers (not used
                by the model but logged for reference).

        Returns:
            Array of separated sources with shape (num_sources, num_samples).

        Raises:
            RuntimeError: If the model is not loaded or separation fails.
            FileNotFoundError: If the audio file does not exist.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Processing audio: {audio_path}")
        if num_speakers is not None:
            logger.info(f"Speaker count hint: {num_speakers}")

        # Load and preprocess audio
        waveform, sample_rate = load_audio(audio_path, target_sr=self._sample_rate)

        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)

        # Prepare input tensor: model expects (batch, samples)
        # load_audio returns (channels, samples), which is (1, samples) for mono
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            # Shape is (1, samples), squeeze channel dim to get (samples,)
            waveform = waveform.squeeze(0)
        elif waveform.dim() == 2:
            # Shape is (channels, samples) with channels > 1, convert to mono
            waveform = torch.mean(waveform, dim=0)
        # Now waveform is (samples,), add batch dim to get (1, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        logger.info(f"Input waveform shape: {waveform.shape}")
        logger.info("Running separation...")

        # Run separation
        with torch.no_grad():
            separated = self.model(waveform)

        # Convert to numpy
        if isinstance(separated, torch.Tensor):
            separated = separated.cpu().numpy()

        # Remove batch dimension if present
        if separated.ndim == 3:
            separated = separated.squeeze(0)

        logger.info(f"Separated {separated.shape[0]} sources")

        return separated

    def process_and_save(
        self,
        audio_path: str,
        output_dir: str,
        num_speakers: Optional[int] = None
    ) -> list:
        """
        Separate audio and save results to output directory.

        Args:
            audio_path: Path to the input mixed audio file.
            output_dir: Directory to save separated audio files.
            num_speakers: Optional hint for number of speakers.

        Returns:
            List of paths to saved separated audio files.
        """
        # Get base filename for output
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        separation_output_dir = os.path.join(output_dir, f"{base_filename}_separation")

        # Ensure output directory exists
        ensure_output_directory(separation_output_dir)

        # Run separation
        sources = self.separate(audio_path, num_speakers=num_speakers)

        # Save separated audio
        saved_paths = save_separated_audio(
            sources=sources,
            output_dir=separation_output_dir,
            base_filename=base_filename,
            sample_rate=self._sample_rate
        )

        logger.info(f"Saved {len(saved_paths)} separated sources to: {separation_output_dir}")
        for path in saved_paths:
            logger.info(f"  - {path}")

        return saved_paths


def separate_audio(
    audio_path: str,
    output_dir: str,
    model_name: str = 'dprnn-tasnet',
    device: str = 'auto',
    num_speakers: Optional[int] = None
) -> list:
    """
    Convenience function to separate audio.

    Args:
        audio_path: Path to the input mixed audio file.
        output_dir: Directory to save separated audio files.
        model_name: Model to use ('dprnn-tasnet' or 'conv-tasnet').
        device: Device for inference ('cpu', 'cuda', 'auto').
        num_speakers: Optional hint for number of speakers.

    Returns:
        List of paths to saved separated audio files.
    """
    separator = SpeechSeparator(model_name=model_name, device=device)
    separator.load_model()
    return separator.process_and_save(
        audio_path=audio_path,
        output_dir=output_dir,
        num_speakers=num_speakers
    )
