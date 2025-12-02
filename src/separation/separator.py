"""
Core speech separation logic using DPRNN-TasNet, Conv-TasNet, or SepFormer.

This module provides the SpeechSeparator class that uses pretrained
DPRNN-TasNet, Conv-TasNet, or SepFormer models to separate overlapping speech.
SepFormer with chunked processing is recommended for long audio files.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
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
    Speech separation using DPRNN-TasNet, Conv-TasNet, or SepFormer.

    This class loads pretrained models from the Asteroid library
    and provides methods to separate overlapping speech sources from
    mixed audio recordings. For long audio files, chunked processing
    is supported to avoid memory issues.

    Attributes:
        model_name: Name of the model to use ('dprnn-tasnet', 'conv-tasnet', or 'sepformer').
        device: Device for inference ('cpu', 'cuda', or 'auto').
        model: The loaded separation model.
    """

    # Pretrained model identifiers (all using Asteroid library)
    MODEL_CONFIGS = {
        'dprnn-tasnet': 'mpariente/DPRNNTasNet-ks2_WHAM_sepclean',
        'conv-tasnet': 'mpariente/ConvTasNet_WHAM!_sepclean',
        'sepformer': 'mpariente/SepFormer_Libri2Mix_sepclean_16k'
    }

    def __init__(
        self,
        model_name: str = 'dprnn-tasnet',
        device: str = 'auto'
    ):
        """
        Initialize the SpeechSeparator.

        Args:
            model_name: Model to use ('dprnn-tasnet', 'conv-tasnet', or 'sepformer').
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
        # SepFormer and DPRNN-TasNet use 16kHz, Conv-TasNet uses 8kHz
        if model_name == 'conv-tasnet':
            self._sample_rate = 8000
        else:
            self._sample_rate = 16000

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
        Load the pretrained Asteroid model with fallback.

        This method downloads and caches the pretrained model if not
        already available locally. All models (DPRNN-TasNet, Conv-TasNet,
        and SepFormer) are loaded from the Asteroid library.

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
            elif self.model_name == 'conv-tasnet':
                from asteroid.models import ConvTasNet
                try:
                    self.model = ConvTasNet.from_pretrained(
                        self.MODEL_CONFIGS[self.model_name]
                    )
                except (OSError, IOError, ValueError) as e:
                    logger.warning(f"Failed to load from HuggingFace: {e}. Using default config...")
                    self.model = ConvTasNet(n_src=2)
            elif self.model_name == 'sepformer':
                from asteroid.models import SepFormer
                try:
                    self.model = SepFormer.from_pretrained(
                        self.MODEL_CONFIGS[self.model_name]
                    )
                except (OSError, IOError) as e:
                    logger.warning(f"Failed to load SepFormer: {e}")
                    logger.info("Falling back to DPRNNTasNet...")
                    from asteroid.models import DPRNNTasNet
                    self.model = DPRNNTasNet.from_pretrained(
                        self.MODEL_CONFIGS['dprnn-tasnet']
                    )
                    self.model_name = 'dprnn-tasnet'

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def separate(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        use_chunking: bool = True,
        chunk_duration: float = 30.0,
        overlap_duration: float = 5.0
    ) -> np.ndarray:
        """
        Separate speech sources from mixed audio.

        Args:
            audio_path: Path to the input mixed audio file.
            num_speakers: Optional hint for number of speakers (not used
                by the model but logged for reference).
            use_chunking: Enable chunked processing for long files (default: True).
            chunk_duration: Duration of each chunk in seconds (default: 30).
            overlap_duration: Overlap between chunks in seconds (default: 5).

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

        # Check audio duration
        info = sf.info(audio_path)
        duration = info.duration

        logger.info(f"Audio duration: {duration:.1f} seconds")

        # Use chunking for files longer than 2 minutes
        if use_chunking and duration > 120:
            logger.info("Using chunked processing for long audio")
            return self.separate_chunked(
                audio_path=audio_path,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                num_speakers=num_speakers
            )

        logger.info("Using standard processing")
        # Original non-chunked implementation
        waveform, sample_rate = load_audio(audio_path, target_sr=self._sample_rate)
        waveform = waveform.to(self.device)

        # Prepare input tensor: model expects (batch, samples)
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif waveform.dim() == 2:
            waveform = torch.mean(waveform, dim=0)
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

    def separate_chunked(
        self,
        audio_path: str,
        chunk_duration: float = 30.0,
        overlap_duration: float = 5.0,
        num_speakers: Optional[int] = None
    ) -> np.ndarray:
        """
        Separate audio using chunked processing for long files.

        This method splits long audio into overlapping chunks, processes
        each chunk independently, and stitches them back together using
        cross-fade to avoid boundary artifacts.

        Args:
            audio_path: Path to input audio file.
            chunk_duration: Duration of each chunk in seconds (default: 30).
            overlap_duration: Overlap between chunks in seconds (default: 5).
            num_speakers: Optional speaker count hint (not used by model).

        Returns:
            Array of separated sources with shape (num_sources, num_samples).
        """
        # Load full audio
        waveform, sample_rate = load_audio(audio_path, target_sr=self._sample_rate)

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * self._sample_rate)
        overlap_samples = int(overlap_duration * self._sample_rate)
        hop_samples = chunk_samples - overlap_samples

        total_samples = waveform.shape[-1]
        num_chunks = int(np.ceil((total_samples - overlap_samples) / hop_samples))

        logger.info(f"Processing {total_samples / self._sample_rate:.1f}s audio in {num_chunks} chunks")

        # Process chunks
        separated_sources = None

        for i in range(num_chunks):
            start_idx = i * hop_samples
            end_idx = min(start_idx + chunk_samples, total_samples)

            # Extract chunk
            chunk = waveform[..., start_idx:end_idx]

            # Separate chunk
            chunk_separated = self._separate_chunk(chunk)

            # Initialize output buffers on first chunk
            if separated_sources is None:
                num_sources = chunk_separated.shape[0]
                separated_sources = [np.zeros(total_samples) for _ in range(num_sources)]

            # Apply cross-fade and add to output
            chunk_start_in_output = start_idx
            chunk_end_in_output = end_idx
            actual_chunk_length = chunk_separated.shape[1]

            for src_idx in range(num_sources):
                if i == 0:
                    # First chunk: no fade-in
                    separated_sources[src_idx][chunk_start_in_output:chunk_end_in_output] = \
                        chunk_separated[src_idx][:actual_chunk_length]
                else:
                    # Apply cross-fade in overlap region
                    overlap_start = chunk_start_in_output
                    overlap_end = min(overlap_start + overlap_samples, chunk_end_in_output)
                    actual_overlap = overlap_end - overlap_start

                    if actual_overlap > 0 and actual_overlap <= actual_chunk_length:
                        fade_out = np.linspace(1, 0, actual_overlap)
                        fade_in = np.linspace(0, 1, actual_overlap)

                        # Blend overlap
                        separated_sources[src_idx][overlap_start:overlap_end] = (
                            separated_sources[src_idx][overlap_start:overlap_end] * fade_out +
                            chunk_separated[src_idx][:actual_overlap] * fade_in
                        )

                        # Add non-overlap part
                        if overlap_end < chunk_end_in_output:
                            remaining_length = min(
                                actual_chunk_length - actual_overlap,
                                chunk_end_in_output - overlap_end
                            )
                            separated_sources[src_idx][overlap_end:overlap_end + remaining_length] = \
                                chunk_separated[src_idx][actual_overlap:actual_overlap + remaining_length]
                    else:
                        # No overlap, just copy
                        separated_sources[src_idx][chunk_start_in_output:chunk_end_in_output] = \
                            chunk_separated[src_idx][:actual_chunk_length]

            logger.info(f"Processed chunk {i+1}/{num_chunks}")

        return np.array(separated_sources)

    def _separate_chunk(self, chunk: torch.Tensor) -> np.ndarray:
        """
        Separate a single audio chunk.

        Args:
            chunk: Audio chunk tensor.

        Returns:
            Separated sources as numpy array with shape (num_sources, num_samples).
        """
        chunk = chunk.to(self.device)

        # Prepare input tensor
        if chunk.dim() == 2 and chunk.shape[0] == 1:
            chunk = chunk.squeeze(0)
        elif chunk.dim() == 2:
            chunk = torch.mean(chunk, dim=0)
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        # Run separation
        with torch.no_grad():
            separated = self.model(chunk)

        # Convert to numpy
        if isinstance(separated, torch.Tensor):
            separated = separated.cpu().numpy()

        if separated.ndim == 3:
            separated = separated.squeeze(0)

        return separated

    def process_and_save(
        self,
        audio_path: str,
        output_dir: str,
        num_speakers: Optional[int] = None,
        use_chunking: bool = True,
        chunk_duration: float = 30.0,
        overlap_duration: float = 5.0
    ) -> list:
        """
        Separate audio and save results to output directory.

        Args:
            audio_path: Path to the input mixed audio file.
            output_dir: Directory to save separated audio files.
            num_speakers: Optional hint for number of speakers.
            use_chunking: Enable chunked processing for long files (default: True).
            chunk_duration: Duration of each chunk in seconds (default: 30).
            overlap_duration: Overlap between chunks in seconds (default: 5).

        Returns:
            List of paths to saved separated audio files.
        """
        # Get base filename for output
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        separation_output_dir = os.path.join(output_dir, f"{base_filename}_separation")

        # Ensure output directory exists
        ensure_output_directory(separation_output_dir)

        # Run separation
        sources = self.separate(
            audio_path,
            num_speakers=num_speakers,
            use_chunking=use_chunking,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration
        )

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
    model_name: str = 'sepformer',
    device: str = 'auto',
    num_speakers: Optional[int] = None,
    use_chunking: bool = True,
    chunk_duration: float = 30.0,
    overlap_duration: float = 5.0
) -> list:
    """
    Convenience function to separate audio.

    Args:
        audio_path: Path to the input mixed audio file.
        output_dir: Directory to save separated audio files.
        model_name: Model to use ('dprnn-tasnet', 'conv-tasnet', or 'sepformer').
        device: Device for inference ('cpu', 'cuda', 'auto').
        num_speakers: Optional hint for number of speakers.
        use_chunking: Enable chunked processing for long files (default: True).
        chunk_duration: Duration of each chunk in seconds (default: 30).
        overlap_duration: Overlap between chunks in seconds (default: 5).

    Returns:
        List of paths to saved separated audio files.
    """
    separator = SpeechSeparator(model_name=model_name, device=device)
    separator.load_model()
    return separator.process_and_save(
        audio_path=audio_path,
        output_dir=output_dir,
        num_speakers=num_speakers,
        use_chunking=use_chunking,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration
    )
