"""
NVIDIA NeMo-based Speaker Diarization for WhisperX

This module provides speaker diarization using NVIDIA NeMo's ClusteringDiarizer
as an alternative to pyannote.audio.

Requirements:
    pip install nemo_toolkit[asr]
    # Or for specific version:
    # pip install nemo_toolkit[asr]==1.23.0

Note: NeMo requires PyTorch and other dependencies. See NeMo documentation for details.
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Union
from pathlib import Path
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.schema import TranscriptionResult, AlignedTranscriptionResult
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class DiarizationPipeline:
    """
    NeMo-based speaker diarization pipeline.
    
    This class provides a drop-in replacement for pyannote-based diarization
    using NVIDIA NeMo's ClusteringDiarizer.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        vad_model: str = "vad_multilingual_marblenet",
        speaker_model: str = "titanet_large",
        window_length_in_sec: float = 1.5,
        shift_length_in_sec: float = 0.75,
    ):
        """
        Initialize NeMo diarization pipeline.
        
        Args:
            model_name: Not used for NeMo (kept for compatibility)
            use_auth_token: Not used for NeMo (kept for compatibility)
            device: Device to run models on ('cpu' or 'cuda')
            vad_model: NeMo VAD model name (default: 'vad_multilingual_marblenet')
            speaker_model: NeMo speaker embedding model (default: 'titanet_large')
            window_length_in_sec: Window length for speaker embeddings
            shift_length_in_sec: Shift length for speaker embeddings
        """
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError(
                "NeMo is not installed. Please install with: pip install nemo_toolkit[asr]"
            )
        
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.vad_model = vad_model
        self.speaker_model = speaker_model
        self.window_length = window_length_in_sec
        self.shift_length = shift_length_in_sec
        
        # Create temporary directory for NeMo outputs
        self.temp_dir = tempfile.mkdtemp(prefix="nemo_diar_")
        
        logger.info(f"Initializing NeMo diarization with speaker model: {speaker_model}")
        logger.info(f"VAD model: {vad_model}")
        
        # We'll create the config when we run diarization
        # since NeMo requires a manifest file
        self.ClusteringDiarizer = ClusteringDiarizer
        self.OmegaConf = OmegaConf

    def _create_manifest(self, audio_path: str, duration: float) -> str:
        """
        Create a temporary manifest file for NeMo.
        
        Args:
            audio_path: Path to audio file
            duration: Duration of audio in seconds
            
        Returns:
            Path to manifest file
        """
        manifest_path = os.path.join(self.temp_dir, "manifest.json")
        
        manifest_entry = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)
            f.write('\n')
        
        return manifest_path

    def _create_config(
        self, 
        manifest_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Create NeMo diarization configuration.
        
        Args:
            manifest_path: Path to manifest file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            OmegaConf configuration object
        """
        config = self.OmegaConf.create({
            "diarizer": {
                "manifest_filepath": manifest_path,
                "out_dir": self.temp_dir,
                "oracle_vad": False,
                "vad": {
                    "model_path": self.vad_model,
                    "parameters": {
                        "onset": 0.5,
                        "offset": 0.5,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1,
                    }
                },
                "speaker_embeddings": {
                    "model_path": self.speaker_model,
                    "parameters": {
                        "window_length_in_sec": self.window_length,
                        "shift_length_in_sec": self.shift_length,
                        "multiscale_weights": None,
                        "save_embeddings": True,
                    }
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": num_speakers is not None,
                        "max_num_speakers": max_speakers or 8,
                        "enhanced_count_thres": 80,
                        "maj_vote_spk_count": False,
                    }
                }
            }
        })
        
        # Set oracle num speakers if provided
        if num_speakers is not None:
            config.diarizer.clustering.parameters.oracle_num_speakers = True
            # NeMo expects this to be set in the manifest, but we can work around it
        
        return config

    def _parse_rttm(self, rttm_path: str) -> pd.DataFrame:
        """
        Parse RTTM file to DataFrame.
        
        Args:
            rttm_path: Path to RTTM file
            
        Returns:
            DataFrame with columns: start, end, speaker
        """
        segments = []
        
        if not os.path.exists(rttm_path):
            logger.warning(f"RTTM file not found: {rttm_path}")
            return pd.DataFrame(columns=["start", "end", "speaker"])
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    speaker = parts[7]
                    segments.append({
                        "start": start,
                        "end": end,
                        "speaker": speaker
                    })
        
        return pd.DataFrame(segments)

    def _load_embeddings(self, audio_filename: str) -> Optional[dict[str, list[float]]]:
        """
        Load speaker embeddings from NeMo output.
        
        Args:
            audio_filename: Base name of audio file
            
        Returns:
            Dictionary mapping speaker IDs to embedding vectors
        """
        # NeMo saves embeddings as pickle files
        import pickle
        
        # Look for embedding file
        base_name = os.path.splitext(audio_filename)[0]
        emb_path = os.path.join(self.temp_dir, "speaker_outputs", f"{base_name}.pkl")
        
        if not os.path.exists(emb_path):
            logger.warning(f"Embeddings file not found: {emb_path}")
            return None
        
        try:
            with open(emb_path, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            # Extract speaker embeddings
            # NeMo format may vary, adapt as needed
            speaker_embeddings = {}
            if isinstance(embeddings_data, dict):
                # Try to extract embeddings per speaker
                for speaker, emb in embeddings_data.items():
                    if isinstance(emb, np.ndarray):
                        speaker_embeddings[speaker] = emb.tolist()
                    elif isinstance(emb, torch.Tensor):
                        speaker_embeddings[speaker] = emb.cpu().numpy().tolist()
            
            return speaker_embeddings if speaker_embeddings else None
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")
            return None

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        # Handle audio input
        temp_audio_path = None
        audio_filename = None
        
        if isinstance(audio, str):
            # Audio is already a file path
            audio_path = audio
            audio_filename = os.path.basename(audio_path)
            # Load to get duration
            audio_array = load_audio(audio_path)
            duration = len(audio_array) / SAMPLE_RATE
        else:
            # Audio is numpy array, save to temporary file
            import soundfile as sf
            temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
            sf.write(temp_audio_path, audio, SAMPLE_RATE)
            audio_path = temp_audio_path
            audio_filename = "temp_audio.wav"
            duration = len(audio) / SAMPLE_RATE
        
        try:
            # Create manifest
            manifest_path = self._create_manifest(audio_path, duration)
            
            # Create config
            config = self._create_config(
                manifest_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            
            # Initialize and run diarization
            logger.info("Running NeMo diarization...")
            diarizer = self.ClusteringDiarizer(cfg=config)
            diarizer.diarize()
            
            # Parse RTTM output
            base_name = os.path.splitext(audio_filename)[0]
            rttm_path = os.path.join(self.temp_dir, "pred_rttms", f"{base_name}.rttm")
            diarize_df = self._parse_rttm(rttm_path)
            
            # Load embeddings if requested
            speaker_embeddings = None
            if return_embeddings:
                speaker_embeddings = self._load_embeddings(audio_filename)
            
            # Clean up temporary audio file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if return_embeddings:
                return diarize_df, speaker_embeddings
            else:
                return diarize_df
                
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Clean up and re-raise
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            raise

    def __del__(self):
        """Clean up temporary directory."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.
    
    This function is identical to the pyannote version and works with both backends.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

        # assign speaker to words
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    """Helper class for representing audio segments with speaker labels."""
    
    def __init__(self, start: float, end: float, speaker: Optional[str] = None):
        """
        Initialize a segment.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            speaker: Speaker label (optional)
        """
        self.start = start
        self.end = end
        self.speaker = speaker
