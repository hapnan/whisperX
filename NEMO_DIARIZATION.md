# NeMo Diarization Implementation for WhisperX

This directory contains an alternative implementation of speaker diarization using **NVIDIA NeMo** instead of pyannote.audio.

## Overview

The `diarize_nemo.py` module provides a drop-in replacement for the pyannote-based diarization in WhisperX, using NVIDIA NeMo's `ClusteringDiarizer`.

## Installation

### Requirements

Install NeMo with ASR support using `uv`:

```bash
uv add nemo_toolkit[asr]
```

Or for a specific version:

```bash
uv add nemo_toolkit[asr]==1.23.0
```

Alternatively, add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "nemo_toolkit[asr]>=1.23.0",
    # other dependencies...
]
```

Then run:

```bash
uv sync
```

### Additional Dependencies

NeMo requires:
- PyTorch (>= 1.13.0)
- omegaconf
- hydra-core
- soundfile (for audio I/O)

These are typically installed automatically with `nemo_toolkit[asr]`.

## Usage

### Basic Usage

Replace the import in your code:

```python
# Original pyannote version
# from whisperx.diarize import DiarizationPipeline

# NeMo version
from whisperx.diarize_nemo import DiarizationPipeline

# Initialize the pipeline
diarize_model = DiarizationPipeline(
    device="cuda",  # or "cpu"
    vad_model="vad_multilingual_marblenet",
    speaker_model="titanet_large",
    window_length_in_sec=1.5,
    shift_length_in_sec=0.75,
)

# Perform diarization
diarize_segments = diarize_model(
    audio="path/to/audio.wav",
    min_speakers=2,
    max_speakers=4,
)
```

### With Speaker Embeddings

```python
diarize_segments, speaker_embeddings = diarize_model(
    audio="path/to/audio.wav",
    num_speakers=3,  # exact number if known
    return_embeddings=True,
)
```

### Assign Speakers to Transcript

The `assign_word_speakers` function works identically with both backends:

```python
from whisperx.diarize_nemo import assign_word_speakers

# After transcription and alignment
result = assign_word_speakers(
    diarize_df=diarize_segments,
    transcript_result=aligned_result,
    speaker_embeddings=speaker_embeddings,
    fill_nearest=False,
)
```

## Key Differences from Pyannote

### Advantages of NeMo

1. **Open Source**: Fully open-source under Apache 2.0 license
2. **No Authentication**: No HuggingFace tokens required
3. **Optimized for NVIDIA GPUs**: Better performance on CUDA devices
4. **Modular**: Separate VAD and speaker embedding models
5. **Multiple Pre-trained Models**: Various speaker models available:
   - `titanet_large` (recommended)
   - `ecapa_tdnn`
   - `speakerverification_speakernet`

### Differences

1. **Configuration**: NeMo uses OmegaConf configuration files
2. **File-based Processing**: NeMo expects manifest files (handled internally)
3. **Output Format**: Uses RTTM format internally (converted to DataFrame)
4. **Embeddings**: Different embedding format and extraction method

## Configuration Options

### Constructor Parameters

```python
DiarizationPipeline(
    model_name=None,              # Not used (kept for compatibility)
    use_auth_token=None,          # Not used (kept for compatibility)
    device="cuda",                # "cuda" or "cpu"
    vad_model="vad_multilingual_marblenet",  # NeMo VAD model
    speaker_model="titanet_large",           # NeMo speaker model
    window_length_in_sec=1.5,     # Window for embeddings
    shift_length_in_sec=0.75,     # Shift for embeddings
)
```

### Call Parameters

```python
diarize_model(
    audio,                  # str (path) or np.ndarray
    num_speakers=None,      # Exact number if known
    min_speakers=None,      # Minimum speakers
    max_speakers=8,         # Maximum speakers (default: 8)
    return_embeddings=False # Return speaker embeddings
)
```

## Available Models

### VAD Models
- `vad_multilingual_marblenet` (recommended, supports multiple languages)
- `vad_telephonic_marblenet` (optimized for telephonic audio)

### Speaker Models
- `titanet_large` (recommended, best accuracy)
- `titanet_small` (faster, slightly lower accuracy)
- `ecapa_tdnn` (alternative architecture)

## Output Format

The output DataFrame has the same format as pyannote:

```python
   start    end      speaker
0   0.5    3.2   SPEAKER_00
1   3.5    7.8   SPEAKER_01
2   8.0   12.3   SPEAKER_00
```

## Performance Considerations

1. **GPU Memory**: NeMo models may require more GPU memory than pyannote
2. **Speed**: Generally faster on NVIDIA GPUs with proper CUDA setup
3. **Accuracy**: Comparable to pyannote, may vary by domain
4. **Multiscale**: Can use multiscale embeddings for better accuracy:

```python
diarize_model = DiarizationPipeline(
    device="cuda",
    speaker_model="titanet_large",
    window_length_in_sec=[1.5, 1.0, 0.5],  # Multiple scales
    shift_length_in_sec=[0.75, 0.5, 0.25],
)
```

## Troubleshooting

### Import Errors

If you get `ImportError: NeMo is not installed`:
```bash
uv pip install nemo_toolkit[asr]
```

### CUDA Errors

Ensure PyTorch is installed with CUDA support:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Or add to your `pyproject.toml`:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu118" }

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
```

### Model Download Issues

Models are downloaded automatically from NGC. If you have network issues:
1. Models are cached in `~/.cache/torch/NeMo/`
2. You can manually download and specify local paths

### Memory Issues

Reduce window sizes or use CPU:
```python
diarize_model = DiarizationPipeline(
    device="cpu",
    window_length_in_sec=1.0,
    shift_length_in_sec=0.5,
)
```

## Example: Complete Pipeline

```python
import whisperx
from whisperx.diarize_nemo import DiarizationPipeline, assign_word_speakers

# Load audio
audio = whisperx.load_audio("audio.wav")

# 1. Transcribe
model = whisperx.load_model("large-v2", device="cuda")
result = model.transcribe(audio)

# 2. Align
align_model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
result = whisperx.align(result["segments"], align_model, metadata, audio, device="cuda")

# 3. Diarize with NeMo
diarize_model = DiarizationPipeline(device="cuda")
diarize_segments = diarize_model(audio)

# 4. Assign speakers
result = assign_word_speakers(diarize_segments, result)

# Print results
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment.get('speaker', 'UNKNOWN')}: {segment['text']}")
```

## References

- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
- [NeMo Speaker Diarization Tutorial](https://github.com/NVIDIA/NeMo/tree/main/tutorials/speaker_tasks)
- [TitaNet Paper](https://arxiv.org/abs/2110.03435)
- [ClusteringDiarizer API](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)

## License

This implementation follows the same license as WhisperX. NVIDIA NeMo is licensed under Apache 2.0.
