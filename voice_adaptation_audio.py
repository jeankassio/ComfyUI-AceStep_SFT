"""Local audio/transcript preparation for per-reference voice adaptation."""
import json
import math

import torch
from torchaudio.functional import resample


WHISPER_MODEL = "openai/whisper-large-v3-turbo"


def _copy_segment(waveform, sample_rate, start, end, text):
    return {
        "audio": {"waveform": waveform[..., start:end].detach().clone(), "sample_rate": sample_rate},
        "text": text,
        "start": start / sample_rate,
        "end": end / sample_rate,
    }


def _transcribe_segments(segments, language):
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as error:
        raise RuntimeError("Voice adaptation needs the installed transformers Whisper backend or a supplied transcript.") from error

    try:
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        ).to("cpu").eval()
    except OSError as error:
        raise RuntimeError(
            "Whisper Turbo is not available in the local Hugging Face cache. Supply a transcript; "
            "voice adaptation does not download models."
        ) from error

    spoken_language = None if language == "auto" else language
    for segment in segments:
        audio = segment["audio"]
        mono = audio["waveform"][0].detach().to(device="cpu", dtype=torch.float32).mean(dim=0)
        if audio["sample_rate"] != 16000:
            mono = resample(mono, audio["sample_rate"], 16000)
        inputs = processor(mono.numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        generated = model.generate(
            **inputs, language=spoken_language, task="transcribe", do_sample=False, num_beams=1,
            max_new_tokens=256, return_timestamps=False,
        )
        segment["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def prepare_reference_segments(audio, transcript="", language="auto", crop_seconds=6.0):
    """Copy aligned reference examples; automatic transcription stays offline on CPU.

    A plain transcript belongs to the whole reference. A JSON list of
    ``{start, end, text}`` records provides exact individual segment timings.
    With no transcript, transcribe three nonoverlapping front/middle/back crops.
    """
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Voice adaptation requires an AUDIO waveform and sample_rate.")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("The reference sample_rate must be a positive integer.")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("The reference waveform must have shape [1, channels, samples].")
    if waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[2] == 0:
        raise ValueError("Voice adaptation requires one nonempty mono or stereo reference.")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError("The reference waveform must contain finite floating-point samples.")
    if not isinstance(transcript, str) or not isinstance(language, str):
        raise ValueError("The transcript and language must be text.")
    crop_seconds = float(crop_seconds)
    if not math.isfinite(crop_seconds) or crop_seconds <= 0:
        raise ValueError("crop_seconds must be finite and positive.")

    samples = waveform.shape[-1]
    duration = samples / sample_rate
    transcript = transcript.strip()
    if transcript:
        if not transcript.startswith(("[", "{")):
            return [_copy_segment(waveform, sample_rate, 0, samples, transcript)]
        try:
            timed = json.loads(transcript)
        except json.JSONDecodeError as error:
            raise ValueError("Timed transcripts must be a JSON list of {start, end, text} records.") from error
        if not isinstance(timed, list) or not timed:
            raise ValueError("Timed transcripts must be a nonempty JSON list.")
        segments = []
        for entry in timed:
            if not isinstance(entry, dict) or not {"start", "end", "text"}.issubset(entry):
                raise ValueError("Each timed transcript needs start, end, and text.")
            start, end, text = entry["start"], entry["end"], entry["text"]
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= duration):
                raise ValueError("Transcript times must be finite and within the reference duration, with start < end.")
            if not isinstance(text, str):
                raise ValueError("Each timed transcript text must be a string.")
            first, last = round(start * sample_rate), round(end * sample_rate)
            if first == last:
                raise ValueError("A timed transcript segment is shorter than one audio sample.")
            segments.append(_copy_segment(waveform, sample_rate, first, last, text.strip()))
        return segments

    count = min(3, samples)
    crop_samples = max(1, min(round(crop_seconds * sample_rate), samples // count))
    starts = [0] if count == 1 else [0, samples - crop_samples]
    if count == 3:
        starts.insert(1, (samples - crop_samples) // 2)
    segments = [_copy_segment(waveform, sample_rate, start, start + crop_samples, "") for start in starts]
    _transcribe_segments(segments, language.strip() or "auto")
    return segments
