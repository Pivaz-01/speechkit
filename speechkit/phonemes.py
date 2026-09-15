"""
Phoneme recognition and pause detection.

The recognition and pause-detection code from ``phonemes_from_audio.py``, with
three changes:

* the constants (SAMPLE_RATE, FRAME_DURATION_MS, MIN_PAUSE_DURATION_S, NOISE_K,
  MODEL_NAME) are arguments instead of module globals, so the interface can
  change them without editing the file;
* the Flask app and the embedded HTML page moved to ``speechkit.web``. This
  module is importable without a server;
* torch and transformers are imported lazily, inside ``get_model``, so the
  interface starts and can report a missing dependency instead of failing at
  import.

The measurement itself is unchanged: the same Otsu-plus-noise-floor threshold,
the same greedy CTC collapse, the same rule that a phoneme whose midpoint falls
inside a pause is dropped.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import _winsetup
from ._files import collect_audio, resolve_output_dir

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 at `sample_rate`."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample

        data = resample(data, int(len(data) * sample_rate / sr)).astype(np.float32)
    return data


def audio_to_data_uri(path: str) -> str:
    """Inline the audio for the editor's waveform player."""
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}.get(ext, "audio/wav")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# --------------------------------------------------------------------------
# Pause detection
# --------------------------------------------------------------------------


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method: the threshold maximising between-class variance on the
    histogram, which separates silence from speech without a hand-set level."""
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmax
    hist, edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return vmax
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return vmax
    cum_mean = np.cumsum(hist * centers)
    mean_bg = np.zeros_like(weight_bg)
    mean_fg = np.zeros_like(weight_fg)
    mean_bg[valid] = cum_mean[valid] / weight_bg[valid]
    total_mean = cum_mean[-1]
    mean_fg[valid] = (total_mean - cum_mean[valid]) / weight_fg[valid]
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var[~valid] = 0
    return float(centers[int(np.argmax(between_var))])


def detect_pauses(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    min_pause_duration_s: float = 0.15,
    noise_k: float = 3.0,
) -> tuple[list[tuple[float, float]], float, list[float]]:
    """
    Pauses, the threshold used, and the RMS envelope.

    The threshold is adaptive: Otsu separates the two regimes, a robust noise
    estimate (median plus k scaled MADs of the lower half of the distribution)
    puts a floor under it, and the higher of the two wins. A noisy room is
    therefore not mistaken for speech, and no level has to be tuned by hand.
    """
    frame_len = int(sample_rate * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_len if frame_len else 0
    if n_frames == 0:
        return [], 0.0, []

    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    nonzero = rms[rms > 0]
    if len(nonzero) == 0:
        return [], 0.0, rms.tolist()

    otsu = otsu_threshold(nonzero)
    noise_floor = nonzero[nonzero <= np.median(nonzero)]
    if len(noise_floor) > 0:
        med = float(np.median(noise_floor))
        mad = float(np.median(np.abs(noise_floor - med))) or 1e-9
        noise_thr = med + noise_k * 1.4826 * mad   # 1.4826 scales MAD to sigma
    else:
        noise_thr = otsu
    threshold = float(max(otsu, noise_thr))

    is_silent = rms < threshold
    pauses: list[tuple[float, float]] = []
    in_pause, pause_start = False, 0.0
    for i in range(n_frames):
        t = i * frame_duration_ms / 1000.0
        if is_silent[i] and not in_pause:
            in_pause, pause_start = True, t
        elif not is_silent[i] and in_pause:
            in_pause = False
            if t - pause_start >= min_pause_duration_s:
                pauses.append((pause_start, t))
    if in_pause:
        end = n_frames * frame_duration_ms / 1000.0
        if end - pause_start >= min_pause_duration_s:
            pauses.append((pause_start, end))

    return pauses, threshold, rms.tolist()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, dict[int, str], int]] = {}


def resolve_device(requested: str = "auto") -> str:
    _winsetup.configure()
    import torch

    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch reports no CUDA device. "
                "Set the compute device to CPU, or install a CUDA build of torch.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str, device: str = "auto", log: Callable[[str], None] = print):
    """
    Load and cache a wav2vec2 CTC phoneme model.

    Cached on (model name, device) so a batch loads it once, and so switching
    model in the interface does not keep serving the old one.
    """
    _winsetup.configure()
    resolved = resolve_device(device)
    key = (model_name, resolved)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import json as _json

    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    # transformers v5 reorganised the preprocessing classes. The deprecation
    # was aimed at the image path (AutoFeatureExtractor -> AutoImageProcessor),
    # and audio feature extractors were meant to survive, but the v5 line ships
    # breaking changes weekly, so take whichever name this install actually has
    # rather than assuming.
    extractor_cls = None
    for module_attr in ("Wav2Vec2FeatureExtractor", "AutoFeatureExtractor", "AutoProcessor"):
        try:
            import transformers

            extractor_cls = getattr(transformers, module_attr)
            break
        except AttributeError:
            continue
    if extractor_cls is None:
        raise RuntimeError(
            "this version of transformers exposes none of "
            "Wav2Vec2FeatureExtractor, AutoFeatureExtractor or AutoProcessor. "
            "Install a version known to work: pip install 'transformers<5'")

    log(f"[model] loading {model_name} on {resolved}")
    extractor = extractor_cls.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(resolved)
    model.eval()

    vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.json")
    vocab = _json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    id_to_token = {v: k for k, v in vocab.items()}
    blank_id = vocab.get("<pad>", 0)

    _MODEL_CACHE[key] = (model, extractor, id_to_token, blank_id)
    log(f"[model] ready, {len(id_to_token)} vocabulary entries")
    return _MODEL_CACHE[key]


def unload_models() -> None:
    _MODEL_CACHE.clear()


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def decode_file(
    wav_path: str,
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
    device: str = "auto",
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    min_pause_duration_s: float = 0.15,
    noise_k: float = 3.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """
    Recognise one file. Returns segments, the RMS envelope, duration and the
    pause threshold, ready for the editor or for `write_tsv`.
    """
    import torch

    model, extractor, id_to_token, blank_id = get_model(model_name, device, log)
    torch_device = next(model.parameters()).device

    audio = load_audio(wav_path, sample_rate)
    duration = len(audio) / sample_rate
    pauses, threshold, rms = detect_pauses(
        audio, sample_rate, frame_duration_ms, min_pause_duration_s, noise_k)

    inputs = extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    # AutoProcessor returns a dict-like without the .input_values attribute that
    # Wav2Vec2FeatureExtractor provides, so accept either shape.
    values = getattr(inputs, "input_values", None)
    if values is None:
        values = inputs["input_values"]
    with torch.no_grad():
        logits = model(values.to(torch_device)).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    n_frames = probs.shape[0]
    if n_frames == 0:
        return {"segments": [], "rms": rms, "duration": duration, "threshold": threshold}
    spf = duration / n_frames
    best_ids = np.argmax(probs, axis=1)

    # Greedy CTC collapse: a run of the same non-blank id is one segment, and
    # its confidence is the mean posterior over the run.
    segs: list[tuple[str, float, float, float]] = []
    cur_id, cur_probs, cur_start = None, [], 0
    for i in range(n_frames):
        tid = int(best_ids[i])
        if tid == blank_id:
            if cur_id is not None:
                segs.append((id_to_token.get(cur_id, "?"), float(np.mean(cur_probs)),
                             cur_start * spf, i * spf))
            cur_id, cur_probs = None, []
        elif tid != cur_id:
            if cur_id is not None:
                segs.append((id_to_token.get(cur_id, "?"), float(np.mean(cur_probs)),
                             cur_start * spf, i * spf))
            cur_id, cur_start, cur_probs = tid, i, [float(probs[i, tid])]
        else:
            cur_probs.append(float(probs[i, tid]))
    if cur_id is not None:
        segs.append((id_to_token.get(cur_id, "?"), float(np.mean(cur_probs)),
                     cur_start * spf, n_frames * spf))

    combined = [{"kind": "phoneme", "label": l, "conf": round(c, 4),
                 "start": round(s, 4), "end": round(e, 4)} for l, c, s, e in segs]
    combined += [{"kind": "pause", "label": "<pause>", "conf": 0.0,
                  "start": round(ps, 4), "end": round(pe, 4)} for ps, pe in pauses]
    combined.sort(key=lambda x: x["start"])

    # A phoneme whose midpoint lands inside a detected pause is spurious.
    filtered = []
    for seg in combined:
        if seg["kind"] == "phoneme":
            mid = (seg["start"] + seg["end"]) / 2
            if any(ps <= mid <= pe for ps, pe in pauses):
                continue
        filtered.append(seg)

    return {"segments": filtered, "rms": rms, "duration": duration, "threshold": threshold}


# --------------------------------------------------------------------------
# TSV output
# --------------------------------------------------------------------------


def rate_metrics(segments, duration: float) -> dict[str, float]:
    """
    Speech rate and articulation rate.

    Speech rate counts phonemes over the whole signal; articulation rate counts
    them over phonation time only, which is the measure used in most clinical
    work because it separates slowing from pausing.
    """
    phonemes = [s for s in segments if s["kind"] == "phoneme"]
    pauses = [s for s in segments if s["kind"] == "pause"]
    n_phonemes = len(phonemes)
    total_pause_s = sum(max(0.0, s["end"] - s["start"]) for s in pauses)
    speaking_time = max(duration - total_pause_s, 1e-9)
    return {
        "n_phonemes": n_phonemes,
        "duration_s": duration,
        "speaking_time_s": speaking_time,
        "total_pause_s": total_pause_s,
        "speech_rate": n_phonemes / duration if duration > 0 else 0.0,
        "articulation_rate": n_phonemes / speaking_time,
    }


def write_tsv(out_path, segments, duration: float, threshold: float) -> dict[str, Any]:
    """
    Write the segment table, with the rate metrics in the comment header.

    Same format the editor's export button has always produced, so files from
    either route are read identically by the alignment stage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = rate_metrics(segments, duration)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Intensity threshold: {threshold:.6f}\n")
        f.write(f"# Total duration (s): {m['duration_s']:.4f}\n")
        f.write(f"# Speaking time (s, pauses excluded): {m['speaking_time_s']:.4f}\n")
        f.write(f"# Phoneme count: {m['n_phonemes']}\n")
        f.write(f"# Speech rate (phonemes/s, incl. pauses): {m['speech_rate']:.4f}\n")
        f.write(f"# Articulation rate (phonemes/s, excl. pauses): {m['articulation_rate']:.4f}\n")
        f.write("type\tlabel\tconfidence\tstart\tend\n")
        for s in segments:
            f.write(f"{s['kind']}\t{s['label']}\t{float(s['conf']):.4f}"
                    f"\t{float(s['start']):.4f}\t{float(s['end']):.4f}\n")
    result = dict(m)
    result["path"] = str(out_path)
    return result


def tsv_path_for(wav_path: str, output_dir: str, suffix: str) -> Path:
    """Where the TSV for this recording goes. An empty output folder means
    beside the audio, which is what the original script did."""
    wav = Path(wav_path)
    folder = resolve_output_dir(output_dir, wav.parent)
    return folder / f"{wav.stem}{suffix}"


# --------------------------------------------------------------------------
# Batch entry point
# --------------------------------------------------------------------------


def run_phonemes(
    cfg: dict[str, Any],
    log: Callable[[str], None] = print,
    write_files: bool = True,
    keep_audio: bool = True,
    should_stop: Callable[[], bool] = lambda: False,
) -> list[dict[str, Any]]:
    """
    Recognise every selected file.

    Returns one record per file, which the interface both hands to the editor
    and summarises. `write_files` also writes a TSV per file; `keep_audio`
    inlines the audio so the editor can draw and play it, and is worth turning
    off for a large batch that will not be reviewed.
    """
    inputs = cfg.get("PHONEME_INPUTS", [])
    wavs = collect_audio(inputs, recursive=bool(cfg.get("PHONEME_RECURSIVE", False)))
    if not wavs:
        raise ValueError(
            "no .wav files found. Check the folders listed under Input and output, "
            "and whether subfolders need to be included.")
    log(f"[input] {len(wavs)} wav file(s)")

    records: list[dict[str, Any]] = []
    for i, wav_path in enumerate(wavs, start=1):
        if should_stop():
            log("[stopped] cancelled before finishing the batch")
            break
        name = Path(wav_path).name
        log(f"[{i}/{len(wavs)}] {name}")
        try:
            decoded = decode_file(
                wav_path,
                model_name=cfg.get("MODEL_NAME", "facebook/wav2vec2-lv-60-espeak-cv-ft"),
                device=cfg.get("DEVICE", "auto"),
                sample_rate=int(cfg.get("SAMPLE_RATE", 16000)),
                frame_duration_ms=int(cfg.get("FRAME_DURATION_MS", 20)),
                min_pause_duration_s=float(cfg.get("MIN_PAUSE_DURATION_S", 0.15)),
                noise_k=float(cfg.get("NOISE_K", 3.0)),
                log=log,
            )
        except Exception as exc:                      # one bad file must not lose the batch
            log(f"    failed: {exc}")
            continue

        record: dict[str, Any] = {
            "wav_path": wav_path,
            "segments": decoded["segments"],
            "rms": decoded["rms"],
            "duration": decoded["duration"],
            "threshold": decoded["threshold"],
            "audio_b64": audio_to_data_uri(wav_path) if keep_audio else "",
        }
        metrics = rate_metrics(decoded["segments"], decoded["duration"])
        record["metrics"] = metrics

        if write_files:
            out = tsv_path_for(wav_path,
                               cfg.get("PHONEME_OUTPUT_DIR", ""),
                               cfg.get("PHONEME_TSV_SUFFIX", "_auto.tsv"))
            write_tsv(out, decoded["segments"], decoded["duration"], decoded["threshold"])
            record["tsv_path"] = str(out)

        n_pauses = sum(1 for s in decoded["segments"] if s["kind"] == "pause")
        log(f"    {metrics['n_phonemes']} phonemes, {n_pauses} pauses, "
            f"{decoded['duration']:.2f}s, articulation "
            f"{metrics['articulation_rate']:.2f} ph/s")
        records.append(record)

    if not records:
        raise RuntimeError("every file failed to analyse; see the log above")
    log(f"[done] {len(records)} file(s) recognised")
    return records
