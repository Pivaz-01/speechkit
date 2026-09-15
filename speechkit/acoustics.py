"""
Acoustic analysis (Praat / parselmouth) — measurement engine.

This is the analysis code from ``praat_v10.py``, unchanged apart from three
things:

* the 730-line version history that used to sit in this docstring now lives in
  ``CHANGELOG.md``, so the file starts with code;
* the ``if __name__ == "__main__"`` block of hardcoded Windows paths is gone.
  Inputs and settings arrive through :func:`run_acoustics`, which the interface
  calls with whatever is in the settings panel;
* ``KNOWN_PASSAGE_SYLLABLES`` can be edited at runtime through
  :func:`set_passage_syllables`, because those counts have to be checked
  against the exact wording of the passage in use.

Nothing about how a number is measured was touched. The public entry point is
still :func:`analyze_files`, with the same signature it always had, so existing
scripts keep working:

    from speechkit.acoustics import analyze_files
    results = analyze_files(wav_files=[...], output_folder="...", task="reading")
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial import ConvexHull
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings
import csv
import re
import json
from datetime import datetime
import parselmouth
from parselmouth.praat import call

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / scripts
import matplotlib.pyplot as plt

# v20: warnings are NOT blanket-silenced any more. A librosa/parselmouth
# warning is often the first sign that a file is degenerate (empty slice,
# all-NaN axis, divide-by-zero), and hiding it hides the QC signal. 'once'
# keeps the console readable on a large batch without discarding the
# information entirely.
warnings.simplefilter("once")


# =============================================================================
# KNOWN-PASSAGE SYLLABLE REFERENCE  (fixes the rate bug)
# =============================================================================
#
# Syllable counts for the standard reading passages, matched against the
# filename stem (case-insensitive substring). When a passage is identified, the
# syllable count is taken from here and the rate is EXACT; otherwise an
# un-clamped estimate is used.
#
# IMPORTANT: verify each count against YOUR exact passage version. A wrong
# constant biases the rate for every file of that passage. The
# speaking_time_fraction metric is independent of these constants and stays
# valid regardless.
KNOWN_PASSAGE_SYLLABLES: Dict[str, Optional[int]] = {
    "caterpillar":   260,    
    "grandfather":   174,    
    "rainbow":       459,    
    "outside":       195,    
    "visittomarket": 204,    
    "daily":         209,
    "house":         181,
    "community":     350,
}


def _token_present(token: str, text: str) -> bool:
    """
    Does `token` appear in `text` at the start of a word?

    Plain substring matching was too loose for the token lists below: "house"
    matched "greenhouse", "text" matched "context", "read" matched "spreadsheet".
    Requiring the match to begin a word (not preceded by a letter) keeps the
    deliberate prefix behaviour - "sust" still finds "sustained", "descript"
    still finds "description" - while removing the accidental hits. Digits,
    underscores, hyphens and separators all count as word boundaries, which is
    what real filenames use.
    """
    if not token:
        return False
    return re.search(r"(?<![a-z])" + re.escape(token.lower()), text) is not None


def identify_passage(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (passage_name, syllable_count) if filename matches a known passage."""
    stem = Path(file_path).stem.lower()
    for token, count in KNOWN_PASSAGE_SYLLABLES.items():
        if count is not None and _token_present(token, stem):
            return token, int(count)
    return None, None


# =============================================================================
# TASK TYPE:  sustained vowel  vs  connected speech (reading / spontaneous)
# =============================================================================
# Almost every threshold in this script was designed for CONNECTED SPEECH. On a
# sustained /a/ the same thresholds produce systematic false positives (SNR
# measured against the vowel itself, "band-limited channel" from the absent
# consonant energy, syllables peak-picked out of amplitude ripple, an EMS
# "deficit" where no syllabic modulation can exist, and a fluency score built
# from all of the above). The task type therefore gates which metrics are
# computed at all, and which are emitted as NaN.
TASK_SUSTAINED = "sustained_vowel"
TASK_READING = "reading"

_SUSTAINED_NAME_TOKENS = (
    "sust", "sustain", "vowel", "voyelle", "vocale", "mpt", "phonation",
    "aaa", "eee", "iii", "ooo", "uuu", "sust_a", "sust_i", "sust_u",
)
_READING_NAME_TOKENS = (
    "read", "passage", "text", "spont", "monolog", "dialog", "convers",
    "picture", "descript", "sentence", "phrase", "dditk", "story",
)


def _path_tokens(file_path: str) -> str:
    """Lower-cased 'parent-folder/stem' string used for filename-based routing."""
    p = Path(file_path)
    return str(p.parent).replace("\\", "/").lower() + "/" + p.stem.lower()


def task_from_filename(file_path: str) -> Optional[str]:
    """
    Decide the task from the file name, then from the folder. Returns None when
    undecidable so the caller can fall back to the acoustic classifier.

    v19 - PRECEDENCE FIXED. The old order was: passage token (stem) -> reading
    tokens (folder + stem) -> sustained tokens (folder + stem). Two consequences,
    both of which routed sustained files into the connected-speech pipeline:

      * a passage token beat everything, so "P03_sust_a_daily.wav" became a
        reading file because "daily" is a key in KNOWN_PASSAGE_SYLLABLES;
      * reading tokens were matched against the whole PARENT PATH before
        sustained tokens were tried at all, so every file under a folder called
        "Readings", "text", "story" or similar was reading regardless of its own
        name - "/study/Readings/P03_sust_a.wav" included.

    The stem describes the recording; the folder only describes where it was
    filed. So the stem is asked first, and within each level the two token sets
    are checked together rather than one before the other.
    """
    p = Path(file_path)
    stem = p.stem.lower()
    parent = str(p.parent).replace("\\", "/").lower()

    def _decide(text: str) -> Optional[str]:
        """
        Three tiers of evidence, because they are not equally strong.

        A token from _SUSTAINED_NAME_TOKENS or _READING_NAME_TOKENS names the
        TASK ("sust", "mpt", "phonation", "read", "passage"). A key of
        KNOWN_PASSAGE_SYLLABLES only names a STIMULUS ("daily", "house",
        "rainbow"), and a stimulus name is weaker evidence: a file called
        "P03_sust_a_daily" is a sustained vowel recorded in a session that also
        used the daily passage, not a reading. So an explicit task token decides
        first, and the passage name is consulted only when no task token is
        present. Genuinely contradictory names return None and let the acoustic
        classifier settle it, which is the one thing that cannot be fooled by a
        filename.
        """
        sust = any(_token_present(t, text) for t in _SUSTAINED_NAME_TOKENS)
        read = any(_token_present(t, text) for t in _READING_NAME_TOKENS)
        if sust and not read:
            return TASK_SUSTAINED
        if read and not sust:
            return TASK_READING
        if sust and read:
            return None
        if any(_token_present(t, text) for t in KNOWN_PASSAGE_SYLLABLES):
            return TASK_READING
        return None

    return _decide(stem) or _decide(parent)


def task_from_acoustics(voiced_fraction: float, ems_3_8hz_ratio: float,
                        centroid_cv: float) -> Tuple[str, str]:
    """
    Acoustic fallback classifier, used only when the filename is uninformative.
    A sustained vowel is (a) almost continuously voiced, (b) carries almost no
    3-8 Hz syllabic envelope modulation, and (c) has a near-stationary spectrum.
    Connected speech fails all three. Two votes out of three decide.
    Returns (task, human-readable reason).
    """
    votes = []
    if np.isfinite(voiced_fraction) and voiced_fraction >= 0.80:
        votes.append(f"voiced {voiced_fraction*100:.0f}%")
    if np.isfinite(ems_3_8hz_ratio) and ems_3_8hz_ratio < 0.22:
        votes.append(f"EMS(3-8Hz) {ems_3_8hz_ratio:.2f}")
    if np.isfinite(centroid_cv) and centroid_cv < 0.20:
        votes.append(f"centroid CV {centroid_cv:.2f}")
    if len(votes) >= 2:
        return TASK_SUSTAINED, "sustained-vowel evidence: " + ", ".join(votes)
    return TASK_READING, "connected-speech evidence (fewer than 2 sustained votes)"


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================

@dataclass
class PauseMetrics:
    """Metrics for silence pauses in speech."""
    count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    std_duration: float = 0.0
    median_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    ratio_to_speech: float = 0.0
    pauses_per_minute: float = 0.0
    durations: List[float] = field(default_factory=list)


@dataclass
class FillerMetrics:
    """Metrics for filler sounds (um, uh, mm, schwa, etc.)."""
    count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    std_duration: float = 0.0
    ratio_to_speech: float = 0.0
    fillers_per_minute: float = 0.0
    durations: List[float] = field(default_factory=list)


@dataclass
class IntensityMetrics:
    """Metrics for intensity/loudness (Praat-equivalent)."""
    mean_db: float = float("nan")
    std_db: float = float("nan")
    min_db: float = float("nan")
    max_db: float = float("nan")
    range_db: float = float("nan")
    median_db: float = float("nan")
    quantile_25_db: float = float("nan")
    quantile_75_db: float = float("nan")
    coefficient_of_variation: float = float("nan")
    active_mean_db: float = float("nan")
    active_std_db: float = float("nan")
    nucleus_std_db: float = float("nan")
    assessment: str = ""


@dataclass
class PitchMetrics:
    """Metrics for pitch/F0 analysis (Praat-equivalent)."""
    mean_hz: float = float("nan")
    std_hz: float = float("nan")
    min_hz: float = float("nan")
    max_hz: float = float("nan")
    range_hz: float = float("nan")
    median_hz: float = float("nan")
    quantile_25_hz: float = float("nan")
    quantile_75_hz: float = float("nan")
    coefficient_of_variation: float = float("nan")
    mean_semitones: float = float("nan")
    std_semitones: float = float("nan")
    range_semitones: float = float("nan")
    voiced_frames_percent: float = float("nan")
    unvoiced_frames_percent: float = float("nan")
    voiced_to_total_ratio: float = float("nan")
    octave_repair_fraction: float = float("nan")   # fraction of frames corrected for octave errors
    assessment: str = ""


@dataclass
class VoiceQualityMetrics:
    """Voice quality metrics (Praat Voice Report equivalent)."""
    jitter_local_percent: float = float("nan")
    jitter_local_abs_sec: float = float("nan")
    jitter_rap_percent: float = float("nan")
    jitter_ppq5_percent: float = float("nan")
    jitter_ddp_percent: float = float("nan")
    shimmer_local_percent: float = float("nan")
    shimmer_local_db: float = float("nan")
    shimmer_apq3_percent: float = float("nan")
    shimmer_apq5_percent: float = float("nan")
    shimmer_apq11_percent: float = float("nan")
    shimmer_dda_percent: float = float("nan")
    hnr_db: float = float("nan")
    nhr: float = float("nan")
    voice_breaks_count: int = 0
    voice_breaks_degree_percent: float = float("nan")
    num_pulses: int = 0
    num_periods: int = 0
    mean_autocorrelation: float = float("nan")
    fraction_unvoiced_percent: float = float("nan")
    assessment: str = ""
    pathology_indicators: List[str] = field(default_factory=list)
    measured: bool = False
    # v20: defaults to False. 'reliable' is an assertion about a measurement
    # that has not been taken yet on a default-constructed object, and the
    # optimistic default let an unmeasured voice look trustworthy.
    reliable: bool = False


@dataclass
class FormantMetrics:
    """Formant analysis metrics."""
    f1_mean_hz: float = float("nan")
    f1_std_hz: float = float("nan")
    f1_min_hz: float = float("nan")
    f1_max_hz: float = float("nan")
    f2_mean_hz: float = float("nan")
    f2_std_hz: float = float("nan")
    f2_min_hz: float = float("nan")
    f2_max_hz: float = float("nan")
    f3_mean_hz: float = float("nan")
    f3_std_hz: float = float("nan")
    f3_min_hz: float = float("nan")
    f3_max_hz: float = float("nan")
    f4_mean_hz: float = float("nan")
    f4_std_hz: float = float("nan")
    vowel_space_area: float = float("nan")
    formant_centralization_ratio: float = float("nan")
    f2_f1_ratio_mean: float = float("nan")
    # v20: defaults to False. track_ok is set explicitly to True on the
    # success path, so the only objects that keep the default are the ones
    # returned by the early-exit and exception paths - where the track was
    # precisely NOT ok.
    track_ok: bool = False         # False when F3 <= F2 (LPC order/ceiling failure)
    max_formant_hz_used: float = float("nan")
    assessment: str = ""
    # per-frame clouds (used by segmentation-free dispersion); not exported
    _f1_values: List[float] = field(default_factory=list)
    _f2_values: List[float] = field(default_factory=list)


@dataclass
class SpectralMetrics:
    """
    Spectral analysis metrics.

    v19: the LTAS-derived fields default to NaN rather than 0.0. A 0 dB alpha
    ratio, a 0 dB Hammarberg index and a 0 dB/kHz tilt are all physically
    possible values, so initialising them to 0.0 made a failed Praat call
    indistinguishable from a measurement and let it into the medians and plots.
    """
    spectral_centroid_mean_hz: float = float("nan")
    spectral_centroid_std_hz: float = float("nan")
    spectral_spread_mean_hz: float = float("nan")
    spectral_skewness_mean: float = float("nan")
    spectral_kurtosis_mean: float = float("nan")
    spectral_slope: float = float("nan")
    spectral_tilt_db: float = float("nan")
    ltas_slope: float = float("nan")
    alpha_ratio: float = float("nan")
    hammarberg_index: float = float("nan")
    # NOTE ON THE NAME (v19): this is a smoothed cepstral peak prominence
    # (Praat PowerCepstrogram "Get CPPS"), not the unsmoothed CPP. The column is
    # kept as cpp_db for continuity with earlier runs, but it is a SECOND CPPS
    # computed with different settings and over a different span than cpps_db -
    # see analyze_spectral(). cpp_source records which of the two code paths
    # produced it, because the FFT fallback is not on the same scale at all.
    cpp_db: float = float("nan")
    cpp_source: str = ""
    tilt_band_hz: Tuple[float, float] = (float("nan"), float("nan"))
    span_source: str = ""
    assessment: str = ""


@dataclass
class RhythmMetrics:
    """Rhythm and temporal metrics."""
    total_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    articulation_time_sec: float = 0.0
    phonation_time_sec: float = 0.0
    speech_rate_syllables_per_sec: float = 0.0
    articulation_rate_syllables_per_sec: float = 0.0
    npvi_v: Optional[float] = None
    rpvi_c: Optional[float] = None
    percent_v: float = 0.0
    varco_v: float = 0.0
    delta_v: float = 0.0
    delta_c: float = 0.0
    assessment: str = ""
    estimated_syllable_count: int = 0


@dataclass
class ReadingMetrics:
    """Reading-task additions for the DBS dysarthria study (connected speech)."""
    syllable_count_used: int = 0
    syllable_source: str = ""          # 'known_passage' | 'estimated'
    passage_name: str = ""
    speaking_time_fraction: float = float("nan")   # artic_time / total_speech, 0-1
    ems_3_8hz_ratio: float = float("nan")
    ems_peak_freq_hz: float = float("nan")
    ems_4_to_lowband_ratio: float = float("nan")
    formant_dispersion_logarea: float = float("nan")
    f1f2_cloud_spread: float = float("nan")
    intensity_decay_db: float = float("nan")
    # v19: the duration-free twin. intensity_decay_db is dB across the
    # analysed span, so it only compares between takes of the SAME length -
    # true for one fixed passage, false as soon as the passage or the
    # recording length changes.
    intensity_decay_db_per_s: float = float("nan")
    hnr_median_perinterval_db: float = float("nan")
    cpps_db: float = float("nan")
    cpps_source: str = ""
    # v16: cross-check on the hand-entered KNOWN_PASSAGE_SYLLABLES constant
    syllable_count_estimated: int = 0
    syllable_count_agreement: float = float("nan")   # estimate / constant
    assessment: str = ""


@dataclass
class SustainedVowelMetrics:
    """
    Sustained-phonation metrics. Perturbation measures are taken from the
    STEADY-STATE MID-WINDOW of each token (onset/offset excluded), not from the
    whole file, which is the standard requirement for jitter/shimmer/HNR/CPPS.
    """
    n_tokens: int = 0
    token_durations_s: List[float] = field(default_factory=list)
    token_spans_s: List[Tuple[float, float]] = field(default_factory=list)
    n_splices_detected: int = 0
    splice_times_s: List[float] = field(default_factory=list)
    n_windows_total: int = 0
    n_windows_valid: int = 0
    window_rejections: str = ""
    # v9 audit trail / usability
    rej_low_voicing: int = 0
    rej_f0_outlier: int = 0
    rej_f0_step: int = 0
    rej_splice: int = 0
    rej_other: int = 0
    measured_total_s: float = float("nan")   # seconds of signal behind the medians
    n_tokens_expected: float = float("nan")  # from breath gaps + edit points + 1
    token_count_mismatch: bool = False
    boundaries_source: str = "inferred"      # 'inferred' | 'file'
    tremor_peak_ratio: float = float("nan")
    # v11: measures that survive when jitter/shimmer do NOT. A voice too
    # aperiodic for perturbation analysis is a FINDING, not missing data, so
    # these are computed over the whole token for every file.
    voiced_fraction: float = float("nan")   # in-range voiced frames / all frames
    window_yield: float = float("nan")      # valid windows / tiles that would fit
    # ---- v18: HOW MUCH OF THE PHONATION SURVIVED WINDOW SELECTION ----------
    # window_yield counts WINDOWS, which answers "was the median well supported"
    # but not "was the vowel well sustained": the same count comes out of a 7 s
    # and a 25 s take, and a window is lost to geometry as easily as to a
    # wobbling voice. These account for the phonation in SECONDS instead.
    analyzed_fraction: float = float("nan")      # measured_total_s / total_phonation_s
    window_yield_s: float = float("nan")         # measured_total_s / analyzable_total_s
    analyzable_total_s: float = float("nan")     # what the tiling geometry allows at best
    steady_frame_fraction: float = float("nan")  # frame-level, quantisation-free
    longest_steady_run_s: float = float("nan")   # longest uninterrupted steady stretch
    n_steady_stretches: int = 0                  # how fragmented that steadiness is
    # loss by CAUSE (these four + measured_total_s = total_phonation_s)
    discard_edge_s: float = float("nan")         # fixed onset/offset trim
    discard_quantisation_s: float = float("nan") # remainder shorter than one window
    discard_unsteady_s: float = float("nan")     # failed the validity checks
    # loss by POSITION (these four + measured_total_s = total_phonation_s)
    discard_onset_s: float = float("nan")        # before the first measured window
    discard_interior_s: float = float("nan")     # gaps BETWEEN measured windows
    discard_offset_s: float = float("nan")       # after the last measured window
    discard_dead_token_s: float = float("nan")   # tokens that yielded nothing at all
    n_interior_gaps: int = 0
    coverage_note: str = ""
    cpps_token_db: float = float("nan")     # CPPS over the token (no periodicity needed)
    hnr_token_db: float = float("nan")      # per-interval median HNR over the token
    signal_type: int = 0                    # 1 periodic | 2 subharmonic | 3 aperiodic
    signal_type_note: str = ""
    voiced_at_045: float = float("nan")     # voicing probe (settings vs signal)
    voiced_at_020: float = float("nan")
    voiced_low_floor: float = float("nan")
    usable: bool = False
    usable_reasons: str = ""
    # v17: tiered usability. windows_usable governs cepstral/window measures,
    # perturbation_usable additionally governs jitter/shimmer.
    windows_usable: bool = False
    perturbation_usable: bool = False
    windows_usable_reasons: str = ""
    perturbation_usable_reasons: str = ""
    jitter_iqr: float = float("nan")
    jitter_ppq5_iqr: float = float("nan")
    shimmer_apq11_iqr: float = float("nan")
    intensity_decay_db: float = float("nan")     # loudness decay ACROSS the vowel
    f1f2_dispersion_logarea: float = float("nan")  # formant steadiness in-vowel
    f1f2_cloud_spread_hz: float = float("nan")
    shimmer_iqr: float = float("nan")
    hnr_iqr: float = float("nan")
    cpps_iqr: float = float("nan")
    mpt_longest_s: float = float("nan")      # maximum phonation time
    # ---- v19: LONGEST PHONATION WITH NO BREAK IN THE MIDDLE ----------------
    # See longest_unbroken_phonation(). The old value could exceed
    # mpt_longest_s, because it searched across token boundaries; it is now
    # measured inside a single token, requires voicing, and is resolved on a
    # 20 ms envelope so a 50 ms break is actually visible.
    mpt_longest_uninterrupted_s: float = float("nan")
    unbroken_start_s: float = float("nan")
    unbroken_end_s: float = float("nan")
    unbroken_token_index: int = -1
    unbroken_source: str = ""
    n_phonation_breaks: int = 0          # breaks inside tokens, all tokens
    n_breaks_in_best_token: int = 0      # breaks inside the token that won
    unbroken_energy_only_s: float = float("nan")   # same measure, voicing ignored
    break_threshold_db: float = float("nan")
    mpt_mean_s: float = float("nan")
    total_phonation_s: float = float("nan")
    analysis_window_s: float = float("nan")  # length of the window actually measured
    analysis_window_start_s: float = float("nan")
    # v19: the spans the medians were actually taken from, so the formant block
    # can be restricted to the same signal instead of the whole speech span.
    measured_spans_s: List[Tuple[float, float]] = field(default_factory=list)
    # steady-state measures on the longest token
    f0_mean_hz: float = float("nan")
    # v19: the rest of the F0 distribution over the SAME valid windows. Without
    # these, the exported pitch block was inconsistent on a sustained file -
    # pitch_mean_hz came from the windows while pitch_std_hz, pitch_median_hz
    # and pitch_cv stayed whole-span, so pitch_cv no longer equalled
    # pitch_std_hz / pitch_mean_hz in the row a reader was looking at.
    f0_median_hz: float = float("nan")
    f0_std_hz: float = float("nan")
    f0_min_hz: float = float("nan")
    f0_max_hz: float = float("nan")
    f0_cv: float = float("nan")
    f0_range_semitones: float = float("nan")
    f0_sd_semitones: float = float("nan")
    f0_drift_st_per_s: float = float("nan")  # slope of F0 across the window
    intensity_decay_db_per_s: float = float("nan")  # duration-free twin of decay
    jitter_local_percent: float = float("nan")
    shimmer_local_percent: float = float("nan")
    hnr_db: float = float("nan")
    cpps_db: float = float("nan")
    cpps_source: str = ""
    intensity_sd_db: float = float("nan")
    # amplitude/frequency tremor (2-12 Hz modulation of F0)
    tremor_rate_hz: float = float("nan")
    tremor_extent_semitones: float = float("nan")
    # within-file reproducibility across tokens (0 if only one token)
    jitter_sd_across_tokens: float = float("nan")
    shimmer_sd_across_tokens: float = float("nan")
    hnr_sd_across_tokens: float = float("nan")
    measured: bool = False
    assessment: str = ""


@dataclass
class FluencyMetrics:
    """Comprehensive fluency metrics (custom composite)."""
    intensity_stability: float = 0.0
    pitch_stability: float = 0.0
    rhythm_regularity: float = 0.0
    voice_quality_score: float = 0.0
    articulation_score: float = 0.0
    pause_penalty: float = 0.0
    filler_penalty: float = 0.0
    voice_break_penalty: float = 0.0
    overall_fluency: float = 0.0
    clinical_severity: str = ""
    assessment: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RecordingQualityMetrics:
    """Acoustic quality of the recording itself (NOT the speaker's voice)."""
    snr_db: float = float("nan")
    effective_bandwidth_hz: float = float("nan")
    spectral_edge_hz: float = float("nan")
    is_bandlimited: bool = False
    clipping_fraction: float = float("nan")
    noise_floor_db: float = float("nan")
    sample_rate_hz: int = 0
    quality_score: float = float("nan")
    quality_label: str = ""
    vq_reliable: bool = True
    warnings: List[str] = field(default_factory=list)


@dataclass
class SpeechAnalysisResult:
    """Complete analysis results for a single audio file."""
    file_path: str
    file_name: str
    duration_total: float
    duration_speech: float
    speech_start: float
    speech_end: float
    pause_metrics: PauseMetrics
    filler_metrics: FillerMetrics
    intensity_metrics: IntensityMetrics
    pitch_metrics: PitchMetrics
    voice_quality_metrics: VoiceQualityMetrics
    formant_metrics: FormantMetrics
    spectral_metrics: SpectralMetrics
    rhythm_metrics: RhythmMetrics
    fluency_metrics: FluencyMetrics
    reading_metrics: ReadingMetrics = field(default_factory=ReadingMetrics)
    sustained_metrics: 'SustainedVowelMetrics' = field(default_factory=lambda: SustainedVowelMetrics())
    recording_quality: RecordingQualityMetrics = field(default_factory=RecordingQualityMetrics)
    task: str = "reading"
    task_source: str = ""       # 'filename' | 'acoustic' | 'forced'
    analysis_settings: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# READING-METRIC HELPERS (standalone; called from the analyzer)
# =============================================================================

def estimate_syllables_unclamped(
    y: np.ndarray, sr: int, speech_start: float, speech_end: float
) -> int:
    """
    Intensity-envelope peak-picking syllable estimate WITHOUT the old
    duration-based clamp. Used only as the FALLBACK when the passage is unknown.
    """
    from scipy.signal import find_peaks
    s0 = int(speech_start * sr); s1 = int(speech_end * sr)
    seg = y[s0:s1]
    if len(seg) < sr * 0.1:
        return 0
    hop = int(sr * 0.01); flen = int(sr * 0.025)
    rms = librosa.feature.rms(y=seg, frame_length=flen, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    win = max(5, int(0.05 * sr / hop))
    if win % 2 == 0:
        win += 1
    smoothed = np.convolve(rms_db, np.ones(win) / win, mode="same")
    min_dist = int(0.12 * sr / hop)
    peaks, _ = find_peaks(smoothed, distance=min_dist, prominence=3.0,
                          height=np.percentile(smoothed, 20))
    margin = int(0.05 * sr / hop)
    thr = np.percentile(smoothed, 30)
    valid = [p for p in peaks
             if margin < p < len(smoothed) - margin and smoothed[p] > thr]
    count = len(valid)
    dur = max(1e-6, speech_end - speech_start)
    count = int(min(count, int(dur * 9.0)))   # loose physical bound only
    return max(0, count)


def detect_band_limit(psd: np.ndarray, freqs: np.ndarray, sr: int,
                      min_edge_hz: float = 1500.0, min_drop_db: float = 30.0
                      ) -> Tuple[bool, float, float]:
    """
    Detect a real channel band-limit (telephone / codec / resampling) as a
    SPECTRAL DISCONTINUITY, and return (is_bandlimited, edge_hz, drop_db).

    Why not the old test: it located the "edge" where the smoothed PSD fell
    35 dB below the spectral PEAK. That point moves with natural spectral tilt,
    so on sustained vowels - which have no fricative or burst energy up top and
    a steep glottal roll-off - it landed anywhere between 1.4 and 4.1 kHz on
    recordings of the same speaker, and any edge under 4200 Hz with a 15 dB
    slope was then declared a telephone channel.

    A genuine cut is a STEP: the level falls by tens of dB within a few hundred
    Hz and then stays pinned at the numerical floor. Natural roll-off is
    gradual and never reaches the floor. So we look for the largest step, and
    require the band above it to sit at the floor.
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if psd.size < 16 or freqs.size != psd.size or np.max(psd) <= 0:
        return False, float(sr / 2.0), 0.0
    db = 10.0 * np.log10(psd / np.max(psd) + 1e-14)
    k = 5
    db = np.convolve(db, np.ones(k) / k, mode="same")
    nyq = sr / 2.0
    floor_db = float(np.percentile(db, 2))
    best_drop, best_edge = 0.0, nyq
    lo_i = int(np.searchsorted(freqs, min_edge_hz))
    hi_i = int(np.searchsorted(freqs, 0.90 * nyq))
    for i in range(lo_i, max(lo_i + 1, hi_i)):
        f_i = freqs[i]
        before = db[(freqs >= f_i - 400.0) & (freqs <= f_i)]
        after = db[(freqs >= f_i + 200.0) & (freqs <= f_i + 1200.0)]
        if before.size < 3 or after.size < 3:
            continue
        drop = float(np.median(before) - np.median(after))
        if drop > best_drop:
            best_drop, best_edge = drop, float(f_i)
            best_after = float(np.median(after))
    if best_drop <= 0.0:
        return False, nyq, 0.0
    at_floor = best_after <= floor_db + 8.0
    is_bl = bool(best_drop >= min_drop_db and at_floor and best_edge < 0.90 * nyq)
    return is_bl, best_edge, best_drop


def hz_to_semitones(hz, ref_hz):
    """Semitone conversion relative to ref_hz (NaN-safe)."""
    hz = np.asarray(hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 12.0 * np.log2(hz / ref_hz)


def repair_octave_jumps(f0: np.ndarray, tol: float = 0.12
                        ) -> Tuple[np.ndarray, float]:
    """
    Correct Praat octave errors: frames sitting within `tol` (relative) of an
    exact octave (x2, x4, /2, /4) from the robust median are divided back.

    This matters a lot. An uncorrected doubling population inflates F0 mean, F0
    SD and CV (a sustained vowel should have CV < 0.05, not 0.35) AND it
    corrupts the period sequence that jitter and shimmer are computed from.
    Returns (repaired_f0, fraction_of_voiced_frames_repaired).
    """
    f = np.asarray(f0, dtype=float).copy()
    ok = np.isfinite(f) & (f > 0)
    if int(ok.sum()) < 5:
        return f, 0.0
    med = float(np.median(f[ok]))
    n_fixed = 0
    for _ in range(4):
        changed = False
        for i in np.where(ok)[0]:
            for g in (4.0, 2.0, 0.5, 0.25):
                if abs(f[i] / (med * g) - 1.0) <= tol:
                    f[i] = f[i] / g
                    n_fixed += 1
                    changed = True
                    break
        med = float(np.median(f[ok]))
        if not changed:
            break
    return f, float(n_fixed) / float(max(1, int(ok.sum())))


# =============================================================================
# OCTAVE VERIFICATION FROM THE HARMONIC COMB  (v9)
# =============================================================================
#
# WHY: a pitch tracker cannot tell F0 from F0/2 by autocorrelation alone -- both
# lags are periodicities of the same signal. The SPECTRUM can: the harmonics of
# the true F0 are all present, whereas a comb placed at F0/2 has its ODD teeth
# (F0/2, 3F0/2, 5F0/2 ...) sitting on the noise between real harmonics. So:
#
#   comb at the candidate, odd teeth as strong as even teeth  -> candidate is F0
#   comb at the candidate, odd teeth missing                  -> candidate is F0/2
#                                                                (true F0 = 2x)
#
# This is the only cheap check that is independent of the tracker, and it is
# what makes an octave error auditable instead of self-confirming.

def _mid_chunks(y: np.ndarray, sr: int, spans: List[Tuple[float, float]],
                chunk_s: float = 0.25, max_chunks: int = 12,
                edge_trim_s: float = 0.4) -> List[np.ndarray]:
    """
    Short chunks taken from inside the supplied spans, preferring the loudest
    ones (steady phonation rather than onsets, offsets or breaths).
    """
    y = np.asarray(y, dtype=float)
    n = int(round(chunk_s * sr))
    if n < 32 or y.size < n:
        return []
    cands: List[Tuple[float, np.ndarray]] = []
    use_spans = spans if spans else [(0.0, y.size / float(sr))]
    for (s, e) in use_spans:
        a = max(0.0, s + edge_trim_s)
        b = min(y.size / float(sr), e - edge_trim_s)
        if b - a < chunk_s:
            a, b = max(0.0, s), min(y.size / float(sr), e)
            if b - a < chunk_s:
                continue
        t = a
        step = max(chunk_s, (b - a) / 8.0)
        while t + chunk_s <= b + 1e-9:
            i0 = int(round(t * sr))
            seg = y[i0:i0 + n]
            if seg.size == n:
                cands.append((float(np.sqrt(np.mean(seg ** 2))), seg))
            t += step
    if not cands:
        return []
    cands.sort(key=lambda p: -p[0])
    return [seg for _, seg in cands[:max_chunks]]


def _mean_spectrum_db(chunks: List[np.ndarray], sr: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Averaged magnitude spectrum (dB) of the chunks, zero-padded 4x."""
    if not chunks:
        return np.array([]), np.array([])
    n = chunks[0].size
    nfft = int(2 ** np.ceil(np.log2(max(4096, n * 4))))
    win = np.hanning(n)
    acc = None
    for seg in chunks:
        if seg.size != n:
            continue
        sp = np.abs(np.fft.rfft(seg * win, n=nfft))
        acc = sp if acc is None else acc + sp
    if acc is None:
        return np.array([]), np.array([])
    acc /= len(chunks)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    return freqs, 20.0 * np.log10(acc + 1e-12)


def _tooth_prominence(freqs: np.ndarray, db: np.ndarray, f_hz: float,
                      tol_hz: float) -> float:
    """
    Prominence (dB) of one comb tooth: the local spectral peak within +/-tol of
    f_hz, minus the median level of the surrounding band. ~0 dB means "nothing
    there"; a real harmonic stands 10-30 dB out.
    """
    if f_hz <= 0 or freqs.size == 0:
        return float("nan")
    core = (freqs >= f_hz - tol_hz) & (freqs <= f_hz + tol_hz)
    if not np.any(core):
        return float("nan")
    ring = ((freqs >= f_hz - 6.0 * tol_hz) & (freqs <= f_hz + 6.0 * tol_hz)
            & ~core)
    if int(np.count_nonzero(ring)) < 5:
        return float("nan")
    return float(np.max(db[core]) - np.median(db[ring]))


def comb_scores(freqs: np.ndarray, db: np.ndarray, f0: float,
                f_max_hz: float = 3500.0, n_max: int = 16
                ) -> Tuple[float, float, int]:
    """
    Score a harmonic comb placed at f0.

    Returns (mean_prominence_db, odd_minus_even_db, n_teeth_used).

    odd_minus_even is computed PAIRWISE (each odd tooth against its adjacent
    even teeth) so the spectral tilt of the voice cancels out. Interpretation:
        >= -4 dB : odd teeth present -> f0 is a real fundamental
        <= -10 dB: odd teeth missing -> f0 is a subharmonic, true F0 = 2*f0
    """
    if not (np.isfinite(f0) and f0 > 0) or freqs.size == 0:
        return float("nan"), float("nan"), 0
    tol = max(4.0, 0.15 * f0)
    n_h = int(min(n_max, np.floor(f_max_hz / f0)))
    if n_h < 4:
        return float("nan"), float("nan"), 0
    prom = {k: _tooth_prominence(freqs, db, k * f0, tol)
            for k in range(1, n_h + 1)}
    vals = [v for v in prom.values() if np.isfinite(v)]
    if len(vals) < 4:
        return float("nan"), float("nan"), 0
    diffs = []
    for k in range(1, n_h + 1, 2):
        if not np.isfinite(prom.get(k, np.nan)):
            continue
        neigh = [prom.get(k - 1, np.nan), prom.get(k + 1, np.nan)]
        neigh = [v for v in neigh if np.isfinite(v)]
        if neigh:
            diffs.append(prom[k] - float(np.mean(neigh)))
    odd_even = float(np.mean(diffs)) if diffs else float("nan")
    return float(np.mean(vals)), odd_even, len(vals)


def resolve_f0_octave(y: np.ndarray, sr: int, spans: List[Tuple[float, float]],
                      f0_ref: float,
                      strong_db: float = -10.0, marginal_db: float = -4.0,
                      min_prominence_db: float = 3.0,
                      f0_min: float = 45.0, f0_max: float = 600.0
                      ) -> Dict[str, Any]:
    """
    Verify a candidate F0 against the harmonic comb of the signal.

    verdict:
      'ok'                 comb is consistent with f0_ref
      'tracker_halved'     odd teeth missing at f0_ref -> true F0 = 2*f0_ref
      'tracker_doubled'    a full comb also exists at f0_ref/2 -> true F0 = f0_ref/2
      'ambiguous'          partial subharmonic energy (period doubling / diplophonia):
                           NOT auto-corrected, must be checked by ear
      'inconclusive'       no usable harmonic structure (too noisy / too short)

    Only 'tracker_halved' and 'tracker_doubled' change f0_true.
    """
    out: Dict[str, Any] = dict(
        f0_ref=float(f0_ref) if np.isfinite(f0_ref) else float("nan"),
        f0_true=float(f0_ref) if np.isfinite(f0_ref) else float("nan"),
        factor=1.0, verdict="inconclusive",
        score_ref_db=float("nan"), score_half_db=float("nan"),
        prominence_ref_db=float("nan"), prominence_half_db=float("nan"),
        note="")
    if not (np.isfinite(f0_ref) and f0_min <= f0_ref <= f0_max):
        out["note"] = "reference F0 outside plausible range"
        return out
    chunks = _mid_chunks(y, sr, spans)
    freqs, db = _mean_spectrum_db(chunks, sr)
    if freqs.size == 0:
        out["note"] = "no usable audio chunk for the comb test"
        return out
    prom_ref, score_ref, n_ref = comb_scores(freqs, db, f0_ref)
    out["prominence_ref_db"] = prom_ref
    out["score_ref_db"] = score_ref
    if not np.isfinite(prom_ref) or prom_ref < min_prominence_db:
        out["note"] = (f"harmonic structure too weak to judge "
                       f"(mean tooth prominence {prom_ref:.1f} dB)")
        return out
    half = f0_ref / 2.0
    if half >= f0_min:
        prom_half, score_half, _ = comb_scores(freqs, db, half)
        out["prominence_half_db"] = prom_half
        out["score_half_db"] = score_half
    else:
        prom_half = score_half = float("nan")

    # (1) odd teeth missing at the candidate -> the candidate IS a subharmonic
    if np.isfinite(score_ref) and score_ref <= strong_db:
        out.update(verdict="tracker_halved", f0_true=2.0 * f0_ref, factor=2.0,
                   note=(f"odd harmonics of {f0_ref:.0f} Hz are "
                         f"{abs(score_ref):.0f} dB weaker than the even ones: "
                         f"{f0_ref:.0f} Hz is a subharmonic, true F0 "
                         f"~{2*f0_ref:.0f} Hz"))
        return out
    # (2) a complete comb also exists an octave BELOW -> the tracker doubled
    if (np.isfinite(score_half) and np.isfinite(prom_half)
            and score_half >= marginal_db and prom_half >= min_prominence_db):
        out.update(verdict="tracker_doubled", f0_true=half, factor=0.5,
                   note=(f"a full harmonic comb also fits {half:.0f} Hz "
                         f"(odd/even {score_half:+.1f} dB): the tracker doubled, "
                         f"true F0 ~{half:.0f} Hz"))
        return out
    # (3) partial subharmonic energy: real period doubling, or a marginal call
    if np.isfinite(score_ref) and score_ref < marginal_db:
        out.update(verdict="ambiguous",
                   note=(f"odd harmonics of {f0_ref:.0f} Hz are "
                         f"{abs(score_ref):.0f} dB down (partial subharmonic / "
                         f"period doubling): CHECK BY EAR, not auto-corrected"))
        return out
    out.update(verdict="ok",
               note=(f"comb consistent with {f0_ref:.0f} Hz "
                     f"(odd/even {score_ref:+.1f} dB)"))
    return out


def voicing_probe(sound, floor: float, ceiling: float,
                  thresholds=(0.45, 0.30, 0.20),
                  low_floor: float = 60.0) -> Dict[str, float]:
    """
    Answer the question "is this voice really unvoiced, or is the tracker just
    refusing it?" (v12)

    A rough/hoarse but continuously produced vowel can be a TYPE 3 signal in the
    Titze sense - no reliably identifiable cycles - and an autocorrelation
    tracker then returns 'unvoiced' for most frames even though the speaker never
    stopped phonating. That is a property of the signal, not a bug, but it must
    be distinguished from a threshold that is merely set too high.

    Returns the voiced fraction at several voicing thresholds, plus one pass with
    a much lower floor (to check the range is not the culprit). If the fraction
    stays low at 0.20, the signal really is aperiodic.
    """
    out: Dict[str, float] = {}
    for thr in thresholds:
        try:
            p = call(sound, "To Pitch (ac)", 0.0, floor, 15, "no",
                     0.03, float(thr), 0.01, 0.35, 0.14, ceiling)
            n = call(p, "Get number of frames")
            v = call(p, "Count voiced frames")
            out[f"voiced_at_{thr:.2f}"] = (float(v) / float(n)) if n else float("nan")
        except Exception:
            out[f"voiced_at_{thr:.2f}"] = float("nan")
    try:
        p = call(sound, "To Pitch (ac)", 0.0, low_floor, 15, "no",
                 0.03, 0.45, 0.01, 0.35, 0.14, ceiling)
        n = call(p, "Get number of frames")
        v = call(p, "Count voiced frames")
        out["voiced_low_floor"] = (float(v) / float(n)) if n else float("nan")
    except Exception:
        out["voiced_low_floor"] = float("nan")
    return out


def interpret_probe(probe: Dict[str, float], hnr_token_db: float,
                    cpps_token_db: float) -> str:
    """
    Interpret the voicing probe (v14).

    The v13 message was one-sided and wrong: it concluded "SETTINGS were the
    limit" whenever the voiced fraction rose above ~60% at a permissive
    threshold. But the voicing threshold IS the normalised autocorrelation
    strength, so lowering it does not reveal hidden periodicity - it accepts
    WEAKER periodicity. Whether that is legitimate depends on how much harmonic
    energy is actually there:

        voiced rises AND HNR/CPPS healthy -> the old threshold really was too
                                             strict; the frames are usable
        voiced rises BUT HNR ~ 0-6 dB     -> harmonic and noise energy are
                                             comparable. Praat can find a
                                             candidate period in noise; jitter
                                             computed there is the variability of
                                             the TRACKER, not of the voice.

    On this dataset the second case is what the follow-up looks like: 75-96%
    voiced at threshold 0.20 with HNR 1-5 dB.
    """
    v020 = probe.get("voiced_at_0.20", float("nan"))
    if not np.isfinite(v020):
        return ""
    hnr = hnr_token_db if np.isfinite(hnr_token_db) else 0.0
    cpps = cpps_token_db if np.isfinite(cpps_token_db) else 0.0
    if v020 < 0.60:
        return ("the SIGNAL is aperiodic (still unvoiced even at a permissive "
                "threshold)")
    # calibrated on this dataset: HNR ~17 dB with CPPS ~13 (0309 PRE-01) is a
    # voice whose harmonics clearly dominate and whose frames are usable, while
    # HNR 1-6 dB (follow-up, ON-01) is not. HNR carries most of the decision;
    # CPPS only guards against a globally degraded recording.
    if hnr >= 12.0 and cpps >= 12.0:
        return ("the THRESHOLD was the limit and the periodicity is strong "
                f"(HNR {hnr:.0f} dB): these frames are usable, consider "
                "voicing_threshold=0.40")
    return (f"periodic candidates exist but are WEAK (HNR {hnr:.0f} dB, CPPS "
            f"{cpps:.0f} dB): do NOT lower the threshold to recover them - "
            "jitter measured there is tracker noise, not voice")


def classify_signal_type(voiced_fraction: float, hnr_token_db: float,
                         cpps_token_db: float, subharm_lock: int,
                         voiced_at_020: float = float("nan")
                         ) -> Tuple[int, str]:
    """
    Titze-style signal typing, which decides WHICH measures are even valid (v12).

        type 1  nearly periodic            -> jitter/shimmer/HNR all valid
        type 2  subharmonics / modulation  -> perturbation unreliable, cepstral OK
        type 3  aperiodic, no clear cycles -> perturbation INVALID by definition;
                                              report cepstral/spectral measures only

    This is the standard reason a severely dysphonic voice has no jitter value:
    jitter is the cycle-to-cycle variation of cycles you can identify, so when
    the cycles cannot be identified the quantity does not exist. Forcing a number
    out of it does not produce a worse-but-comparable value - it produces the
    variability of the TRACKER, which is not comparable to anything.

    v17 - THE VOICED-FRACTION TRIGGER NO LONGER FIRES ALONE.
    The old rule was `vf < 0.55 or hnr < 7 or cpps < 12` -> type 3. Because it was
    an OR over a hard threshold, a voiced fraction of 0.54 forced "perturbation
    UNDEFINED" while HNR 17.2 dB and CPPS 13.2 dB - both strong, both well clear of
    their own triggers - were ignored, and 0.56 on the same file would have given
    type 2. One hundredth of a proportion should not flip a categorical validity
    verdict, and voiced fraction at a FIXED voicing threshold is the weakest of the
    three inputs: it drops when the tracker's threshold is too strict, not only
    when the voice is aperiodic.

    Now: HNR and CPPS (properties of the harmonic structure itself) still trigger
    type 3 on their own. Low voiced fraction triggers type 3 only when the harmonic
    evidence does NOT contradict it - and if voiced_at_020 is supplied and shows the
    frames were recoverable at a permissive threshold, the low fraction is treated
    as a settings artifact rather than aperiodicity.
    """
    vf = voiced_fraction if np.isfinite(voiced_fraction) else 0.0
    hnr = hnr_token_db if np.isfinite(hnr_token_db) else 0.0
    cpps = cpps_token_db if np.isfinite(cpps_token_db) else 0.0
    # Is the low voiced fraction explainable as a threshold artifact rather than
    # as aperiodicity? Requires the permissive-threshold probe to recover most of
    # the signal AND the harmonic structure to be strong.
    threshold_artifact = (np.isfinite(voiced_at_020) and voiced_at_020 >= 0.85
                          and hnr >= 12.0 and cpps >= 12.0)
    if vf >= 0.85 and hnr >= 12.0 and not subharm_lock:
        return 1, "type 1 (nearly periodic): perturbation measures valid"
    # Harmonic-structure failures stand on their own.
    if hnr < 7.0 or cpps < 12.0:
        return 3, ("type 3 (aperiodic / severely dysphonic): jitter, shimmer and "
                   "window HNR are UNDEFINED, not merely worse. Compare these files "
                   "on CPPS/CPP, spectral tilt and voiced fraction instead")
    # Low voicing only counts when nothing contradicts it.
    if vf < 0.55 and not threshold_artifact:
        return 3, (f"type 3 (aperiodic: only {vf*100:.0f}% of the token is periodic, "
                   "and the permissive-threshold probe did not recover it): jitter, "
                   "shimmer and window HNR are UNDEFINED, not merely worse. Compare "
                   "these files on CPPS/CPP, spectral tilt and voiced fraction instead")
    if vf < 0.55 and threshold_artifact:
        return 2, (f"type 2 (only {vf*100:.0f}% voiced at the default threshold, but "
                   f"{voiced_at_020*100:.0f}% at a permissive one with HNR {hnr:.0f} dB "
                   f"and CPPS {cpps:.0f} dB - a THRESHOLD artifact, not aperiodicity): "
                   "perturbation measures unreliable, cepstral and spectral preferred")
    if subharm_lock:
        return 2, ("type 2 (subharmonic energy / period doubling detected): "
                   "perturbation measures unreliable, cepstral and spectral "
                   "measures preferred")
    return 2, ("type 2 (not clean enough for type 1, harmonic structure still "
               "intact): perturbation measures unreliable, cepstral and spectral "
               "measures preferred")


def read_boundaries_file(path: Path) -> List[Tuple[float, float]]:
    """
    Read token boundaries supplied by the user instead of inferring them.

    Accepts either:
      - a CSV/TSV with two numeric columns per row: onset_s, offset_s
        (a header row is tolerated and skipped)
      - a Praat .TextGrid: the xmin/xmax of every non-empty interval of the
        first interval tier
    Returns [] when the file cannot be parsed.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    spans: List[Tuple[float, float]] = []
    if Path(path).suffix.lower() == ".textgrid":
        xmins = [float(m) for m in re.findall(r"xmin\s*=\s*([0-9.eE+-]+)", text)]
        xmaxs = [float(m) for m in re.findall(r"xmax\s*=\s*([0-9.eE+-]+)", text)]
        texts = re.findall(r'text\s*=\s*"([^"]*)"', text)
        # first two xmin/xmax are the file and the tier bounds
        iv_min, iv_max = xmins[2:], xmaxs[2:]
        for k, (a, b) in enumerate(zip(iv_min, iv_max)):
            label = texts[k].strip() if k < len(texts) else ""
            if label and b > a:
                spans.append((float(a), float(b)))
        return spans
    for line in text.splitlines():
        parts = re.split(r"[,;\t ]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue                      # header or comment
        if b > a >= 0:
            spans.append((a, b))
    return spans


def segment_voiced_tokens(times: np.ndarray, voiced: np.ndarray,
                          min_token_s: float = 0.8,
                          max_bridge_gap_s: float = 0.15
                          ) -> List[Tuple[float, float]]:
    """
    Group a boolean voicing track into tokens: unvoiced gaps shorter than
    max_bridge_gap_s are bridged (they are creak/tracker dropouts, not the end
    of a production), then only runs of at least min_token_s are kept.

    Used to split a multi-trial sustained-vowel recording into the individual
    /a/ productions, so that MPT is per-token and perturbation measures are not
    averaged across breath resets.
    """
    times = np.asarray(times, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    if times.size == 0 or times.size != voiced.size:
        return []
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    runs = []
    i = 0
    n = len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j + 1 < n and voiced[j + 1]:
                j += 1
            runs.append([times[i], times[j] + dt])
            i = j + 1
        else:
            i += 1
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= max_bridge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged if (e - s) >= min_token_s]


def segment_phonation_tokens(rms_times: np.ndarray, rms_db: np.ndarray,
                             silence_threshold_db: float,
                             pitch_times: np.ndarray, voiced: np.ndarray,
                             speech_start: float, speech_end: float,
                             min_token_s: float = 0.8,
                             bridge_gap_s: float = 0.35,
                             min_voiced_fraction: float = 0.5
                             ) -> List[Tuple[float, float]]:
    """
    Split a multi-trial sustained-vowel file into individual productions.

    A production ends when the speaker STOPS MAKING SOUND (to breathe), so the
    boundary must come from the AMPLITUDE envelope, not from the pitch tracker's
    voicing decision. Inside a long /a/ a dysarthric or fatiguing voice goes
    creaky, diplophonic or briefly aperiodic and Praat drops voicing for
    0.2-0.5 s while the amplitude never falls - no breath was taken, so this is
    still one token. Segmenting on voicing therefore over-counts productions
    (one 3-repetition file was reported as 10 tokens).

    Energy runs above `silence_threshold_db` are bridged across gaps shorter
    than `bridge_gap_s`; runs shorter than `min_token_s` are dropped; and each
    surviving run must be at least `min_voiced_fraction` voiced, which rejects
    coughs, throat-clears and chair noise that carry energy but no periodicity.
    """
    rms_times = np.asarray(rms_times, dtype=float)
    rms_db = np.asarray(rms_db, dtype=float)
    if rms_times.size == 0 or rms_times.size != rms_db.size:
        return []
    dt = float(np.median(np.diff(rms_times))) if rms_times.size > 1 else 0.032
    inside = (rms_times >= speech_start - dt) & (rms_times <= speech_end + dt)
    loud = inside & (rms_db >= silence_threshold_db)
    runs = []
    i, n = 0, len(loud)
    while i < n:
        if loud[i]:
            j = i
            while j + 1 < n and loud[j + 1]:
                j += 1
            runs.append([rms_times[i], rms_times[j] + dt])
            i = j + 1
        else:
            i += 1
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= bridge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pitch_times = np.asarray(pitch_times, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    tokens = []
    for s, e in merged:
        if (e - s) < min_token_s:
            continue
        if pitch_times.size:
            m = (pitch_times >= s) & (pitch_times <= e)
            if np.any(m) and float(np.mean(voiced[m])) < min_voiced_fraction:
                continue
        tokens.append((float(s), float(e)))
    return tokens


def _fine_envelope_db(y: np.ndarray, sr: int, win_s: float = 0.020,
                      hop_s: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-frame RMS envelope in dB relative to the peak of THIS signal.

    The script's global envelope is 2048 samples long (46 ms at 44.1 kHz), which
    is wider than the shortest event break detection has to resolve: a 50 ms
    glottal stop falls inside one frame and is averaged away together with the
    phonation on either side of it. Break detection therefore computes its own
    envelope (20 ms frame, 5 ms hop) instead of inheriting that one.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 8 or sr <= 0:
        return np.array([]), np.array([])
    n_fft = max(64, int(round(win_s * sr)))
    hop = max(8, int(round(hop_s * sr)))
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop,
                              center=True)[0]
    t = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    peak = float(np.max(rms)) if rms.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return t, np.full(rms.shape, -120.0)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    return t, db


def _mask_runs(mask: np.ndarray, times: np.ndarray, dt: float
               ) -> List[List[float]]:
    """Contiguous True runs of `mask` as [start_s, end_s] pairs."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[cuts + 1]))
    ends = np.concatenate((idx[cuts], [idx[-1]]))
    return [[float(times[a] - dt / 2.0), float(times[b] + dt / 2.0)]
            for a, b in zip(starts, ends)]


def _split_runs_at(runs: List[List[float]], cuts) -> List[List[float]]:
    """Break any run that straddles a cut time into two runs."""
    cuts = sorted(float(c) for c in (cuts or []))
    if not cuts:
        return [list(r) for r in runs]
    out = []
    for a, b in runs:
        pieces, prev = [], a
        for c in cuts:
            if a < c < b:
                pieces.append([prev, c]); prev = c
        pieces.append([prev, b])
        out.extend(pieces)
    return out


def _bridge_runs(runs: List[List[float]], max_gap_s: float) -> List[List[float]]:
    """Join runs separated by less than max_gap_s; the rest stay separate."""
    merged: List[List[float]] = []
    for a, b in sorted(runs, key=lambda r: r[0]):
        if merged and (a - merged[-1][1]) < max_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def longest_unbroken_phonation(
        y: np.ndarray, sr: int,
        tokens: List[Tuple[float, float]],
        pitch_times: Optional[np.ndarray] = None,
        voiced: Optional[np.ndarray] = None,
        splices=(),
        min_break_s: float = 0.05,
        drop_db: float = 15.0,
        require_voicing: bool = True,
        env_win_s: float = 0.020,
        env_hop_s: float = 0.005) -> Dict[str, Any]:
    """
    Longest stretch of CONTINUOUS phonation, with no break in the middle.

    Replaces longest_uninterrupted_sound(), which was wrong in three ways and
    is the reason this metric could exceed sv_mpt_longest_s:

      1. IT SEARCHED ACROSS TOKENS. The span handed to it ran from the first
         token's start to the last token's end, so two productions separated by
         a breath that never dropped below the threshold were reported as one
         unbroken sound. A "longest uninterrupted phonation" longer than the
         longest token is not a measurement, it is a segmentation failure, and
         on a multi-trial file it inflated the number without limit. The search
         is now strictly WITHIN one token, and clamped to that token's duration.
      2. IT WAS ENERGY-ONLY. Any frame above the silence line counted, so a
         breathy unvoiced exhale, a whispered tail or room noise at phonation
         level extended the run. Phonation requires the vocal folds to be
         vibrating, so voicing is now required as well (require_voicing).
      3. IT INHERITED THE 46 ms ENVELOPE. See _fine_envelope_db: a break
         shorter than one frame was averaged away. It now uses a 20 ms / 5 ms
         envelope of its own.

    A BREAK is defined explicitly, because "no break in the middle" is not
    self-defining: a gap of at least min_break_s (default 50 ms) in which the
    signal is either below the phonation threshold or unvoiced. Shorter
    dropouts - one creaky period, a single mistracked frame - are bridged, since
    calling them breaks would make the metric a measure of tracker noise. A
    detected splice always breaks a run regardless of duration.

    Returns a dict (never raises):
      longest_s              seconds, NaN if nothing measurable
      start_s, end_s         where that stretch sits in the file
      token_index            which token it came from
      n_breaks_total         breaks inside tokens, summed over all tokens
      n_breaks_in_best       breaks inside the token that produced the maximum
      longest_energy_only_s  same measure without the voicing requirement, as a
                             diagnostic: a large gap between the two means the
                             phonation is continuous in ENERGY but keeps losing
                             voicing, i.e. aphonic breaks rather than pauses
      threshold_db           the phonation threshold actually used
      source                 which definition produced longest_s
    """
    out: Dict[str, Any] = dict(
        longest_s=float("nan"), start_s=float("nan"), end_s=float("nan"),
        token_index=-1, n_breaks_total=0, n_breaks_in_best=0,
        longest_energy_only_s=float("nan"), threshold_db=float("nan"),
        source="none")
    try:
        toks = [(float(a), float(b)) for a, b in (tokens or []) if b > a]
        if not toks:
            return out
        t_env, db_env = _fine_envelope_db(y, sr, env_win_s, env_hop_s)
        if t_env.size < 3:
            return out
        dt = float(np.median(np.diff(t_env)))
        if not np.isfinite(dt) or dt <= 0:
            return out

        in_tok = np.zeros(t_env.shape, dtype=bool)
        for a, b in toks:
            in_tok |= (t_env >= a) & (t_env <= b)
        if not np.any(in_tok):
            return out

        # Threshold referenced to the LOCAL phonation level inside the tokens,
        # not to the file peak: on an edited file the peak may sit in a token
        # that is not the one being measured.
        in_db = db_env[in_tok]
        loud = in_db[in_db >= np.percentile(in_db, 60)]
        level = float(np.median(loud)) if loud.size else float(np.median(in_db))
        threshold = level - float(drop_db)
        out["threshold_db"] = threshold
        above = db_env >= threshold

        # Voicing, resampled from the pitch grid onto the envelope grid by
        # nearest neighbour (the pitch grid is coarser, so interpolating a
        # boolean would invent half-voiced frames).
        voiced_env = np.ones(t_env.shape, dtype=bool)
        have_voicing = False
        if (pitch_times is not None and voiced is not None
                and np.size(pitch_times) >= 2
                and np.size(pitch_times) == np.size(voiced)):
            pt = np.asarray(pitch_times, dtype=float)
            pv = np.asarray(voiced, dtype=bool)
            order = np.argsort(pt)
            pt, pv = pt[order], pv[order]
            j = np.clip(np.searchsorted(pt, t_env), 1, pt.size - 1)
            left_closer = (t_env - pt[j - 1]) <= (pt[j] - t_env)
            voiced_env = np.where(left_closer, pv[j - 1], pv[j])
            have_voicing = True

        def _scan(mask: np.ndarray) -> Tuple[float, float, float, int, int, int]:
            """(longest, start, end, token_index, breaks_total, breaks_in_best)."""
            best_len, best_a, best_b, best_i, best_brk = -1.0, np.nan, np.nan, -1, 0
            brk_total = 0
            for i, (ta, tb) in enumerate(toks):
                sel = mask & (t_env >= ta) & (t_env <= tb)
                runs = _mask_runs(sel, t_env, dt)
                if not runs:
                    continue
                runs = [[max(r[0], ta), min(r[1], tb)] for r in runs
                        if min(r[1], tb) > max(r[0], ta)]
                # ORDER MATTERS. Bridging must come FIRST and splitting SECOND.
                # Done the other way round, splitting a run at a splice leaves
                # two segments that touch at exactly the cut time - a zero-length
                # gap - and the bridge then immediately re-joins them, so the
                # splice had no effect at all.
                merged = _bridge_runs(runs, float(min_break_s))
                merged = _split_runs_at(merged, [float(c) for c in (splices or [])
                                                 if ta < float(c) < tb])
                if not merged:
                    continue
                brk_total += max(0, len(merged) - 1)
                for a, b in merged:
                    if (b - a) > best_len:
                        best_len = b - a
                        best_a, best_b, best_i = a, b, i
                        best_brk = max(0, len(merged) - 1)
            if best_len < 0:
                return float("nan"), float("nan"), float("nan"), -1, brk_total, 0
            # Can never exceed the token it came from.
            ta, tb = toks[best_i]
            best_len = float(min(best_len, tb - ta))
            return best_len, float(best_a), float(best_b), int(best_i), \
                brk_total, int(best_brk)

        e_len, _, _, _, _, _ = _scan(above)
        out["longest_energy_only_s"] = e_len

        if require_voicing and have_voicing:
            v_len, v_a, v_b, v_i, v_tot, v_brk = _scan(above & voiced_env)
            out.update(longest_s=v_len, start_s=v_a, end_s=v_b, token_index=v_i,
                       n_breaks_total=v_tot, n_breaks_in_best=v_brk,
                       source="voiced+energy")
            if not np.isfinite(v_len):
                # No voiced stretch survived; the energy figure is all there is,
                # and it is reported as such rather than silently substituted.
                out["source"] = "energy_only (no voiced stretch found)"
        else:
            e2, e_a, e_b, e_i, e_tot, e_brk = _scan(above)
            out.update(longest_s=e2, start_s=e_a, end_s=e_b, token_index=e_i,
                       n_breaks_total=e_tot, n_breaks_in_best=e_brk,
                       source=("energy_only" if not require_voicing
                               else "energy_only (no pitch track available)"))
        return out
    except Exception:
        return out


def detect_splices(pitch_times: np.ndarray, f0: np.ndarray,
                   rms_times: np.ndarray, rms_db: np.ndarray,
                   spans: List[Tuple[float, float]],
                   bin_s: float = 0.10, side_s: float = 0.30,
                   f0_step_st: float = 1.5, plateau_sd_st: float = 0.8,
                   int_step_db: float = 8.0, persist_s: float = 1.0,
                   cluster_s: float = 0.6,
                   silence_threshold_db: Optional[float] = None,
                   min_gap_s: float = 0.05,
                   edge_guard_s: float = 1.5,
                   gap_drop_db: float = 25.0) -> List[float]:
    """
    Find EDIT POINTS inside apparently continuous phonation.

    When separate productions are cut and concatenated, the silence between them
    is trimmed, so the gap can be shorter than a creaky dropout inside a single
    production - duration-based segmentation then becomes a coin flip. The join
    itself, however, leaves a signature: two STABLE F0 PLATEAUS at different
    levels meeting at one instant (and usually an intensity step too).

    For each candidate instant we compare the `side_s` of signal before and
    after: a splice requires |level difference| >= f0_step_st semitones with
    each side internally stable (SD < plateau_sd_st). Crucially the shift must
    also PERSIST over `persist_s` on both sides - that is what separates a
    splice from a transient creak dip or pitch break, which returns to the
    previous level within a few hundred ms and so fails the long-horizon test.
    An intensity step of >= int_step_db, likewise persistent, also counts.

    The F0 step must also be large RELATIVE to the within-plateau scatter
    (>= 2x the larger side SD), so a 1 st threshold is usable without firing on
    ordinary micro-variation: two productions can differ by little more than a
    semitone and still be separate productions.

    Third and strongest cue: a BRIEF DROP TO THE NOISE FLOOR inside otherwise
    continuous phonation. Aperiodic or creaky stretches keep their amplitude; a
    true silence of even 0.05 s mid-vowel is the trimmed gap left by an edit.
    Pass `silence_threshold_db` to enable this cue.

    Detections within `cluster_s` are merged and reported at their midpoint,
    because a trimmed join produces a hit on each side of the removed silence.

    Returns one split time per detected join.
    """
    pitch_times = np.asarray(pitch_times, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    rms_times = np.asarray(rms_times, dtype=float)
    rms_db = np.asarray(rms_db, dtype=float)
    out: List[List[float]] = []
    if pitch_times.size < 10:
        return []
    def _median_in(tt, vv, a, b, need=5):
        m = (tt >= a) & (tt <= b)
        v = vv[m]
        v = v[np.isfinite(v) & (v > 0)] if vv is f0 else v[np.isfinite(v)]
        return (float(np.median(v)), v) if v.size >= need else (None, v)

    hits: List[float] = []
    for (s, e) in spans:
        if (e - s) < 2 * side_s + 2 * bin_s:
            continue
        # Onsets and offsets are unstable by nature; a "join" found 1 s into a
        # token is almost always the voice settling, not an edit. Requiring
        # edge_guard_s of context on each side also stops a useless 1-2 s stub
        # being split off the front of a production.
        t = max(s + side_s, s + edge_guard_s)
        t_end = min(e - side_s, e - edge_guard_s)
        while t <= t_end:
            is_splice = False
            # ---- F0 level shift, stable on both sides and persistent --------
            lo, pre = _median_in(pitch_times, f0, t - side_s, t)
            hi, post = _median_in(pitch_times, f0, t, t + side_s)
            if lo is not None and hi is not None and lo > 0 and hi > 0:
                step = abs(12.0 * np.log2(hi / lo))
                sd_pre = float(np.std(hz_to_semitones(pre, lo)))
                sd_post = float(np.std(hz_to_semitones(post, hi)))
                stable = (sd_pre < plateau_sd_st and sd_post < plateau_sd_st)
                # discriminability: the step must clear the local scatter
                separable = step >= max(f0_step_st, 2.0 * max(sd_pre, sd_post))
                if separable and stable:
                    lo_l, _ = _median_in(pitch_times, f0, max(s, t - persist_s), t)
                    hi_l, _ = _median_in(pitch_times, f0, t, min(e, t + persist_s))
                    if (lo_l is not None and hi_l is not None and lo_l > 0 and hi_l > 0
                            and abs(12.0 * np.log2(hi_l / lo_l)) >= f0_step_st):
                        is_splice = True
            # ---- persistent intensity step ----------------------------------
            if not is_splice and rms_times.size > 2:
                a, _ = _median_in(rms_times, rms_db, t - side_s, t, need=3)
                b, _ = _median_in(rms_times, rms_db, t, t + side_s, need=3)
                if a is not None and b is not None and abs(a - b) >= int_step_db:
                    a_l, _ = _median_in(rms_times, rms_db, max(s, t - persist_s), t, need=3)
                    b_l, _ = _median_in(rms_times, rms_db, t, min(e, t + persist_s), need=3)
                    if (a_l is not None and b_l is not None
                            and abs(a_l - b_l) >= int_step_db):
                        is_splice = True
            # ---- trimmed silence inside phonation ---------------------------
            # FIXED in v9. The old test was
            #     median(+-50 ms) <= silence_threshold_db + 3
            # with silence_threshold_db referenced to the FILE PEAK, i.e. any dip
            # of ~15-19 dB below the loudest moment of the file counted as an
            # edit. Inside a tremulous or fatiguing vowel that happens
            # constantly, which is how 3-token files were split into 5 tokens and
            # MPT lost ~5 s. Now the dip is measured against the LOCAL phonation
            # level (median of the surrounding 0.5 s on each side), must reach
            # gap_drop_db, and must persist for min_gap_s.
            if not is_splice and rms_times.size > 2:
                half = max(min_gap_s, 0.03)
                core = (rms_times >= t - half) & (rms_times <= t + half)
                ring = (((rms_times >= t - 0.6) & (rms_times < t - half)) |
                        ((rms_times > t + half) & (rms_times <= t + 0.6)))
                if int(np.count_nonzero(core)) >= 2 and int(np.count_nonzero(ring)) >= 4:
                    local_level = float(np.median(rms_db[ring]))
                    dip = local_level - float(np.max(rms_db[core]))
                    floor_ok = True
                    if silence_threshold_db is not None:
                        # and it really has to be down at the silence line, not
                        # merely quieter than its neighbourhood
                        floor_ok = float(np.max(rms_db[core])) <= silence_threshold_db + 3.0
                    if dip >= gap_drop_db and floor_ok:
                        is_splice = True
            if is_splice:
                hits.append(float(t))
            t += bin_s

    # ---- cluster: a trimmed join fires on both sides of the removed silence -
    for h in sorted(hits):
        if out and h - out[-1][-1] <= cluster_s:
            out[-1].append(h)
        else:
            out.append([h])
    return [float(np.mean(g)) for g in out]


def split_spans_at(spans: List[Tuple[float, float]], cuts: List[float],
                   min_piece_s: float = 0.8) -> List[Tuple[float, float]]:
    """Split spans at the given instants, dropping pieces below min_piece_s."""
    out = []
    for (s, e) in spans:
        pts = [s] + sorted(c for c in cuts if s < c < e) + [e]
        for a, b in zip(pts[:-1], pts[1:]):
            if (b - a) >= min_piece_s:
                out.append((float(a), float(b)))
    return out


def tile_windows(spans: List[Tuple[float, float]], win_s: float = 3.0,
                 hop_s: float = 3.0, edge_trim_s: float = 0.25,
                 min_window_s: float = 1.0) -> List[Tuple[float, float]]:
    """
    Lay non-overlapping analysis windows across every token, after trimming
    onset/offset. A token too short for a full window contributes one centred
    window if at least min_window_s survives the trim.
    """
    out = []
    for (s, e) in spans:
        a, b = s + edge_trim_s, e - edge_trim_s
        if b - a < min_window_s:
            continue
        if b - a < win_s:
            mid = 0.5 * (a + b)
            half = (b - a) / 2.0
            out.append((float(mid - half), float(mid + half)))
            continue
        x = a
        while x + win_s <= b + 1e-9:
            out.append((float(x), float(x + win_s)))
            x += hop_s
    return out


def window_validity(w: Tuple[float, float], pitch_times: np.ndarray, f0: np.ndarray,
                    file_median_f0: float, cuts: List[float],
                    min_voiced_fraction: float = 0.90,
                    max_f0_deviation_st: float = 3.0,
                    max_internal_step_st: float = 1.0) -> Tuple[bool, str]:
    """
    Decide whether one window is a legitimate sample of steady-state phonation.

    Jitter, shimmer and HNR are only defined on quasi-periodic phonation, so a
    window is rejected when it is:
      - not almost fully voiced           -> 'low_voicing'
      - far from the speaker's own F0     -> 'f0_outlier'   (creak, subharmonics,
        diplophonia: a window at 72 Hz against a 97 Hz median is not the same
        phonatory mode and must not be averaged with the rest)
      - internally discontinuous          -> 'f0_step'      (first half vs second
        half differ by >= 1 st: an edit point inside the window, caught locally
        whether or not the global splice detector found it)
      - overlapping a detected edit point -> 'splice'
    Returns (is_valid, reason).
    """
    m = (pitch_times >= w[0]) & (pitch_times <= w[1])
    if not np.any(m):
        return False, "no_data"
    seg = f0[m]
    ok = np.isfinite(seg) & (seg > 0)
    if float(np.mean(ok)) < min_voiced_fraction:
        return False, "low_voicing"
    v = seg[ok]
    if v.size < 10:
        return False, "no_data"
    med = float(np.median(v))
    if file_median_f0 > 0 and med > 0:
        if abs(12.0 * np.log2(med / file_median_f0)) > max_f0_deviation_st:
            return False, "f0_outlier"
    h = v.size // 2
    m1, m2 = float(np.median(v[:h])), float(np.median(v[h:]))
    if m1 > 0 and m2 > 0 and abs(12.0 * np.log2(m2 / m1)) >= max_internal_step_st:
        return False, "f0_step"
    if any(w[0] < c < w[1] for c in cuts):
        return False, "splice"
    return True, "ok"


def _median_iqr(vals) -> Tuple[float, float]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    q1, q3 = np.percentile(a, [25, 75])
    return float(np.median(a)), float(q3 - q1)


def phonation_coverage(tokens: List[Tuple[float, float]],
                       measured_spans: List[Tuple[float, float]],
                       win_s: float, edge_trim_s: float, min_window_s: float,
                       gap_tol_s: float = 0.02) -> Dict[str, float]:
    """
    HOW MUCH OF THE PHONATION IS BEHIND THE REPORTED NUMBERS, AND WHERE THE
    REST WENT  (v18).

    n_windows_valid / n_windows_total counts windows, which says whether the
    median is well supported but not whether the VOWEL was well sustained: the
    same count comes out of a 7 s take and a 25 s one. This accounts for the
    phonation in seconds instead, and splits the discarded time two ways.

    Returned (all seconds unless stated):

      total_phonation_s       sum of the token durations = what the speaker held
      analyzable_total_s      the most the tiling geometry could ever measure,
                              i.e. sum over tokens of floor(u/win)*win with
                              u = duration - 2*edge_trim (or u itself when the
                              token is shorter than one window but longer than
                              min_window_s). This is a CEILING, independent of
                              the voice.
      measured_total_s        union of the windows actually measured
      analyzed_fraction       measured / total          <- the headline number
      window_yield_s          measured / analyzable     <- duration-weighted twin
                              of window_yield: unaffected by take length, so a
                              low value really does mean an unsteady voice

      by CAUSE (sums to total_phonation_s together with measured_total_s):
        discard_edge_s          the fixed onset/offset trim
        discard_quantisation_s  remainder too short to host another window
        discard_unsteady_s      windows the validity screen (or Praat, or the
                                max_windows_measured cap) threw away

      by POSITION (also sums to total_phonation_s with measured_total_s):
        discard_onset_s         phonation before the first measured window
        discard_interior_s      unmeasured gaps BETWEEN measured windows
        discard_offset_s        phonation after the last measured window
        discard_dead_token_s    tokens that produced no measured window at all
        n_interior_gaps         how many separate interior gaps there were
        longest_measured_run_s  longest uninterrupted measured stretch

    The point of the second decomposition: losing 1.5 s at the onset and 2 s at
    the offset of a 15 s take is normal and expected, and losing the same 3.5 s
    in five interior gaps is not - the first is a well-sustained vowel measured
    conservatively, the second is a voice that kept breaking down. Both give the
    same analyzed_fraction, so the fraction alone must not be read as a quality
    score without the position columns beside it.
    """
    out: Dict[str, float] = {
        "total_phonation_s": 0.0, "analyzable_total_s": 0.0,
        "measured_total_s": 0.0, "analyzed_fraction": float("nan"),
        "window_yield_s": float("nan"),
        "discard_edge_s": 0.0, "discard_quantisation_s": 0.0,
        "discard_unsteady_s": 0.0, "discard_onset_s": 0.0,
        "discard_interior_s": 0.0, "discard_offset_s": 0.0,
        "discard_dead_token_s": 0.0, "n_interior_gaps": 0,
        "longest_measured_run_s": 0.0,
    }
    toks = sorted((float(a), float(b)) for a, b in (tokens or [])
                  if float(b) > float(a))
    if not toks:
        return out
    win_s, trim, min_win = float(win_s), float(edge_trim_s), float(min_window_s)
    total = float(sum(b - a for a, b in toks))
    edge = 0.0
    analyzable = 0.0
    for a, b in toks:
        dur = b - a
        edge += min(2.0 * trim, dur)
        u = dur - 2.0 * trim
        if u < min_win:
            continue                      # no window can be placed in this token
        analyzable += u if u < win_s else float(np.floor(u / win_s + 1e-9) * win_s)
    meas = sorted((float(x), float(z)) for x, z in (measured_spans or [])
                  if float(z) > float(x))
    measured = 0.0
    onset = interior = offset = dead = 0.0
    n_gaps = 0
    longest = 0.0
    for a, b in toks:
        inside = sorted((max(a, x), min(b, z)) for x, z in meas
                        if min(b, z) - max(a, x) > 1e-9)
        if not inside:
            dead += (b - a)
            continue
        cov = float(sum(z - x for x, z in inside))
        measured += cov
        lead = max(0.0, inside[0][0] - a)
        tail = max(0.0, b - inside[-1][1])
        onset += lead
        offset += tail
        # everything not measured, not before the first and not after the last
        # window: computed as a residual so the decomposition is exact
        interior += max(0.0, (b - a) - cov - lead - tail)
        run = inside[0][1] - inside[0][0]
        for (x0, z0), (x1, z1) in zip(inside[:-1], inside[1:]):
            gap = x1 - z0
            if gap > gap_tol_s:
                n_gaps += 1
                longest = max(longest, run)
                run = z1 - x1
            else:
                run += max(0.0, gap) + (z1 - x1)
        longest = max(longest, run)
    geometry = max(0.0, total - analyzable)
    out.update(
        total_phonation_s=total,
        analyzable_total_s=analyzable,
        measured_total_s=measured,
        analyzed_fraction=(measured / total) if total > 0 else float("nan"),
        window_yield_s=(measured / analyzable) if analyzable > 0 else float("nan"),
        discard_edge_s=min(edge, geometry),
        discard_quantisation_s=max(0.0, geometry - edge),
        discard_unsteady_s=max(0.0, analyzable - measured),
        discard_onset_s=onset, discard_interior_s=interior,
        discard_offset_s=offset, discard_dead_token_s=dead,
        n_interior_gaps=int(n_gaps), longest_measured_run_s=float(longest))
    return out


def steady_frame_profile(times: np.ndarray, f0: np.ndarray,
                         tokens: List[Tuple[float, float]],
                         ref_f0: float = float("nan"),
                         scale_s: float = 2.0,
                         max_dev_st: float = 3.0,
                         max_step_st: float = 1.0,
                         max_wobble_st: float = 2.0,
                         min_voiced_fraction: float = 0.90,
                         min_scale_s: float = 1.0) -> Dict[str, float]:
    """
    STEADINESS WITHOUT THE WINDOW GRID  (v18).

    For every frame, ask the question window_validity() asks of a window:
    could a valid analysis window be CENTRED here? A frame counts as steady when
    the +-scale_s/2 neighbourhood around it is almost fully voiced, sits within
    max_dev_st of the speaker's own F0, holds together internally (first half vs
    second half of the neighbourhood differ by < max_step_st) and the frame
    itself is not a spike away from the local median (max_wobble_st).

    Why this exists next to window_yield: the tiled measures are quantised in
    units of win_s, so a 300 ms wobble costs a full 2 s slot and a 5 s take can
    never reach a high yield however clean it is. This measure has no grid, so
    it separates "short but perfectly steady" from "long but patchy", which is
    exactly the distinction the tiling cannot make.

    Deliberately NOT flagged as unsteady, because they are measured elsewhere
    and must not be counted twice: frame-to-frame jitter (roughness), vibrato /
    tremor at 2-12 Hz, and slow drift below max_step_st per scale_s/2 - all of
    these pass, exactly as they pass the window screen.

    Returns steady_frame_fraction, longest_steady_run_s and n_steady_stretches
    (fragmentation: 80% steady in one block is not the same finding as 80%
    steady scattered over ten islands).
    """
    out = {"steady_frame_fraction": float("nan"),
           "longest_steady_run_s": float("nan"),
           "n_steady_stretches": 0}
    t = np.asarray(times, dtype=float)
    f = np.asarray(f0, dtype=float)
    if t.size < 10 or t.size != f.size or not tokens:
        return out
    voiced = np.isfinite(f) & (f > 0)
    ref = (float(ref_f0) if (ref_f0 is not None and np.isfinite(ref_f0)
                             and float(ref_f0) > 0)
           else (float(np.median(f[voiced])) if np.any(voiced) else float("nan")))
    if not np.isfinite(ref) or ref <= 0:
        return out
    dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    if not np.isfinite(dt) or dt <= 0:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        st = 12.0 * np.log2(np.where(voiced, f, np.nan) / ref)
    half = max(2, int(round((float(scale_s) / 2.0) / dt)))
    need_s = max(float(min_scale_s), 0.5 * float(scale_s))
    n = t.size
    steady = np.zeros(n, dtype=bool)
    in_tok = np.zeros(n, dtype=bool)
    for (a, b) in tokens:
        idx = np.where((t >= float(a)) & (t <= float(b)))[0]
        if idx.size == 0:
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        in_tok[i0:i1 + 1] = True
        for i in range(i0, i1 + 1):
            if not voiced[i]:
                continue
            lo, hi = max(i0, i - half), min(i1, i + half)
            if (hi - lo + 1) * dt < need_s:
                continue                      # too close to the token edge to judge
            seg = st[lo:hi + 1]
            ok = np.isfinite(seg)
            if float(np.mean(ok)) < min_voiced_fraction:
                continue
            fin = seg[ok]
            if fin.size < 10:
                continue
            med = float(np.median(fin))
            if abs(med) > max_dev_st:
                continue                      # creak / subharmonic / other register
            if abs(float(st[i]) - med) > max_wobble_st:
                continue                      # local excursion, not a steady frame
            h = fin.size // 2
            m1, m2 = float(np.median(fin[:h])), float(np.median(fin[h:]))
            if abs(m2 - m1) >= max_step_st:
                continue                      # drifting or stepping through here
            steady[i] = True
    if not np.any(in_tok):
        return out
    good = steady & in_tok
    best = cur = runs = 0
    for v in good:
        if v:
            cur += 1
            if cur == 1:
                runs += 1
        else:
            best = max(best, cur)
            cur = 0
    best = max(best, cur)
    out.update(steady_frame_fraction=float(np.mean(steady[in_tok])),
               longest_steady_run_s=float(best * dt),
               n_steady_stretches=int(runs))
    return out


def describe_coverage(cov: Dict[str, float], steady_frame_fraction: float,
                     longest_run_s: float, n_stretches: int) -> str:
    """One line saying whether the take was held, and what was lost where."""
    total = cov.get("total_phonation_s", float("nan"))
    meas = cov.get("measured_total_s", float("nan"))
    frac = cov.get("analyzed_fraction", float("nan"))
    interior = cov.get("discard_interior_s", float("nan"))
    gaps = int(cov.get("n_interior_gaps", 0) or 0)
    sff = float(steady_frame_fraction) if steady_frame_fraction is not None else float("nan")
    inter_share = (interior / total) if (np.isfinite(interior) and total and total > 0) else float("nan")
    if not np.isfinite(sff):
        grade = "steadiness not assessable"
    elif sff >= 0.85 and (not np.isfinite(inter_share) or inter_share <= 0.05):
        grade = "STEADY THROUGHOUT"
    elif sff >= 0.70:
        grade = "MOSTLY STEADY"
    elif sff >= 0.40:
        grade = "PATCHY"
    else:
        grade = "LARGELY UNSTEADY"
    if gaps >= 2 and np.isfinite(inter_share) and inter_share > 0.05:
        grade += f" (broken into pieces: {gaps} interior gap(s))"
    bits = [grade]
    if np.isfinite(frac):
        bits.append(f"analysed {frac * 100:.0f}% of {total:.1f} s phonated "
                    f"({meas:.1f} s)")
    if np.isfinite(sff):
        bits.append(f"{sff * 100:.0f}% of frames steady, longest run "
                    f"{longest_run_s:.1f} s in {max(1, int(n_stretches or 0))} stretch(es)")
    onset = cov.get("discard_onset_s", float("nan"))
    offset = cov.get("discard_offset_s", float("nan"))
    if np.isfinite(onset) and np.isfinite(offset) and np.isfinite(interior):
        bits.append(f"lost: onset {onset:.1f} s, interior {interior:.1f} s, "
                    f"offset {offset:.1f} s")
    dead = cov.get("discard_dead_token_s", 0.0)
    if dead and dead > 0.05:
        bits.append(f"{dead:.1f} s in token(s) that yielded no window")
    return "; ".join(bits)


def choose_clean_window(t0: float, t1: float, cuts: List[float],
                        edge_trim_s: float = 0.25, max_window_s: float = 3.0,
                        min_window_s: float = 0.5, hop_s: float = 0.5
                        ) -> Optional[Tuple[float, float]]:
    """
    Pick the analysis window inside a token, avoiding any detected edit point.

    Candidate windows are slid across the trimmed token; those containing a cut
    are rejected; of the rest the one CLOSEST TO THE TOKEN CENTRE is chosen.
    Selection is deliberately NOT based on steadiness - picking the calmest
    window would bias jitter, F0 SD and tremor downwards.
    """
    base = stable_window(t0, t1, edge_trim_s=edge_trim_s,
                         max_window_s=max_window_s, min_window_s=min_window_s)
    if base is None:
        return None
    if not cuts:
        return base
    a0, b0 = t0 + edge_trim_s, t1 - edge_trim_s
    length = base[1] - base[0]
    if b0 - a0 < length:
        return base
    centre = 0.5 * (t0 + t1)
    best, best_d = None, None
    x = a0
    while x + length <= b0 + 1e-9:
        if not any(x < c < x + length for c in cuts):
            d = abs((x + length / 2.0) - centre)
            if best_d is None or d < best_d:
                best, best_d = (float(x), float(x + length)), d
        x += hop_s
    return best if best is not None else base


def stable_window(t0: float, t1: float, edge_trim_s: float = 0.25,
                  max_window_s: float = 2.0, min_window_s: float = 0.5
                  ) -> Optional[Tuple[float, float]]:
    """
    Steady-state analysis window inside a token: drop onset/offset transients,
    then take up to max_window_s centred in what remains. Returns None when the
    token is too short to yield a usable window.
    """
    dur = t1 - t0
    if dur <= 2 * edge_trim_s + min_window_s:
        edge_trim_s = max(0.05, (dur - min_window_s) / 2.0)
    a, b = t0 + edge_trim_s, t1 - edge_trim_s
    if b - a < min_window_s:
        return None
    if b - a > max_window_s:
        mid = 0.5 * (a + b)
        a, b = mid - max_window_s / 2.0, mid + max_window_s / 2.0
    return float(a), float(b)


def f0_tremor(times: np.ndarray, f0: np.ndarray,
              lo_hz: float = 2.0, hi_hz: float = 12.0,
              min_peak_ratio: float = 6.0
              ) -> Tuple[float, float, float]:
    """
    Frequency-tremor rate and extent from the F0 contour of a sustained vowel.

    FIXED in v9. Two bugs:
      1. EXTENT was 2*sqrt(2)*SD of the WHOLE detrended contour, i.e. every kind
         of F0 instability - creak, octave jumps, residual drift - was reported
         as "tremor extent". That is how an unstable file came out at 5.89 st, a
         6-semitone tremor, which does not exist. The extent is now measured on
         the BAND-LIMITED 2-12 Hz component only.
      2. RATE was argmax over the band, which always returns something, so every
         file got a plausible-looking 3-4 Hz "tremor". A peak is now required to
         stand out from the rest of the band by min_peak_ratio; otherwise both
         rate and extent are NaN (= no measurable tremor).

    The contour is interpolated onto a uniform grid (voiced-only frames are not
    equally spaced), converted to semitones and linearly detrended.
    Returns (rate_hz, extent_semitones, peak_ratio). Extent is 2*sqrt(2)*SD of
    the band-passed signal, i.e. the peak-to-peak of an equivalent sinusoid.

    Calibration on synthetic contours (3 s, 10 ms frames): a real 5 Hz / 1.0 st
    tremor gives peak_ratio ~1e4 and is recovered as 5.00 Hz / 1.00 st; white F0
    noise of 0.3 st gives ~3.0, a pure drift ~1.9 and a mid-window octave jump
    ~3.1. The default threshold of 6.0 is therefore deliberately conservative:
    it will miss a mild tremor buried in equal-amplitude noise (ratio ~4.4)
    rather than report tremor where there is only instability. Lower it to ~4.0
    if sensitivity matters more than specificity for your question.
    """
    nan3 = (float("nan"), float("nan"), float("nan"))
    f0 = np.asarray(f0, dtype=float)
    times = np.asarray(times, dtype=float)
    ok = np.isfinite(f0) & (f0 > 0) & np.isfinite(times)
    if int(ok.sum()) < 24:
        return nan3
    t, v = times[ok], f0[ok]
    st = hz_to_semitones(v, float(np.median(v)))
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return nan3
    # uniform grid (gaps are bridged linearly; windows are >=90% voiced)
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    if grid.size < 24:
        return nan3
    stg = np.interp(grid, t, st)
    stg = stg - np.polyval(np.polyfit(grid, stg, 1), grid)   # remove drift
    fs = 1.0 / dt
    hi = min(hi_hz, 0.45 * fs)
    if hi <= lo_hz + 0.5:
        return nan3
    # ---- rate: dominant band peak, required to be prominent ----------------
    win = np.hanning(stg.size)
    spec = np.abs(np.fft.rfft(stg * win))
    freqs = np.fft.rfftfreq(stg.size, d=dt)
    band = (freqs >= lo_hz) & (freqs <= hi)
    if not np.any(band) or np.all(spec[band] <= 0):
        return nan3
    k = int(np.argmax(spec[band]))
    rate = float(freqs[band][k])
    band_vals = spec[band]
    med = float(np.median(band_vals))
    peak_ratio = float(band_vals[k] / med) if med > 0 else float("inf")
    if peak_ratio < min_peak_ratio:
        return float("nan"), float("nan"), peak_ratio      # no real tremor
    # ---- extent: SD of the band-passed component only ----------------------
    try:
        sos = signal.butter(2, [lo_hz / (0.5 * fs), hi / (0.5 * fs)],
                            btype="bandpass", output="sos")
        bp = signal.sosfiltfilt(sos, stg)
    except Exception:
        keep = np.zeros_like(spec)
        keep[band] = spec[band]
        bp = np.fft.irfft(keep * np.exp(1j * np.angle(np.fft.rfft(stg * win))),
                          n=stg.size)
    extent = float(2.0 * np.sqrt(2.0) * np.std(bp))
    return rate, extent, peak_ratio


def compute_rate_from_passage(
    file_path: str, y: np.ndarray, sr: int,
    speech_start: float, speech_end: float, articulation_time_sec: float,
) -> Tuple[int, str, str, float, float, float, int, float]:
    """
    Use the KNOWN passage syllable count when identifiable, else an un-clamped
    estimate. Returns (count, source, passage, speech_rate, artic_rate,
    speaking_time_fraction, estimated_count, agreement_ratio).

    speaking_time_fraction = artic_time/total_speech, in [0,1]; 1.0 = no pauses,
    lower = more time lost to pausing. It is independent of the syllable
    constant, so it stays valid even if the constant is wrong.

    v16 - AUTOMATIC CHECK ON THE CONSTANT. KNOWN_PASSAGE_SYLLABLES is hand-entered
    and cannot be validated from the audio, but a wrong constant silently rescales
    speech_rate and articulation_rate for EVERY file of that passage - and because
    it is wrong by the same factor every time, the error is invisible in
    within-subject comparisons and shows up only when you compare against
    published norms. The envelope estimate is now computed even when the passage
    is known, purely as a cross-check: it is far too noisy to replace the
    constant, but it is quite good enough to catch a constant that is off by 30%
    or an octave-style factor-of-two typo. agreement_ratio = estimate / constant;
    values outside roughly 0.7-1.4 mean one of the two is wrong. Reported as
    syllable_count_estimated / syllable_count_agreement in the AUDIT block.
    """
    speech_dur = max(1e-6, speech_end - speech_start)
    artic_time = max(1e-6, articulation_time_sec)
    passage_name, known = identify_passage(file_path)
    est = estimate_syllables_unclamped(y, sr, speech_start, speech_end)
    if known is not None:
        count, source = known, "known_passage"
    else:
        count = est
        source = "estimated"
        passage_name = passage_name or ""
    agreement = (float(est) / float(known)) if (known and known > 0) else float("nan")
    speech_rate = count / speech_dur
    artic_rate = count / artic_time
    speaking_time_fraction = artic_time / speech_dur if speech_dur > 0 else float("nan")
    return (count, source, (passage_name or ""), speech_rate, artic_rate,
            speaking_time_fraction, int(est), agreement)


def envelope_modulation_spectrum(
    y: np.ndarray, sr: int, speech_start: float, speech_end: float
) -> Tuple[float, float, float]:
    """
    Envelope modulation spectrum: fraction of envelope-modulation energy in the
    3-8 Hz syllabic band, the peak modulation frequency, and the syllabic/slow
    (3-8 vs 0.5-3 Hz) ratio. Reduced/flattened in hypokinetic dysarthria.
    """
    try:
        s0 = int(speech_start * sr); s1 = int(speech_end * sr)
        seg = y[s0:s1]
        if len(seg) < sr * 0.5:
            return float("nan"), float("nan"), float("nan")
        from scipy.signal import hilbert, butter, filtfilt
        env = np.abs(hilbert(seg))
        nyq = sr / 2.0
        b, a = butter(4, min(25.0, 0.99 * nyq) / nyq, btype="low")
        env_lp = filtfilt(b, a, env)
        step = max(1, int(round(sr / 100.0)))
        env_ds = env_lp[::step]; env_sr = sr / step
        env_ds = env_ds - np.mean(env_ds)
        if env_ds.size < 16 or not np.any(env_ds):
            return float("nan"), float("nan"), float("nan")
        env_w = env_ds * np.hanning(len(env_ds))
        spec = np.abs(np.fft.rfft(env_w)) ** 2
        freqs = np.fft.rfftfreq(len(env_w), d=1.0 / env_sr)

        def bp(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(np.sum(spec[m])) if np.any(m) else 0.0

        total = bp(0.5, 20.0)
        if total <= 0:
            return float("nan"), float("nan"), float("nan")
        p_syll = bp(3.0, 8.0); p_slow = bp(0.5, 3.0)
        ratio_3_8 = p_syll / total
        m = (freqs >= 1.0) & (freqs <= 12.0)
        peak_f = float(freqs[m][int(np.argmax(spec[m]))]) if np.any(m) else float("nan")
        ratio_syll_slow = (p_syll / p_slow) if p_slow > 0 else float("nan")
        return float(ratio_3_8), peak_f, float(ratio_syll_slow)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def formant_cloud_dispersion(f1_values, f2_values) -> Tuple[float, float]:
    """
    Segmentation-free vowel-space dispersion: log-area of the F1-F2 covariance
    ellipse and the generalized SD (Hz). Shrinks with hypokinetic centralization.
    """
    try:
        f1 = np.asarray([v for v in f1_values if v and v > 0], dtype=float)
        f2 = np.asarray([v for v in f2_values if v and v > 0], dtype=float)
        n = min(f1.size, f2.size)
        if n < 10:
            return float("nan"), float("nan")
        pts = np.vstack([f1[:n], f2[:n]])
        cov = np.cov(pts); det = np.linalg.det(cov)
        if det <= 0:
            return float("nan"), float("nan")
        return float(np.log(np.pi * np.sqrt(det))), float(det ** 0.25)
    except Exception:
        return float("nan"), float("nan")


def intensity_decay_over(sound, intervals: List[Tuple[float, float]],
                         adaptive_floor: float) -> Tuple[float, float, float]:
    """
    Loudness trend over the given interval(s). Negative = fading.

    Returns (db_across_span, db_per_second, span_s).

    Two numbers because the original single one was ambiguous. The slope was
    fitted against time NORMALISED to 0..1, so the value is "dB from the start
    of the analysed span to the end of it" - which does not compare between a
    10 s take and a 70 s one, since the same physiological fade rate produces a
    seven times larger number on the longer take. db_across_span keeps the old
    definition (so a fixed-length reading passage stays comparable with earlier
    runs); db_per_second is the duration-free twin and is the one to use when
    take length varies, as it does for sustained phonation.

    intervals: measure only inside these spans. On a sustained file the caller
    passes ONE token, because a fit across several tokens and the breaths
    between them describes the recording session, not the vowel.
    """
    nan3 = (float("nan"), float("nan"), float("nan"))
    try:
        iv = [(float(a), float(b)) for a, b in (intervals or []) if b > a]
        if not iv:
            return nan3
        intensity = call(sound, "To Intensity", adaptive_floor, 0.0, "yes")
        n = call(intensity, "Get number of frames")
        t0 = call(intensity, "Get time from frame number", 1)
        dt = call(intensity, "Get time step") or 0.01
        ts, vs = [], []
        for i in range(1, n + 1):
            t = t0 + (i - 1) * dt
            if any(a <= t <= b for a, b in iv):
                v = call(intensity, "Get value in frame", i)
                if v is not None and not np.isnan(v) and v > 0:
                    ts.append(t); vs.append(v)
        ts = np.asarray(ts, dtype=float); vs = np.asarray(vs, dtype=float)
        if ts.size < 10:
            return nan3
        thr = np.percentile(vs, 30); keep = vs >= thr
        if np.count_nonzero(keep) < 10:
            keep = np.ones_like(vs, dtype=bool)
        tk = ts[keep]; vk = vs[keep]
        span = float(tk.max() - tk.min())
        if span <= 1e-6:
            return nan3
        per_s = float(np.polyfit(tk, vk, 1)[0])
        return float(per_s * span), per_s, span
    except Exception:
        return nan3


def intensity_decay(sound, speech_start: float, speech_end: float,
                    adaptive_floor: float) -> float:
    """
    Backward-compatible wrapper: dB change across [speech_start, speech_end].
    New code should call intensity_decay_over(), which also returns dB/s.
    """
    return intensity_decay_over(sound, [(speech_start, speech_end)],
                                adaptive_floor)[0]


def hnr_median_perinterval(sound, intervals, adaptive_floor: float) -> float:
    """Median HNR over voiced intervals (robust complement to Voice Report HNR)."""
    if not intervals:
        return float("nan")
    try:
        harm = call(sound, "To Harmonicity (cc)", 0.01, adaptive_floor, 0.1, 4.5)
        n = call(harm, "Get number of frames")
        dt = call(harm, "Get time step") or 0.01
        t0 = call(harm, "Get time from frame number", 1)
        iv = np.asarray(intervals, dtype=float)
        vals = []
        for i in range(1, n + 1):
            t = t0 + (i - 1) * dt
            if np.any((t >= iv[:, 0]) & (t <= iv[:, 1])):
                hv = call(harm, "Get value in frame", i)
                # v20: was `hv > 0`, which discarded every NEGATIVE
                # harmonicity - i.e. exactly the breathy/aperiodic frames that
                # carry the clinical signal - and biased the median upwards.
                if hv is not None and np.isfinite(hv):
                    vals.append(float(hv))
        return float(np.median(vals)) if vals else float("nan")
    except Exception:
        return float("nan")


def _cpps_whole_span(sound, y: np.ndarray, sr: int, speech_start: float,
                     speech_end: float, adaptive_floor: float,
                     adaptive_ceiling: float) -> Tuple[float, str]:
    """
    Original whole-span CPPS (Praat PowerCepstrogram over speech_start..speech_end,
    FFT fallback if it fails). Kept only as a last-resort path for when no voiced
    intervals are available -- this is the behavior that dilutes CPPS with any
    silence/pauses inside the span, which is exactly what cpps_hardened() below
    is designed to avoid whenever intervals are supplied.
    """
    try:
        snd = sound.extract_part(speech_start, speech_end)
        cepg = call(snd, "To PowerCepstrogram", adaptive_floor, 0.002, 5000, 50)
        cpps = call(cepg, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
                    "Parabolic", 0.001, 0, "Exponential decay", "Robust")
        if cpps is not None and not np.isnan(cpps):
            return float(cpps), "praat_whole_span"
    except Exception:
        pass
    try:
        s0 = int(speech_start * sr); s1 = int(speech_end * sr)
        seg = y[s0:s1]
        if seg.size < sr * 0.1:
            return float("nan"), "fft_fallback_whole_span"
        cep = np.abs(np.fft.ifft(np.log(np.abs(np.fft.fft(seg)) + 1e-10)))
        min_q = int(sr / adaptive_ceiling); max_q = int(sr / adaptive_floor)
        if max_q >= len(cep):
            return float("nan"), "fft_fallback_whole_span"
        region = cep[min_q:max_q]
        peak = float(np.max(region)); base = float(np.mean(region))
        return (20.0 * np.log10(peak / base) if base > 0 else float("nan")), "fft_fallback_whole_span"
    except Exception:
        return float("nan"), "fft_fallback_whole_span"


def cpps_hardened(sound, y: np.ndarray, sr: int, speech_start: float,
                  speech_end: float, adaptive_floor: float,
                  adaptive_ceiling: float,
                  intervals: Optional[List[Tuple[float, float]]] = None,
                  min_interval_dur: float = 0.15) -> Tuple[float, str]:
    """
    Smoothed CPP, computed PER VOICED INTERVAL -- the same `intervals` list
    passed to jitter/shimmer/HNR in analyze_voice_quality -- then combined with
    the identical duration-weighted, top-decile-trimmed aggregation used there
    (mirrors the agg() helper). This keeps silence OUT of the cepstral estimate:
    pauses between repeated tokens (e.g. 3x sustained "aaaa" with breaths in
    between) or between phrases in connected speech no longer get folded into
    one PowerCepstrogram call the way the old whole-span version did.

    Falls back, in order: (1) FFT cepstrum per interval if Praat fails on every
    interval, (2) the original whole-span behavior if no usable intervals exist
    at all (e.g. `intervals` not supplied, for backward compatibility).

    min_interval_dur: intervals shorter than this are skipped for CPPS specifically
    (too few cepstrogram time-frames to give a stable estimate), even though the
    same interval may still be long enough for jitter/shimmer (which only need
    dur >= 0.10s). Tune via CompleteSpeechAnalyzer(cpps_min_interval_dur=...).
    """
    snd_start = call(sound, "Get start time")
    snd_end = call(sound, "Get end time")

    if not intervals:
        return _cpps_whole_span(sound, y, sr, speech_start, speech_end,
                                adaptive_floor, adaptive_ceiling)

    cpps_vals, weights = [], []
    for s, e in intervals:
        s = max(s, snd_start, speech_start); e = min(e, snd_end, speech_end)
        dur = e - s
        if dur < min_interval_dur:
            continue
        try:
            snd = sound.extract_part(s, e)
            cepg = call(snd, "To PowerCepstrogram", adaptive_floor, 0.002, 5000, 50)
            cpps = call(cepg, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
                        "Parabolic", 0.001, 0, "Exponential decay", "Robust")
            if cpps is not None and not np.isnan(cpps):
                cpps_vals.append(float(cpps)); weights.append(dur)
        except Exception:
            continue

    if cpps_vals:
        a = np.asarray(cpps_vals); w = np.asarray(weights)
        if a.size >= 5:
            cutoff = np.percentile(a, 90); keep = a <= cutoff
            if np.any(keep):
                a, w = a[keep], w[keep]
        cpps_final = float(np.average(a, weights=w)) if w.sum() > 0 else float(np.mean(a))
        return cpps_final, "praat_per_interval"

    # Praat failed (or every interval was too short) -> FFT fallback, still
    # computed per interval and duration-weighted rather than over the whole span.
    fft_vals, fft_weights = [], []
    for s, e in intervals:
        s = max(s, speech_start); e = min(e, speech_end)
        dur = e - s
        if dur < min_interval_dur:
            continue
        s0 = int(s * sr); s1 = int(e * sr)
        seg = y[s0:s1]
        if seg.size < sr * 0.1:
            continue
        cep = np.abs(np.fft.ifft(np.log(np.abs(np.fft.fft(seg)) + 1e-10)))
        min_q = int(sr / adaptive_ceiling); max_q = int(sr / adaptive_floor)
        if max_q >= len(cep):
            continue
        region = cep[min_q:max_q]
        peak = float(np.max(region)); base = float(np.mean(region))
        if base > 0:
            fft_vals.append(20.0 * np.log10(peak / base)); fft_weights.append(dur)

    if fft_vals:
        a = np.asarray(fft_vals); w = np.asarray(fft_weights)
        cpps_final = float(np.average(a, weights=w)) if w.sum() > 0 else float(np.mean(a))
        return cpps_final, "fft_fallback_per_interval"

    # Last resort: nothing usable per-interval -> old whole-span behavior.
    return _cpps_whole_span(sound, y, sr, speech_start, speech_end,
                            adaptive_floor, adaptive_ceiling)


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class CompleteSpeechAnalyzer:
    """Comprehensive speech analyzer matching and extending Praat's capabilities."""

    def __init__(
        self,
        pitch_floor: float = 50.0,
        pitch_ceiling: float = 400.0,
        voice_report_pitch_floor: float = 75.0,
        voice_report_pitch_ceiling: float = 500.0,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45,
        silence_threshold_db: Optional[float] = None,
        min_pause_duration: float = 0.2,
        noise_margin_db: float = 6.0,
        min_filler_duration: float = 0.25,
        max_filler_duration: float = 1.5,
        filler_freq_range: Tuple[float, float] = (80, 400),
        max_formant_hz: float = 5500.0,
        num_formants: int = 5,
        max_period_factor: float = 1.3,
        max_amplitude_factor: float = 1.6,
        telephone_mode: Any = "auto",
        time_step: float = 0.0,
        cpps_min_interval_dur: float = 0.15,
        task: str = "auto",
        min_vowel_token_dur: float = 0.8,
        vowel_edge_trim_s: float = 0.25,
        vowel_window_max_s: float = 2.0,
        token_bridge_gap_s: float = 0.35,
        token_min_voiced_fraction: float = 0.5,
        split_at_splices: bool = True,
        splice_f0_step_st: float = 1.0,
        vowel_window_hop_s: float = 3.0,       # DEAD since v8, see below
        vowel_window_grid_hop_s: Optional[float] = None,
        vowel_window_min_s: float = 1.0,
        window_min_voiced_fraction: float = 0.90,
        window_max_f0_deviation_st: float = 3.0,
        window_max_internal_step_st: float = 1.0,
        max_windows_measured: int = 40,
        # ---- v9 ----
        pitch_range: Optional[Tuple[float, float]] = None,
        octave_check: bool = True,
        octave_autocorrect: bool = True,
        octave_strong_db: float = -10.0,
        # v20: 2 windows / 4 s, not 3 / 6 s. With 2.0 s windows the old gate
        # demanded ~6.5 s of flawless phonation before anything about the
        # voice was reported, which discarded usable short takes from
        # patients who cannot sustain longer.
        min_valid_windows: int = 2,
        min_measured_s: float = 4.0,
        boundaries_dir: Optional[str] = None,
        boundaries_suffix: str = "_tokens.csv",
        # ---- v10 ----
        # v20: TRUE by default. The recording protocol is one reading passage
        # OR one sustained phonation per file, so there is nothing to segment
        # into trials; an internal dropout is a FINDING about this one
        # production, not evidence of a second one.
        single_token_per_file: bool = True,
        formant_ceiling_hz: Optional[float] = None,
        # ---- v13 ----
        legacy_compat: bool = False,
        # ---- v20: the composite fluency/severity index is OFF by default.
        # It is an unvalidated weighted sum whose output looks like a clinical
        # severity rating. Enable it only if you have validated it locally.
        enable_fluency_index: bool = False,
        # ---- v19: definition of a BREAK in phonation ----
        # "The longest phonation with no break in the middle" needs a break to
        # be defined, and the definition drives the number. Defaults: a break is
        # >=50 ms during which the signal is either 15 dB below the local
        # phonation level or unvoiced. 50 ms is short enough to catch a glottal
        # stop and long enough to ignore one creaky period or a single
        # mistracked frame; 15 dB matches the sustained-vowel silence line in
        # get_silence_threshold() so the two do not disagree about the same gap.
        phonation_break_min_s: float = 0.05,
        phonation_break_drop_db: float = 15.0,
        phonation_require_voicing: bool = True,
        phonation_env_win_s: float = 0.020,
        phonation_env_hop_s: float = 0.005,
    ):
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.voice_report_pitch_floor = voice_report_pitch_floor
        self.voice_report_pitch_ceiling = voice_report_pitch_ceiling
        self.silence_threshold = silence_threshold
        self.voicing_threshold = voicing_threshold
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_duration = min_pause_duration
        self.noise_margin_db = noise_margin_db
        self.min_filler_duration = min_filler_duration
        self.max_filler_duration = max_filler_duration
        self.filler_freq_range = filler_freq_range
        self.max_formant_hz = max_formant_hz
        self.num_formants = num_formants
        self.max_period_factor = max_period_factor
        self.max_amplitude_factor = max_amplitude_factor
        self.telephone_mode = telephone_mode
        self.cpps_min_interval_dur = cpps_min_interval_dur
        # task: "auto" | "reading" | "sustained_vowel"
        self.task = task
        self.min_vowel_token_dur = min_vowel_token_dur
        self.vowel_edge_trim_s = vowel_edge_trim_s
        self.vowel_window_max_s = vowel_window_max_s
        self.token_bridge_gap_s = token_bridge_gap_s
        self.token_min_voiced_fraction = token_min_voiced_fraction
        self.split_at_splices = split_at_splices
        self.splice_f0_step_st = splice_f0_step_st
        # vowel_window_hop_s has been ignored since v8: the candidate grid is
        # laid at hop = win/2 inside _sustained_from_tokens and the survivors are
        # then reduced to a non-overlapping set. It is kept so old call sites do
        # not break. vowel_window_grid_hop_s is the knob that actually works:
        # None keeps the win/2 default, while a finer step (e.g. 0.5 s) lets a
        # window slide around a short wobble instead of losing the whole slot -
        # the cheapest way to raise the yield on takes that keep failing the
        # usability gate. It changes only WHERE windows may start, never how
        # they are screened.
        self.vowel_window_hop_s = vowel_window_hop_s
        self.vowel_window_grid_hop_s = (float(vowel_window_grid_hop_s)
                                        if vowel_window_grid_hop_s else None)
        self.vowel_window_min_s = vowel_window_min_s
        self.window_min_voiced_fraction = window_min_voiced_fraction
        self.window_max_f0_deviation_st = window_max_f0_deviation_st
        self.window_max_internal_step_st = window_max_internal_step_st
        self.max_windows_measured = max_windows_measured
        # ---- v9 ----
        self.pitch_range = tuple(pitch_range) if pitch_range else None
        self.octave_check = octave_check
        self.octave_autocorrect = octave_autocorrect
        self.octave_strong_db = float(octave_strong_db)
        self.min_valid_windows = int(min_valid_windows)
        self.min_measured_s = float(min_measured_s)
        self.boundaries_dir = boundaries_dir
        self.boundaries_suffix = boundaries_suffix
        # v10: one file = one production. With the recordings split per
        # repetition there is nothing to segment and nothing to un-splice, and
        # any "edit point" found inside a single production is a false positive
        # that shortens MPT.
        self.single_token_per_file = bool(single_token_per_file)
        self.formant_ceiling_hz = (float(formant_ceiling_hz)
                                   if formant_ceiling_hz else None)
        # v13: restores the two PRE-v9 behaviours that also affect the reading
        # branch - the one-sided top-decile trim with a 0.0 fallback in agg(), and
        # the relative-only formant check. Use it ONLY to reproduce numbers from
        # an earlier run for comparison. Both behaviours are wrong: the trim
        # biases every perturbation value downwards by an amount that depends on
        # how many voiced intervals the file happens to contain, and the 0.0
        # fallback writes a perfect zero into the CSV when Praat measured nothing.
        self.legacy_compat = bool(legacy_compat)
        self.enable_fluency_index = bool(enable_fluency_index)
        # ---- v19 ----
        self.phonation_break_min_s = float(phonation_break_min_s)
        self.phonation_break_drop_db = float(phonation_break_drop_db)
        self.phonation_require_voicing = bool(phonation_require_voicing)
        self.phonation_env_win_s = float(phonation_env_win_s)
        self.phonation_env_hop_s = float(phonation_env_hop_s)
        self._octave: Dict[str, Any] = {}
        self._f0_wide_median = float("nan")
        self._pp_cache_key = None
        self._pp_cache = None
        self._snr_reference = "none"
        self._digital_silence = False
        self._silence_threshold_used = float("nan")
        self._task = TASK_READING
        self._task_source = ""
        self._is_telephone = False
        self._hnr_max_hz = 4500.0
        self._voicing_thr = 0.45
        self.time_step = time_step
        self._voice_report_extras = {}
        self._adaptive_floor = pitch_floor
        self._adaptive_ceiling = pitch_ceiling
        self._adaptive_median = (pitch_floor + pitch_ceiling) / 2.0
        self._settings = {
            'pitch_floor': pitch_floor, 'pitch_ceiling': pitch_ceiling,
            'voice_report_pitch_floor': voice_report_pitch_floor,
            'voice_report_pitch_ceiling': voice_report_pitch_ceiling,
            'silence_threshold': silence_threshold,
            'voicing_threshold': voicing_threshold,
            'min_pause_duration': min_pause_duration,
            'noise_margin_db': noise_margin_db,
            'max_formant_hz': max_formant_hz, 'num_formants': num_formants,
            'max_period_factor': max_period_factor,
            'max_amplitude_factor': max_amplitude_factor,
            'cpps_min_interval_dur': cpps_min_interval_dur,
            'task_requested': task,
            'one_production_per_file': bool(single_token_per_file),
            'enable_fluency_index': bool(enable_fluency_index),
            'min_vowel_token_dur': min_vowel_token_dur,
            'vowel_edge_trim_s': vowel_edge_trim_s,
            'vowel_window_max_s': vowel_window_max_s,
            'token_bridge_gap_s': token_bridge_gap_s,
            'token_min_voiced_fraction': token_min_voiced_fraction,
            'split_at_splices': split_at_splices,
            'splice_f0_step_st': splice_f0_step_st,
            'vowel_window_hop_s': vowel_window_hop_s,
            'vowel_window_grid_hop_s': (vowel_window_grid_hop_s
                                        if vowel_window_grid_hop_s else 'win/2'),
            'window_min_voiced_fraction': window_min_voiced_fraction,
            'window_max_f0_deviation_st': window_max_f0_deviation_st,
            'window_max_internal_step_st': window_max_internal_step_st,
            'pitch_range_fixed': str(self.pitch_range) if self.pitch_range else "auto",
            'octave_check': octave_check,
            'octave_autocorrect': octave_autocorrect,
            'min_valid_windows': min_valid_windows,
            'min_measured_s': min_measured_s,
            'single_token_per_file': single_token_per_file,
            'formant_ceiling_hz_fixed': formant_ceiling_hz or 'auto',
            'octave_strong_db': octave_strong_db,
            'phonation_break_min_s': phonation_break_min_s,
            'phonation_break_drop_db': phonation_break_drop_db,
            'phonation_require_voicing': phonation_require_voicing,
        }

    # ----------------------------- AUDIO LOADING -------------------------------
    def load_audio(self, file_path: str):
        sound = parselmouth.Sound(file_path)
        y, sr = librosa.load(file_path, sr=None, mono=True)
        return sound, y, sr

    # ----------------------- RECORDING QUALITY ASSESSMENT ----------------------
    def assess_recording_quality(self, y, sr, speech_start, speech_end,
                                 silence_threshold_db, rms=None, rms_times=None,
                                 task=None) -> RecordingQualityMetrics:
        """
        Quality of the RECORDING, not of the voice.

        FIXED in v8: the noise floor is measured from a genuine SILENT REFERENCE
        (frames outside the detected speech span, or below-threshold frames
        inside it), never from a percentile of the whole file. On a wall-to-wall
        sustained vowel the old percentile floor was the vowel itself, so SNR
        collapsed to 5-10 dB and every studio recording was flagged "noisy" and
        its jitter/shimmer/HNR marked UNRELIABLE. When no silent reference
        exists, SNR is reported as NOT ESTIMABLE (NaN) and does not penalise the
        score or the reliability flag.
        """
        task = task or self._task
        warnings_list = []
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            clip_level = 0.999 * peak
            clipping_fraction = float(np.mean(np.abs(y) >= clip_level))
        else:
            clipping_fraction = 0.0
        if clipping_fraction > 0.005:
            warnings_list.append(f"Clipping detected ({clipping_fraction*100:.1f}% of samples)")

        # ---- SNR from a real silent reference -------------------------------
        noise_floor_db = float("nan")
        snr_db = float("nan")
        snr_estimable = False
        try:
            if rms is None or rms_times is None:
                rms, rms_times = self.compute_rms_envelope(y, sr)
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            hop_s = float(np.median(np.diff(rms_times))) if rms_times.size > 1 else 0.032
            inside = (rms_times >= speech_start) & (rms_times <= speech_end)
            outside = ~inside
            # 1st choice: leading/trailing silence (room tone)
            sil = outside
            self._snr_reference = "leading_trailing"
            if float(np.count_nonzero(sil)) * hop_s < 0.30:
                # 2nd choice: below-threshold frames inside the speech span
                sil = inside & (rms_db < silence_threshold_db)
                self._snr_reference = "internal_below_threshold"
            # v9: on a file that was CUT AND CONCATENATED, the "silence" is the
            # silence the editor pasted in, not room tone, so the SNR describes
            # the edit and not the recording chain. Flag near-digital silence.
            if np.any(sil):
                q10 = float(np.percentile(rms_db[sil], 10))
                self._digital_silence = bool(q10 < -85.0)
                if self._digital_silence:
                    warnings_list.append(
                        "The silent reference is at the numerical floor (<-85 dB): this "
                        "looks like pasted/edited silence, so SNR describes the EDIT, "
                        "not the recording. Measure SNR on the original uncut file.")
            if float(np.count_nonzero(sil)) * hop_s >= 0.30:
                noise_floor_db = float(np.median(rms_db[sil]))
                sp = rms_db[inside] if np.any(inside) else rms_db
                speech_db = float(np.median(sp[sp >= np.percentile(sp, 50)]))
                snr_db = speech_db - noise_floor_db
                snr_estimable = True
        except Exception:
            pass
        if snr_estimable:
            if snr_db < 20:
                warnings_list.append(f"Low SNR ({snr_db:.0f} dB) - noisy recording")
        else:
            self._snr_reference = "none"
            warnings_list.append(
                "SNR NOT ESTIMABLE: no silent reference in this file (speech/phonation "
                "runs edge to edge). Record 1-2 s of room tone before and after the task "
                "to enable a real SNR. Voice-quality metrics are NOT penalised for this.")

        effective_bw = sr / 2.0
        spectral_edge_hz = sr / 2.0
        is_bandlimited = False
        try:
            s0 = int(max(0, speech_start) * sr)
            s1 = int(min(len(y) / sr, speech_end) * sr)
            seg = y[s0:s1] if s1 > s0 else y
            if seg.size >= 2048:
                S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))
                psd = np.mean(S ** 2, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                total = float(np.sum(psd))
                if total > 0:
                    cumulative = np.cumsum(psd) / total
                    idx = int(min(np.searchsorted(cumulative, 0.99), len(freqs) - 1))
                    effective_bw = float(freqs[idx])
                    # v8: discontinuity-based band-limit test (see detect_band_limit)
                    is_bandlimited, cut_edge_hz, cliff_drop_db = detect_band_limit(
                        psd, freqs, sr)
                    spectral_edge_hz = cut_edge_hz
                    if is_bandlimited:
                        effective_bw = cut_edge_hz
                        warnings_list.append(
                            f"Band-limited CHANNEL detected (hard cutoff ~{cut_edge_hz:.0f} Hz, "
                            f"{cliff_drop_db:.0f} dB step down to the noise floor) - "
                            "telephone/codec; shimmer & HNR will read worse than reality")
                    elif effective_bw < 2000.0 and task != TASK_SUSTAINED:
                        warnings_list.append(
                            f"Note: low 99%-energy roll-off (~{effective_bw:.0f} Hz) on a clean, "
                            "full-band channel (no cutoff detected); voice-quality metrics kept "
                            "and trusted - informational only")
        except Exception:
            pass
        if sr < 16000:
            warnings_list.append(f"Low sample rate ({sr} Hz)")

        score = 100.0
        if snr_estimable:
            score -= max(0.0, (25 - snr_db)) * 2.0
        if is_bandlimited:
            score -= 30.0
        score -= min(30.0, clipping_fraction * 2000.0)
        if sr < 16000:
            score -= 15.0
        score = float(max(0.0, min(100.0, score)))
        label = "Good" if score >= 75 else ("Fair" if score >= 50 else "Poor")
        # A NON-ESTIMABLE SNR is not evidence of a bad recording, so it must not
        # switch off the voice-quality metrics.
        snr_ok = (snr_db >= 18) if snr_estimable else True
        vq_reliable = (not is_bandlimited) and snr_ok and (clipping_fraction < 0.01)
        if not vq_reliable:
            reasons = []
            if is_bandlimited: reasons.append("band-limited channel")
            if snr_estimable and snr_db < 18: reasons.append(f"low SNR ({snr_db:.0f} dB)")
            if clipping_fraction >= 0.01: reasons.append("clipping")
            warnings_list.append(
                "Voice-quality metrics (jitter/shimmer/HNR) flagged UNRELIABLE: "
                + ", ".join(reasons))
        return RecordingQualityMetrics(
            snr_db=snr_db, effective_bandwidth_hz=effective_bw,
            spectral_edge_hz=spectral_edge_hz, is_bandlimited=is_bandlimited,
            clipping_fraction=clipping_fraction, noise_floor_db=noise_floor_db,
            sample_rate_hz=int(sr), quality_score=score, quality_label=label,
            vq_reliable=vq_reliable, warnings=warnings_list)

    # ----------------------------- INTENSITY -----------------------------------
    def analyze_intensity(self, sound, speech_start, speech_end,
                          active_speech_mask=None, rms_times=None) -> IntensityMetrics:
        try:
            intensity = call(sound, "To Intensity", self._adaptive_floor, self.time_step, "yes")
            n_frames = call(intensity, "Get number of frames")
            i_t0 = call(intensity, "Get time from frame number", 1)
            i_dt = call(intensity, "Get time step")
            if i_dt is None or i_dt <= 0 or np.isnan(i_dt):
                i_dt = 0.01
            all_db, all_t = [], []
            for i in range(1, n_frames + 1):
                t = i_t0 + (i - 1) * i_dt
                if speech_start <= t <= speech_end:
                    v = call(intensity, "Get value in frame", i)
                    if v is not None and not np.isnan(v) and v > 0:
                        all_db.append(float(v)); all_t.append(float(t))
            all_db = np.asarray(all_db); all_t = np.asarray(all_t)
            if all_db.size < 3:
                return IntensityMetrics(assessment="Insufficient intensity data")
            ref_hi = float(np.percentile(all_db, 95))
            active = all_db >= (ref_hi - 25.0)
            if np.count_nonzero(active) < 3:
                active = np.ones_like(all_db, dtype=bool)
            active_db = all_db[active]
            if active_speech_mask is not None and rms_times is not None:
                try:
                    speech_mask = (rms_times >= speech_start) & (rms_times <= speech_end)
                    rms_speech_t = rms_times[speech_mask]
                    if len(active_speech_mask) == len(rms_speech_t) and len(rms_speech_t) > 3:
                        amask = np.interp(all_t, rms_speech_t,
                                          active_speech_mask.astype(float)) > 0.5
                        if np.count_nonzero(amask) >= 3:
                            active_db = all_db[amask]
                except Exception:
                    pass
            mean_db = float(np.mean(active_db)); std_db = float(np.std(active_db))
            min_db = float(np.min(active_db)); max_db = float(np.max(active_db))
            median_db = float(np.median(active_db))
            q25_db = float(np.percentile(active_db, 25))
            q75_db = float(np.percentile(active_db, 75))
            cv = float("nan")
            nucleus = all_db >= (ref_hi - 15.0)
            nucleus_std_db = float(np.std(all_db[nucleus])) \
                if np.count_nonzero(nucleus) >= 3 else std_db
            if std_db < 3:
                assessment = "Very stable intensity - consistent voice projection"
            elif std_db < 5:
                assessment = "Stable intensity - normal variation"
            elif std_db < 8:
                assessment = "Moderate intensity variation"
            elif std_db < 12:
                assessment = "High intensity variation - noticeable fluctuations"
            else:
                assessment = "Very high intensity variation - may indicate control issues"
            return IntensityMetrics(
                mean_db=mean_db, std_db=std_db, min_db=min_db, max_db=max_db,
                range_db=max_db - min_db, median_db=median_db,
                quantile_25_db=q25_db, quantile_75_db=q75_db,
                coefficient_of_variation=cv, active_mean_db=mean_db,
                active_std_db=std_db, nucleus_std_db=nucleus_std_db,
                assessment=assessment)
        except Exception as e:
            return IntensityMetrics(assessment=f"Error in intensity analysis: {e}")

    # ------------------------------- PITCH -------------------------------------
    def analyze_pitch(self, sound, speech_start, speech_end):
        try:
            floor = self._adaptive_floor; ceiling = self._adaptive_ceiling
            # FIXED in v12: voicing_threshold was accepted by __init__, stored,
            # and then ignored - this call hardcoded 0.50, i.e. STRICTER than
            # Praat's own default of 0.45. On a rough voice that difference alone
            # decides whether a frame counts as voiced, and therefore whether a
            # whole file becomes "not measurable".
            pitch = call(sound, "To Pitch (ac)", 0.0, floor, 15, "no",
                         self.silence_threshold, self.voicing_threshold,
                         0.01, 0.35, 0.14, ceiling)
            p_start = float(call(pitch, "Get start time"))
            p_end = float(call(pitch, "Get end time"))
            start_time = max(float(speech_start), p_start)
            end_time = min(float(speech_end), p_end)
            n_frames = int(call(pitch, "Get number of frames"))
            if n_frames <= 0 or end_time <= start_time:
                return PitchMetrics(
                    assessment="Insufficient frames for pitch analysis"), pitch

            # v20: TIMING IS PRESERVED. The old loop appended only the voiced
            # values, throwing the time axis away, so an unvoiced frame and a
            # missing frame became indistinguishable and nothing downstream
            # could ask WHERE the voicing failed. Unvoiced frames are now NaN in
            # place, which is also what repair_octave_jumps expects.
            times, vals = [], []
            n_in_range = 0
            for i in range(1, n_frames + 1):
                t = float(call(pitch, "Get time from frame number", i))
                if start_time <= t <= end_time:
                    n_in_range += 1
                    v = call(pitch, "Get value in frame", i, "Hertz")
                    vals.append(float(v) if v is not None and np.isfinite(v) and v > 0
                                else np.nan)
                    times.append(t)

            f0 = np.asarray(vals, dtype=float)
            voiced = np.isfinite(f0) & (f0 > 0)
            n_voiced = int(np.count_nonzero(voiced))
            if n_voiced < 5:
                vp = (100.0 * n_voiced / n_in_range) if n_in_range else float("nan")
                return PitchMetrics(
                    voiced_frames_percent=vp,
                    unvoiced_frames_percent=(100.0 - vp if np.isfinite(vp) else np.nan),
                    voiced_to_total_ratio=(vp / 100.0 if np.isfinite(vp) else np.nan),
                    assessment="Insufficient voiced frames for pitch analysis"), pitch

            # Octave-error repair BEFORE any statistic is taken (see docstring of
            # repair_octave_jumps). Without this, F0 mean/SD/CV and every
            # perturbation measure inherit the doubling population.
            repaired, oct_frac = repair_octave_jumps(f0)
            ok = np.isfinite(repaired) & (repaired > 0)
            voiced_vals = repaired[ok]
            if voiced_vals.size < 5:
                return PitchMetrics(
                    assessment="Insufficient repaired F0 frames"), pitch

            q01, q99 = np.percentile(voiced_vals, [1, 99])
            used = voiced_vals[(voiced_vals >= q01) & (voiced_vals <= q99)]
            if used.size < 5:
                used = voiced_vals

            mean_hz = float(np.mean(used)); median_hz = float(np.median(used))
            std_hz = float(np.std(used)); min_hz = float(np.min(used)); max_hz = float(np.max(used))
            q25_hz, q75_hz = [float(x) for x in np.percentile(used, [25, 75])]

            # ---- SEMITONE STATISTICS (v20) ----------------------------------
            # std_semitones was NOT an SD. It was 12*log2((mean+sd)/mean), i.e.
            # the semitone DISTANCE from the mean to one Hz-SD above it - a
            # one-sided transform of the Hz SD, not the SD of the semitone
            # contour. Because the Hz->semitone map is logarithmic the two
            # differ, and the error grows with F0 variability, so it was largest
            # exactly on the unstable voices the measure is meant to describe.
            # Now: convert the contour to semitones first, then take the SD.
            st = hz_to_semitones(used, median_hz)
            std_st = float(np.nanstd(st))
            range_st = float(np.nanmax(st) - np.nanmin(st))
            # mean_semitones stays a PITCH LEVEL re 100 Hz (as before). Taken re
            # median_hz it would be ~0 by construction and carry no information.
            mean_st = float(12.0 * np.log2(mean_hz / 100.0)) if mean_hz > 0 else float("nan")
            cv = float(std_hz / mean_hz) if mean_hz > 0 else float("nan")

            vp = float(np.clip(100.0 * n_voiced / n_in_range, 0.0, 100.0)) \
                if n_in_range else float("nan")
            uvp = (100.0 - vp) if np.isfinite(vp) else float("nan")

            # v20: the old labels ("could benefit from more expression",
            # "may sound stressed") read as coaching feedback and were applied
            # to sustained phonation too, where low F0 variation is the TASK.
            if cv < 0.05:
                assessment = "Very low F0 variation"
            elif cv < 0.10:
                assessment = "Low-to-moderate F0 variation"
            elif cv < 0.20:
                assessment = "Moderate F0 variation"
            else:
                assessment = "High F0 variation"

            return PitchMetrics(
                mean_hz=mean_hz, std_hz=std_hz, min_hz=min_hz, max_hz=max_hz,
                range_hz=max_hz - min_hz, median_hz=median_hz,
                quantile_25_hz=q25_hz, quantile_75_hz=q75_hz,
                coefficient_of_variation=cv, mean_semitones=mean_st,
                std_semitones=std_st, range_semitones=range_st,
                voiced_frames_percent=vp, unvoiced_frames_percent=uvp,
                voiced_to_total_ratio=(vp / 100.0 if np.isfinite(vp) else float("nan")),
                octave_repair_fraction=float(oct_frac),
                assessment=assessment), pitch
        except Exception as e:
            return PitchMetrics(assessment=f"Error in pitch analysis: {e}"), None

    # --------------------------- VOICE QUALITY ---------------------------------
    def analyze_voice_quality(self, sound, pitch, speech_start, speech_end,
                              recording_quality=None,
                              intervals_override=None) -> VoiceQualityMetrics:
        """
        Praat Voice Report equivalent.

        intervals_override (v8): restrict measurement to specific spans, e.g. the
        steady-state mid-window of a sustained vowel. Averaging perturbation
        measures over a whole multi-trial file - across onsets, offsets and
        breath resets - is the single largest source of inflated jitter/shimmer
        and depressed HNR in sustained-phonation data.
        """
        try:
            floor = self._adaptive_floor; ceiling = self._adaptive_ceiling
            if pitch is None:
                pitch = call(sound, "To Pitch (cc)", 0.0, floor, 15, "no",
                             0.03, 0.45, 0.01, 0.35, 0.14, ceiling)
            if intervals_override:
                intervals = list(intervals_override)
                speech_start = min(s for s, _ in intervals)
                speech_end = max(e for _, e in intervals)
            else:
                intervals = self.get_voiced_intervals(sound, pitch, floor)
            if not intervals:
                return VoiceQualityMetrics(
                    assessment="No reliable voiced segments found for voice-quality analysis")
            snd_start = call(sound, "Get start time"); snd_end = call(sound, "Get end time")
            # v9: the PointProcess covers the WHOLE file and does not depend on
            # the interval being measured, but it used to be recomputed for
            # every analysis window (~20 Praat calls over 70 s of audio per
            # file). Cache it per (sound, pitch) pair.
            # v13: keyed on id() plus the file currently being analysed. CPython
            # REUSES object addresses once an object is freed, so across files a
            # new (sound, pitch) pair could land on the same id pair as a freed
            # one and silently be handed a STALE PointProcess - i.e. one file
            # measured with another file's glottal pulses. The cache is also
            # cleared at the start of every analyze(), which is the real
            # guarantee; the file tag makes the failure impossible rather than
            # unlikely.
            pp_key = (getattr(self, "_current_file", ""), id(sound), id(pitch))
            if getattr(self, "_pp_cache_key", None) == pp_key:
                point_process = self._pp_cache
            else:
                point_process = call([sound, pitch], "To PointProcess (cc)")
                self._pp_cache_key, self._pp_cache = pp_key, point_process
            period_floor = 1.0 / ceiling; period_ceiling = 1.0 / floor
            mpf = self.max_period_factor; maf = self.max_amplitude_factor

            def q(fn, default=np.nan):
                try:
                    v = fn()
                    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        return default
                    return v
                except Exception:
                    return default

            jl_list, jabs_list, jrap_list, jppq_list, jddp_list = [], [], [], [], []
            sl_list, sldb_list, sa3_list, sa5_list, sa11_list, sdda_list = [], [], [], [], [], []
            weights = []; total_voiced_dur = 0.0
            for s, e in intervals:
                s = max(s, snd_start); e = min(e, snd_end); dur = e - s
                if dur < 0.10:
                    continue
                # v20: EACH METRIC FAILS INDEPENDENTLY. This used to be
                # `if np.isnan(jl): continue`, so a single unmeasurable jitter
                # threw away the shimmer, HNR and autocorrelation that Praat had
                # measured on the same interval - biasing every OTHER column
                # towards the intervals where jitter happened to succeed.
                jl = q(lambda: call(point_process, "Get jitter (local)",
                                    s, e, period_floor, period_ceiling, mpf))
                jabs = q(lambda: call(point_process, "Get jitter (local, absolute)",
                                      s, e, period_floor, period_ceiling, mpf))
                jrap = q(lambda: call(point_process, "Get jitter (rap)",
                                      s, e, period_floor, period_ceiling, mpf))
                jppq = q(lambda: call(point_process, "Get jitter (ppq5)",
                                      s, e, period_floor, period_ceiling, mpf))
                jddp = q(lambda: call(point_process, "Get jitter (ddp)",
                                      s, e, period_floor, period_ceiling, mpf))
                sl = q(lambda: call([sound, point_process], "Get shimmer (local)",
                                    s, e, period_floor, period_ceiling, mpf, maf))
                sldb = q(lambda: call([sound, point_process], "Get shimmer (local_dB)",
                                      s, e, period_floor, period_ceiling, mpf, maf))
                sa3 = q(lambda: call([sound, point_process], "Get shimmer (apq3)",
                                     s, e, period_floor, period_ceiling, mpf, maf))
                sa5 = q(lambda: call([sound, point_process], "Get shimmer (apq5)",
                                     s, e, period_floor, period_ceiling, mpf, maf))
                sa11 = q(lambda: call([sound, point_process], "Get shimmer (apq11)",
                                      s, e, period_floor, period_ceiling, mpf, maf))
                sdda = q(lambda: call([sound, point_process], "Get shimmer (dda)",
                                      s, e, period_floor, period_ceiling, mpf, maf))
                jl_list.append(jl); jabs_list.append(jabs); jrap_list.append(jrap)
                jppq_list.append(jppq); jddp_list.append(jddp)
                sl_list.append(sl); sldb_list.append(sldb); sa3_list.append(sa3)
                sa5_list.append(sa5); sa11_list.append(sa11); sdda_list.append(sdda)
                weights.append(dur); total_voiced_dur += dur
            if not weights:
                return VoiceQualityMetrics(
                    assessment="No voiced interval was long enough for voice-quality analysis")
            w = np.asarray(weights)

            def agg(vals, scale=1.0):
                """
                FIXED in v9 (two bugs):
                  - returned 0.0 when every Praat call had failed, so an
                    UNMEASURABLE jitter/shimmer entered the CSV and the plots as
                    a perfect 0.0. Now NaN.
                  - trimmed only the UPPER decile (a <= p90), i.e. a one-sided
                    trim that biases the estimate downwards by an amount that
                    depends on how many intervals the file happens to have -
                    so files were not comparable. Now the plain MEDIAN across
                    intervals (robust and symmetric); the duration-weighted mean
                    is kept only for the 1-2 interval case, where a median is
                    meaningless.
                """
                a = np.asarray(vals, dtype=float) * scale
                ww = w.copy()
                ok = np.isfinite(a)
                a, ww = a[ok], ww[ok]
                if getattr(self, "legacy_compat", False):
                    if a.size == 0:
                        return 0.0                      # pre-v9 behaviour (wrong)
                    if a.size >= 5:
                        cutoff = np.percentile(a, 90); keep = a <= cutoff
                        a, ww = a[keep], ww[keep]
                    if ww.sum() <= 0:
                        return float(np.mean(a))
                    return float(np.average(a, weights=ww))
                if a.size == 0:
                    return float("nan")
                if a.size >= 3:
                    return float(np.median(a))
                if ww.sum() <= 0:
                    return float(np.mean(a))
                return float(np.average(a, weights=ww))

            jitter_local = agg(jl_list, 100.0); jitter_local_abs = agg(jabs_list, 1.0)
            jitter_rap = agg(jrap_list, 100.0); jitter_ppq5 = agg(jppq_list, 100.0)
            jitter_ddp = agg(jddp_list, 100.0); shimmer_local = agg(sl_list, 100.0)
            shimmer_local_db = agg(sldb_list, 1.0); shimmer_apq3 = agg(sa3_list, 100.0)
            shimmer_apq5 = agg(sa5_list, 100.0); shimmer_apq11 = agg(sa11_list, 100.0)
            shimmer_dda = agg(sdda_list, 100.0)

            # v16 BUG FIX: these were initialised to 0.0. If the "Voice report"
            # call itself RAISED (rather than returning an unparseable line), the
            # bare `except: pass` below left hnr_db at 0.0 - and 0.0 is finite, so
            # the `if not np.isfinite(hnr_db)` harmonicity fallback was skipped
            # entirely and HNR was reported as exactly 0.0 dB. That is the same
            # fake-zero failure v9 item D set out to remove: the fix had been
            # applied to the PARSING path but not to the EXCEPTION path. Start
            # from NaN so an unmeasurable value can never masquerade as a
            # measured one, and so the fallback actually runs.
            hnr_db = float("nan"); mean_autocorr = float("nan")
            try:
                report = call([sound, pitch, point_process], "Voice report",
                              speech_start, speech_end, floor, ceiling,
                              mpf, maf, 0.03, 0.45)
                def vr(label):
                    for line in report.split("\n"):
                        if label in line:
                            seg = line.split(":", 1)[1]
                            seg = seg.replace("%", "").replace("Hz", "").replace("dB", "")
                            seg = seg.strip().split()[0]
                            try:
                                return float(seg)
                            except ValueError:
                                return None
                    return None
                # v9: `x or 0.0` turned a legitimate HNR of 0 dB (and any
                # negative HNR, which a very noisy voice really can have) into
                # "missing", and then into a silent fallback. Distinguish
                # missing (None) from a measured value.
                _h = vr("Mean harmonics-to-noise ratio")
                _a = vr("Mean autocorrelation")
                hnr_db = float(_h) if _h is not None else float("nan")
                mean_autocorr = float(_a) if _a is not None else float("nan")
            except Exception:
                pass
            if not np.isfinite(hnr_db):
                try:
                    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, floor, 0.1, 4.5)
                    hn = call(harmonicity, "Get number of frames")
                    hdt = q(lambda: call(harmonicity, "Get time step"), 0.01)
                    ht0 = q(lambda: call(harmonicity, "Get time from frame number", 1), snd_start)
                    iv = np.array(intervals); hvals = []
                    for i in range(1, hn + 1):
                        t = ht0 + (i - 1) * hdt
                        inside = np.any((t >= iv[:, 0]) & (t <= iv[:, 1])) if iv.size else False
                        if not inside:
                            continue
                        hv = call(harmonicity, "Get value in frame", i)
                        # v20: see note in hnr_median_perinterval - a
                        # NEGATIVE harmonicity is a real value.
                        if hv is not None and np.isfinite(hv):
                            hvals.append(float(hv))
                    if hvals:
                        hnr_db = float(np.mean(hvals))
                except Exception:
                    # v16: was `hnr_db = 0.0`, i.e. a second fake zero on the
                    # fallback's own failure path.
                    hnr_db = float("nan")
            # v16 BUG FIX: NHR was `10**(-hnr/10) if hnr > 0 else inf`, which was
            # wrong twice over.
            #   - An unmeasurable HNR (NaN) failed the `> 0` test and became
            #     +inf: a value that then propagated into medians, means and bar
            #     plots as if it were measured.
            #   - HNR of exactly 0 dB, or negative, is PHYSICALLY REAL for a very
            #     noisy voice (equal or greater noise than harmonic energy) and
            #     also became +inf instead of the correct 1.0 / >1.0.
            # NHR is the reciprocal power ratio at any sign of HNR; only a
            # genuinely missing HNR is missing.
            nhr = float(10.0 ** (-hnr_db / 10.0)) if np.isfinite(hnr_db) else float("nan")

            num_pulses = int(q(lambda: call(point_process, "Get number of points"), 0))
            voice_breaks_count = 0; total_break_dur = 0.0
            try:
                if num_pulses >= 3:
                    pulse_t = np.array([call(point_process, "Get time from index", k)
                                        for k in range(1, num_pulses + 1)])
                    iv = np.array(intervals)
                    for s, e in iv:
                        seg_pulses = pulse_t[(pulse_t >= s) & (pulse_t <= e)]
                        if seg_pulses.size >= 4:
                            gaps = np.diff(seg_pulses); med_gap = np.median(gaps)
                            thr = max(4.0 * med_gap, 0.040); brk = gaps > thr
                            voice_breaks_count += int(np.sum(brk))
                            total_break_dur += float(np.sum(gaps[brk]))
            except Exception:
                pass
            voice_breaks_degree = (total_break_dur / total_voiced_dur * 100.0) \
                if total_voiced_dur > 0 else 0.0
            num_periods = max(0, num_pulses - len(intervals))

            pathology_indicators = []
            # NOTE (v16): each test below is written so that a NaN input fails it
            # silently (NaN < 10 is False), which is correct - missing data must
            # not raise a pathology flag. But the converse then needs guarding:
            # with no flags raised the assessment used to read "within normal
            # limits" even when the measure that would have raised the flag was
            # never obtained. Unmeasurable is not normal, so track it.
            unmeasured = [name for name, val in (("jitter", jitter_local),
                                                 ("shimmer", shimmer_local),
                                                 ("HNR", hnr_db))
                          if not np.isfinite(val)]
            if jitter_local > 3.0:
                pathology_indicators.append(f"Elevated jitter ({jitter_local:.2f}% > 3.0%)")
            if shimmer_local > 10.0:
                pathology_indicators.append(f"Elevated shimmer ({shimmer_local:.2f}% > 10.0%)")
            if hnr_db < 10:
                pathology_indicators.append(f"Low HNR ({hnr_db:.1f} dB < 10 dB)")
            if voice_breaks_degree > 8:
                pathology_indicators.append(f"Voice breaks ({voice_breaks_degree:.1f}%)")
            if not pathology_indicators:
                assessment = ("Voice quality within normal limits (running speech)"
                              if not unmeasured else
                              "No deviation flagged, but "
                              + "/".join(unmeasured)
                              + " could NOT be measured - this is missing data, "
                                "not a normal result")
            elif len(pathology_indicators) == 1:
                assessment = "Mild voice-quality deviation"
            elif len(pathology_indicators) == 2:
                assessment = "Moderate voice-quality deviation"
            else:
                assessment = "Multiple voice-quality deviations detected"
            if pathology_indicators and unmeasured:
                assessment += (" (note: " + "/".join(unmeasured)
                               + " not measurable, so the severity is based on "
                                 "an incomplete panel)")
            if recording_quality is not None and not recording_quality.vq_reliable:
                assessment = ("Voice-quality metrics UNRELIABLE due to recording quality "
                              f"({recording_quality.quality_label}); interpret jitter/shimmer/HNR "
                              "with caution. Raw measures: " + assessment)
                pathology_indicators = [
                    "Recording quality too low for reliable voice-quality assessment"]
            return VoiceQualityMetrics(
                jitter_local_percent=jitter_local, jitter_local_abs_sec=jitter_local_abs,
                jitter_rap_percent=jitter_rap, jitter_ppq5_percent=jitter_ppq5,
                jitter_ddp_percent=jitter_ddp, shimmer_local_percent=shimmer_local,
                shimmer_local_db=shimmer_local_db, shimmer_apq3_percent=shimmer_apq3,
                shimmer_apq5_percent=shimmer_apq5, shimmer_apq11_percent=shimmer_apq11,
                shimmer_dda_percent=shimmer_dda, hnr_db=hnr_db, nhr=nhr,
                voice_breaks_count=voice_breaks_count,
                voice_breaks_degree_percent=voice_breaks_degree,
                num_pulses=num_pulses, num_periods=num_periods,
                mean_autocorrelation=mean_autocorr, fraction_unvoiced_percent=0.0,
                assessment=assessment, pathology_indicators=pathology_indicators,
                # v20: was a hardcoded True, which claimed a measurement even
                # when every single metric came back NaN.
                measured=bool(np.isfinite(jitter_local) or np.isfinite(shimmer_local)
                              or np.isfinite(hnr_db)),
                reliable=(recording_quality.vq_reliable if recording_quality is not None else True))
        except Exception as e:
            import traceback; traceback.print_exc()
            return VoiceQualityMetrics(assessment=f"Error in voice quality analysis: {e}")

    # ------------------------------ FORMANTS -----------------------------------
    def analyze_formants(self, sound, speech_start, speech_end, pitch=None,
                         intervals: Optional[List[Tuple[float, float]]] = None
                         ) -> FormantMetrics:
        """
        LPC formant tracking with plausibility screening.

        intervals (v19): keep only frames inside these spans. F1/F2 were
        previously collected from every voiced frame between speech_start and
        speech_end even on a sustained file, so sv_f1f2_cloud_spread_hz - sold
        as ARTICULATORY STEADINESS on one held vowel - was inflated by the
        onset and offset transitions of every token and by any drift between
        them. The caller now passes the valid steady-state windows, which is the
        same signal the perturbation medians come from.
        """
        try:
            # Formant ceiling from the speaker's own F0 (vocal-tract length
            # proxy). Keeping the 5500 Hz default for a ~100 Hz male voice is
            # what lets F3 collapse onto F2.
            # v10: with a per-subject formant_ceiling_hz the ceiling stops
            # depending on the TRACKED F0 - which is the wrong dependency when
            # the tracker can be an octave off, and which made F1/F2 shift
            # between sessions of the same speaker (5000 vs 5250 vs 5500 Hz).
            if self.formant_ceiling_hz:
                max_formant = float(self.formant_ceiling_hz)
            elif self._adaptive_median >= 190.0:
                max_formant = 5500.0
            elif self._adaptive_median >= 145.0:
                max_formant = 5250.0
            else:
                max_formant = 5000.0
            formant = call(sound, "To Formant (burg)", self.time_step,
                           self.num_formants, max_formant, 0.025, 50)
            start_time = max(speech_start, call(formant, "Get start time"))
            end_time = min(speech_end, call(formant, "Get end time"))
            spans = [(max(float(a), start_time), min(float(b), end_time))
                     for a, b in (intervals or [])]
            spans = [(a, b) for a, b in spans if b > a]

            def in_span(t):
                if not spans:
                    return start_time <= t <= end_time
                return any(a <= t <= b for a, b in spans)

            def is_voiced_at(t):
                if pitch is None:
                    return True
                try:
                    v = call(pitch, "Get value at time", t, "Hertz", "Linear")
                    return v is not None and v > 0 and not np.isnan(v)
                except Exception:
                    return True

            def trim_iqr(arr):
                a = np.asarray(arr)
                if a.size < 4:
                    return a
                q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                t = a[(a >= lo) & (a <= hi)]
                return t if t.size >= 3 else a

            def get_formant_stats(fn):
                values = []
                nfr = call(formant, "Get number of frames")
                for i in range(1, nfr + 1):
                    t = call(formant, "Get time from frame number", i)
                    if in_span(t) and is_voiced_at(t):
                        val = call(formant, "Get value at time", fn, t, "Hertz", "Linear")
                        if val > 0 and not np.isnan(val):
                            values.append(val)
                if values:
                    vt = trim_iqr(values)
                    return {'mean': float(np.mean(vt)), 'std': float(np.std(vt)),
                            'min': float(np.min(vt)), 'max': float(np.max(vt)),
                            'values': list(vt)}
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'values': []}

            f1 = get_formant_stats(1); f2 = get_formant_stats(2)
            f3 = get_formant_stats(3); f4 = get_formant_stats(4)
            vowel_space_area = 0
            if f1['values'] and f2['values']:
                try:
                    points = np.array(list(zip(f1['values'][:100], f2['values'][:100])))
                    if len(points) >= 4:
                        hull = ConvexHull(points); vowel_space_area = hull.volume
                except Exception:
                    pass
            fcr = f2['mean'] / f1['mean'] if f1['mean'] > 0 else 0
            f2_f1_ratio = f2['mean'] / f1['mean'] if f1['mean'] > 0 else 0
            # SANITY CHECK (v8): F3 must sit clearly above F2. When it does not,
            # the LPC tracker has merged or swapped formants (this is what an
            # "F3 = 1666 Hz with F2 = 1327 Hz" line means) and F3/F4 must not be
            # reported as if they were measurements.
            #
            # v9: the relative test alone let an obvious failure through -
            # "F3 = 1686 Hz, F2 = 1326 Hz" clears F2 + 250 Hz but no adult /a/
            # has F3 below ~2 kHz. Absolute plausibility bounds are now applied
            # per formant as well, and the ordering F1 < F2 < F3 is enforced.
            track_ok = True
            fail_reasons = []
            F_BOUNDS = {'F1': (200.0, 1200.0), 'F2': (600.0, 3000.0),
                        'F3': (1800.0, 4200.0)}
            # v16: failures are now separated by TIER, because the earlier code
            # blanked only F3/F4 no matter what had failed. If the implausible
            # formant was F1 or F2, the bad F1/F2 values were still returned as
            # measurements AND still flowed into vowel_space_area,
            # vowel_dispersion_logarea, vowel_cloud_spread_hz and
            # sv_f1f2_cloud_spread_hz - i.e. into the articulation outcomes people
            # actually compare between sessions - with only formant_track_ok=0 in
            # the audit block to hint that anything was wrong. A low-tier failure
            # must invalidate everything derived from the low formants.
            low_fail, high_fail = [], []
            if f3['mean'] > 0 and f2['mean'] > 0 and f3['mean'] < f2['mean'] + 250.0:
                high_fail.append("F3 <= F2 + 250 Hz")
            if getattr(self, "legacy_compat", False):
                F_BOUNDS = {'F1': (0.0, 1e9), 'F2': (0.0, 1e9), 'F3': (0.0, 1e9)}
            for name, vals in (('F1', f1), ('F2', f2), ('F3', f3)):
                lo, hi = F_BOUNDS[name]
                m = vals['mean']
                if m is not None and np.isfinite(m) and m > 0 and not (lo <= m <= hi):
                    msg = f"{name}={m:.0f} Hz outside {lo:.0f}-{hi:.0f} Hz"
                    (low_fail if name in ('F1', 'F2') else high_fail).append(msg)
            if (not getattr(self, "legacy_compat", False)
                    and f1['mean'] > 0 and f2['mean'] > 0
                    and f2['mean'] <= f1['mean'] + 100.0):
                low_fail.append("F2 <= F1 + 100 Hz")
            fail_reasons = low_fail + high_fail
            _nanf = lambda: {'mean': float('nan'), 'std': float('nan'),
                             'min': float('nan'), 'max': float('nan'), 'values': []}
            if fail_reasons:
                track_ok = False
                f3 = _nanf()
                f4 = _nanf()
            if low_fail:
                # The F1-F2 plane itself is untrustworthy: discard it and
                # everything computed from it rather than exporting a vowel space
                # measured off a failed track.
                f1 = _nanf()
                f2 = _nanf()
                vowel_space_area = float('nan')
                fcr = float('nan')
                f2_f1_ratio = float('nan')
            if f1['mean'] > 0 and f2['mean'] > 0:
                if f1['std'] < 100 and f2['std'] < 200:
                    assessment = "Stable formant production"
                elif f1['std'] < 150 and f2['std'] < 300:
                    assessment = "Normal formant variation"
                else:
                    assessment = "High formant variability - may indicate articulatory instability"
            else:
                assessment = "Insufficient data for formant assessment"
            if not track_ok:
                scope = ("F1/F2 AND F3/F4 (the whole formant track, so the vowel-space "
                         "metrics are NaN too)" if low_fail else "F3/F4")
                assessment = (f"FORMANT TRACKING FAILED ({'; '.join(fail_reasons)}): "
                              f"{scope} set to NaN. "
                              f"Try a max_formant other than {max_formant:.0f} Hz. "
                              + assessment)
            return FormantMetrics(
                f1_mean_hz=f1['mean'], f1_std_hz=f1['std'],
                f1_min_hz=f1['min'], f1_max_hz=f1['max'],
                f2_mean_hz=f2['mean'], f2_std_hz=f2['std'],
                f2_min_hz=f2['min'], f2_max_hz=f2['max'],
                f3_mean_hz=f3['mean'], f3_std_hz=f3['std'],
                f3_min_hz=f3['min'], f3_max_hz=f3['max'],
                f4_mean_hz=f4['mean'], f4_std_hz=f4['std'],
                vowel_space_area=vowel_space_area,
                formant_centralization_ratio=fcr, f2_f1_ratio_mean=f2_f1_ratio,
                track_ok=track_ok, max_formant_hz_used=float(max_formant),
                assessment=assessment,
                _f1_values=f1['values'], _f2_values=f2['values'])
        except Exception as e:
            return FormantMetrics(assessment=f"Error in formant analysis: {e}")

    # ------------------------------ SPECTRAL -----------------------------------
    def _concat_intervals(self, y, sr, intervals, fade_s: float = 0.005):
        """
        Samples inside `intervals`, concatenated, with a short raised-cosine
        fade at every join.

        Used to keep the LTAS and the cepstrogram off the silence BETWEEN
        sustained-vowel tokens. Without the fades each join is a step
        discontinuity, i.e. broadband click energy, which is exactly what the
        alpha ratio and the Hammarberg index measure.
        """
        n = int(round(fade_s * sr))
        pieces = []
        for a, b in intervals:
            i0 = max(0, int(a * sr)); i1 = min(len(y), int(b * sr))
            if i1 - i0 < 4:
                continue
            seg = np.array(y[i0:i1], dtype=float, copy=True)
            k = min(n, seg.size // 2)
            if k > 1:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, k)))
                seg[:k] *= ramp
                seg[-k:] *= ramp[::-1]
            pieces.append(seg)
        return np.concatenate(pieces) if pieces else np.array([], dtype=float)

    def analyze_spectral(self, sound, y, sr, speech_start, speech_end,
                         intervals: Optional[List[Tuple[float, float]]] = None,
                         task: Optional[str] = None) -> SpectralMetrics:
        """
        LTAS shape, spectral moments and a smoothed cepstral peak prominence.

        intervals (v19): measure ONLY inside these spans. This block used to run
        on the whole [speech_start, speech_end] slice unconditionally, which on a
        sustained-vowel file with several tokens meant the alpha ratio, the
        Hammarberg index, the tilt and cpp_db all included the inter-token
        silence and the breaths - and those four are precisely the measures this
        script recommends fall back on when jitter and shimmer come back empty.
        The caller now passes the token spans for sustained phonation.

        For connected speech intervals stays None (the whole speech span), which
        is both the standard LTAS practice - fricatives and stops are part of the
        signal there, not gaps in it - and unchanged from earlier versions, so
        reading files remain comparable with previous runs.
        """
        task = task or self._task
        try:
            span_source = "speech_span"
            if intervals:
                y_speech = self._concat_intervals(y, sr, intervals)
                span_source = f"{len(intervals)} interval(s)"
            else:
                start_sample = int(speech_start * sr); end_sample = int(speech_end * sr)
                y_speech = y[start_sample:end_sample]
            if len(y_speech) < sr * 0.1:
                return SpectralMetrics(assessment="Insufficient audio for spectral analysis",
                                       span_source=span_source)
            spectral_centroid = librosa.feature.spectral_centroid(y=y_speech, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_speech, sr=sr)[0]
            rms = librosa.feature.rms(y=y_speech)[0]
            voiced_mask = rms > np.percentile(rms, 20)
            if np.sum(voiced_mask) > 10:
                centroid_voiced = spectral_centroid[voiced_mask]
                bandwidth_voiced = spectral_bandwidth[voiced_mask]
            else:
                centroid_voiced = spectral_centroid; bandwidth_voiced = spectral_bandwidth
            centroid_mean = np.mean(centroid_voiced); centroid_std = np.std(centroid_voiced)
            spread_mean = np.mean(bandwidth_voiced)
            S = np.abs(librosa.stft(y_speech)); freqs = librosa.fft_frequencies(sr=sr)
            skewness_vals, kurtosis_vals = [], []
            for i in range(S.shape[1]):
                spectrum = S[:, i]
                if np.sum(spectrum) > 0:
                    sn = spectrum / np.sum(spectrum)
                    mf = np.sum(freqs * sn); vf = np.sum(((freqs - mf) ** 2) * sn)
                    sf = np.sqrt(vf) if vf > 0 else 1
                    skewness_vals.append(np.sum(((freqs - mf) ** 3) * sn) / (sf ** 3))
                    kurtosis_vals.append(np.sum(((freqs - mf) ** 4) * sn) / (sf ** 4) - 3)
            skewness_mean = np.mean(skewness_vals) if skewness_vals else 0
            kurtosis_mean = np.mean(kurtosis_vals) if kurtosis_vals else 0
            # The Praat Sound the LTAS and the cepstrogram are computed on. When
            # intervals were given, it is built from the concatenated samples so
            # that neither measure ever sees the silence between tokens.
            try:
                if intervals:
                    sound_speech = parselmouth.Sound(y_speech, sampling_frequency=sr)
                else:
                    sound_speech = sound.extract_part(speech_start, speech_end)
            except Exception:
                sound_speech = None
            alpha_ratio = hammarberg = ltas_slope = float("nan")
            if sound_speech is not None:
                try:
                    ltas = call(sound_speech, "To Ltas", 100)
                    energy_low = call(ltas, "Get mean", 50, 1000, "dB")
                    energy_high = call(ltas, "Get mean", 1000, 5000, "dB")
                    alpha_ratio = energy_high - energy_low
                    max_low = call(ltas, "Get maximum", 0, 2000, "None")
                    max_high = call(ltas, "Get maximum", 2000, 5000, "None")
                    hammarberg = max_low - max_high
                    ltas_slope = call(ltas, "Get slope", 0, 1000, 1000, 4000, "dB")
                except Exception:
                    pass

            # ---- SPECTRAL TILT OVER A FIXED BAND (v19) ----------------------
            # The fit used to run over the entire linear axis, 0 Hz to Nyquist,
            # so the value depended on the SAMPLE RATE: the same voice gives one
            # tilt at 16 kHz and another at 48 kHz, because the extra octave is
            # almost pure noise floor and drags the regression down. v17 item 5
            # already warns that a recording-chain difference is
            # indistinguishable from a treatment effect in every spectral
            # measure; this was one such difference built into the metric
            # itself. The band is now fixed at 50-5000 Hz (clipped to Nyquist),
            # which is inside the voice band at every sample rate this script
            # accepts, so files recorded on different equipment are comparable.
            spectral_slope = spectral_tilt = float("nan")
            tilt_lo, tilt_hi = 50.0, min(5000.0, 0.95 * (sr / 2.0))
            mean_spectrum = np.mean(S, axis=1)
            fq = freqs[:len(mean_spectrum)]
            band = (fq >= tilt_lo) & (fq <= tilt_hi)
            if np.count_nonzero(band) > 8 and np.max(mean_spectrum) > 0:
                log_spectrum = 20 * np.log10(mean_spectrum[band] + 1e-10)
                slope, _ = np.polyfit(fq[band], log_spectrum, 1)
                spectral_slope = float(slope); spectral_tilt = float(slope * 1000.0)

            # ---- SMOOTHED CEPSTRAL PEAK PROMINENCE (v19) --------------------
            # This is Praat's "Get CPPS", i.e. a SMOOTHED CPP. Under the old code
            # a Praat failure silently fell through to an FFT peak-to-mean
            # cepstral ratio, and a failure of THAT wrote 0.0 - so one column
            # could hold three incommensurable quantities with no way to tell
            # them apart. The fallback is now labelled and a total failure is
            # NaN, never 0.
            cpp_db = float("nan"); cpp_source = "unavailable"
            if sound_speech is not None:
                try:
                    cepstrogram = call(sound_speech, "To PowerCepstrogram",
                                       self._adaptive_floor, 0.002, 5000, 50)
                    v = call(cepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330,
                             0.05, "Parabolic", 0.001, 0, "Exponential decay", "Robust")
                    if v is not None and np.isfinite(v):
                        cpp_db = float(v); cpp_source = "praat_cpps"
                except Exception:
                    pass
            if not np.isfinite(cpp_db):
                try:
                    cepstrum = np.abs(np.fft.ifft(
                        np.log(np.abs(np.fft.fft(y_speech)) + 1e-10)))
                    min_q = int(sr / self._adaptive_ceiling)
                    max_q = int(sr / self._adaptive_floor)
                    if 0 <= min_q < max_q < len(cepstrum):
                        region = cepstrum[min_q:max_q]
                        baseline = float(np.mean(region))
                        if baseline > 0:
                            cpp_db = float(20 * np.log10(np.max(region) / baseline))
                            # NOT on the same scale as praat_cpps. Flagged so it
                            # can be excluded rather than averaged in.
                            cpp_source = "fft_fallback_NOT_comparable"
                except Exception:
                    pass

            # ---- ASSESSMENT ON THE RIGHT SCALE (v19) ------------------------
            # The thresholds were 8 / 5 / 3 dB, which belong to the unsmoothed
            # CPP. On the CPPS scale a healthy sustained vowel sits around
            # 13-20 dB and connected speech around 8-14 dB, so ">8 = clear
            # voice quality" fired on essentially every file, including severely
            # dysphonic ones, and "breathy or rough" was unreachable.
            if not np.isfinite(cpp_db):
                assessment = "Cepstral peak prominence not measurable"
            elif cpp_source.startswith("fft"):
                assessment = ("Cepstral peak from the FFT fallback - not on the CPPS "
                              "scale, do not compare with other files")
            else:
                hi, mid, lo = ((14.0, 11.0, 8.0) if task == TASK_SUSTAINED
                               else (11.0, 8.5, 6.0))
                if cpp_db >= hi:
                    assessment = f"Strong cepstral peak ({cpp_db:.1f} dB CPPS)"
                elif cpp_db >= mid:
                    assessment = f"Moderate cepstral peak ({cpp_db:.1f} dB CPPS)"
                elif cpp_db >= lo:
                    assessment = (f"Reduced cepstral peak ({cpp_db:.1f} dB CPPS) - "
                                  "consistent with mild dysphonia")
                else:
                    assessment = (f"Low cepstral peak ({cpp_db:.1f} dB CPPS) - "
                                  "breathy or rough voice")
            return SpectralMetrics(
                spectral_centroid_mean_hz=centroid_mean, spectral_centroid_std_hz=centroid_std,
                spectral_spread_mean_hz=spread_mean, spectral_skewness_mean=skewness_mean,
                spectral_kurtosis_mean=kurtosis_mean, spectral_slope=spectral_slope,
                spectral_tilt_db=spectral_tilt, ltas_slope=ltas_slope,
                alpha_ratio=alpha_ratio, hammarberg_index=hammarberg,
                cpp_db=cpp_db, cpp_source=cpp_source,
                tilt_band_hz=(tilt_lo, tilt_hi), span_source=span_source,
                assessment=assessment)
        except Exception as e:
            return SpectralMetrics(assessment=f"Error in spectral analysis: {e}")

    # --------------------------- PAUSE DETECTION -------------------------------
    def compute_rms_envelope(self, y, sr, frame_length=2048, hop_length=512):
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        return rms, times

    def estimate_noise_floor(self, rms):
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        noise_percentile = np.percentile(rms_db, 10)
        quiet_threshold = np.percentile(rms_db, 30)
        quiet_frames = rms_db[rms_db < quiet_threshold]
        noise_mode = np.median(quiet_frames) if len(quiet_frames) > 10 else noise_percentile
        return max(noise_percentile, noise_mode)

    def get_silence_threshold(self, rms, task=None):
        """
        Silence line in dB relative to the file peak.

        FIXED in v8 for sustained phonation. The connected-speech rule
        (noise_floor + 6 dB, clamped at -20 dB) ends up only a few dB below a
        continuous vowel, because the "noise floor" is estimated from vowel
        frames. Vibrato, tremor and natural taper then cross it and are counted
        as pauses. For a sustained vowel the line is instead placed a fixed
        15 dB below the phonation level: a genuine break in phonation drops far
        more than that, a tremor cycle far less.
        """
        task = task or self._task
        if self.silence_threshold_db is not None:
            return self.silence_threshold_db
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        if task == TASK_SUSTAINED:
            loud = rms_db[rms_db >= np.percentile(rms_db, 60)]
            level = float(np.median(loud)) if loud.size else float(np.median(rms_db))
            return float(max(min(level - 15.0, -12.0), -60.0))
        noise_floor = self.estimate_noise_floor(rms)
        threshold = noise_floor + self.noise_margin_db
        return max(min(threshold, -20), -60)

    def find_speech_boundaries(self, rms, times, silence_threshold):
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        boundary_margin_db = 5.0
        threshold = silence_threshold + boundary_margin_db
        speech_mask = rms_db > threshold
        if not np.any(speech_mask):
            return times[0], times[-1]
        speech_indices = np.where(speech_mask)[0]
        start_idx = max(0, speech_indices[0] - 5)
        end_idx = min(len(times) - 1, speech_indices[-1] + 5)
        return times[start_idx], times[end_idx]

    def detect_pauses(self, rms, times, speech_start, speech_end, silence_threshold):
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        speech_mask = (times >= speech_start) & (times <= speech_end)
        speech_rms_db = rms_db[speech_mask]; speech_times = times[speech_mask]
        if len(speech_times) < 2:
            return PauseMetrics(), np.array([])
        silence_mask = speech_rms_db < silence_threshold
        pauses = []; in_pause = False; pause_start = 0
        for i, (is_silent, t) in enumerate(zip(silence_mask, speech_times)):
            if is_silent and not in_pause:
                in_pause = True; pause_start = t
            elif not is_silent and in_pause:
                in_pause = False; pd = t - pause_start
                if pd >= self.min_pause_duration:
                    pauses.append(pd)
        if in_pause:
            pd = speech_times[-1] - pause_start
            if pd >= self.min_pause_duration:
                pauses.append(pd)
        speech_duration = speech_end - speech_start
        total_pause = sum(pauses); actual_speech = speech_duration - total_pause
        metrics = PauseMetrics(
            count=len(pauses), total_duration=total_pause,
            avg_duration=np.mean(pauses) if pauses else 0.0,
            std_duration=np.std(pauses) if len(pauses) > 1 else 0.0,
            median_duration=np.median(pauses) if pauses else 0.0,
            min_duration=np.min(pauses) if pauses else 0.0,
            max_duration=np.max(pauses) if pauses else 0.0,
            ratio_to_speech=total_pause / actual_speech if actual_speech > 0 else 0.0,
            pauses_per_minute=(len(pauses) / speech_duration * 60) if speech_duration > 0 else 0.0,
            durations=pauses)
        return metrics, ~silence_mask

    # --------------------------- FILLER DETECTION ------------------------------
    def detect_fillers(self, y, sr, speech_start, speech_end, rms, rms_times,
                       silence_threshold, sound=None, pitch=None) -> FillerMetrics:
        if pitch is None or sound is None:
            return FillerMetrics()
        filler_min = max(self.min_filler_duration, 0.35)
        try:
            n_frames = call(pitch, "Get number of frames")
            dt = call(pitch, "Get time step") or 0.01
        except Exception:
            return FillerMetrics()
        times = []; f0 = []
        for i in range(1, n_frames + 1):
            t = call(pitch, "Get time from frame number", i)
            if speech_start <= t <= speech_end:
                v = call(pitch, "Get value in frame", i, "Hertz")
                times.append(t); f0.append(v if (v and v > 0 and not np.isnan(v)) else np.nan)
        if len(times) < 5:
            return FillerMetrics()
        times = np.asarray(times); f0 = np.asarray(f0); voiced = ~np.isnan(f0)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        def silence_before(t0):
            m = (rms_times >= t0 - 0.18) & (rms_times < t0 - 0.02)
            return np.any(m) and np.mean(rms_db[m]) < silence_threshold
        def silence_after(t1):
            m = (rms_times > t1 + 0.02) & (rms_times <= t1 + 0.18)
            return np.any(m) and np.mean(rms_db[m]) < silence_threshold
        fillers = []; i = 0; N = len(times)
        while i < N:
            if voiced[i]:
                j = i
                while j + 1 < N and voiced[j + 1]:
                    j += 1
                seg_t = times[i:j + 1]; seg_f = f0[i:j + 1]
                dur = seg_t[-1] - seg_t[0]
                if filler_min <= dur <= self.max_filler_duration and seg_f.size >= 6:
                    st = 12.0 * np.log2(seg_f / np.median(seg_f))
                    f0_st_std = float(np.std(st))
                    bounded = silence_before(seg_t[0]) or silence_after(seg_t[-1])
                    low_range = np.median(seg_f) < self._adaptive_median * 1.15
                    if f0_st_std < 0.5 and bounded and low_range:
                        fillers.append(dur)
                i = j + 1
            else:
                i += 1
        speech_duration = speech_end - speech_start
        total_filler = sum(fillers); actual_speech = speech_duration - total_filler
        return FillerMetrics(
            count=len(fillers), total_duration=total_filler,
            avg_duration=float(np.mean(fillers)) if fillers else 0.0,
            std_duration=float(np.std(fillers)) if len(fillers) > 1 else 0.0,
            ratio_to_speech=total_filler / actual_speech if actual_speech > 0 else 0.0,
            fillers_per_minute=(len(fillers) / speech_duration * 60) if speech_duration > 0 else 0.0,
            durations=fillers)

    # ------------------------------- RHYTHM ------------------------------------
    def analyze_rhythm(self, y, sr, speech_start, speech_end,
                       pause_metrics, pitch_metrics, syllable_count=None) -> RhythmMetrics:
        """
        Rhythm/temporal metrics. If syllable_count is provided (from the known
        passage), speech_rate/articulation_rate are EXACT; otherwise the
        un-clamped estimate is used.
        """
        start_sample = int(speech_start * sr); end_sample = int(speech_end * sr)
        y_speech = y[start_sample:end_sample]
        speech_duration = speech_end - speech_start
        articulation_time = speech_duration - pause_metrics.total_duration
        phonation_time = speech_duration * (pitch_metrics.voiced_frames_percent / 100)
        if len(y_speech) < sr * 0.5:
            return RhythmMetrics(total_duration_sec=speech_duration,
                                 speech_duration_sec=speech_duration,
                                 articulation_time_sec=articulation_time,
                                 phonation_time_sec=phonation_time,
                                 assessment="Insufficient audio for rhythm analysis")
        onset_env = librosa.onset.onset_strength(y=y_speech, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        if syllable_count is None:
            syllable_count = estimate_syllables_unclamped(y, sr, speech_start, speech_end)
        speech_rate = syllable_count / speech_duration if speech_duration > 0 else 0
        articulation_rate = syllable_count / articulation_time if articulation_time > 0 else 0
        if len(onset_times) > 3:
            intervals = np.diff(onset_times)
            intervals_clean = intervals[(intervals > 0.05) & (intervals < 1.0)]
            if len(intervals_clean) > 2:
                npvi_values = []
                for i in range(len(intervals_clean) - 1):
                    d1, d2 = intervals_clean[i], intervals_clean[i + 1]
                    if d1 + d2 > 0:
                        npvi_values.append(abs(d1 - d2) / ((d1 + d2) / 2))
                npvi = 100 * np.mean(npvi_values) if npvi_values else None
                rpvi = np.mean(np.abs(np.diff(intervals_clean)))
                varco = (np.std(intervals_clean) / np.mean(intervals_clean) * 100)
                delta = np.std(intervals_clean)
            else:
                npvi, rpvi, varco, delta = None, None, 0, 0
        else:
            npvi, rpvi, varco, delta = None, None, 0, 0
        if 3.5 <= speech_rate <= 5.5:
            rate_assessment = "normal speaking rate"
        elif speech_rate < 3.5:
            rate_assessment = "slow speaking rate"
        else:
            rate_assessment = "fast speaking rate"
        if npvi is not None:
            if 40 <= npvi <= 60:
                rhythm_assessment = "typical rhythm pattern"
            elif npvi < 40:
                rhythm_assessment = "syllable-timed rhythm"
            else:
                rhythm_assessment = "stress-timed rhythm"
        else:
            rhythm_assessment = "insufficient data for rhythm assessment"
        assessment = f"{rate_assessment.capitalize()}; {rhythm_assessment}"
        return RhythmMetrics(
            total_duration_sec=speech_duration, speech_duration_sec=speech_duration,
            articulation_time_sec=articulation_time, phonation_time_sec=phonation_time,
            speech_rate_syllables_per_sec=speech_rate,
            articulation_rate_syllables_per_sec=articulation_rate,
            npvi_v=npvi, rpvi_c=rpvi, percent_v=pitch_metrics.voiced_frames_percent,
            varco_v=varco, delta_v=delta, delta_c=delta * 0.8,
            assessment=assessment, estimated_syllable_count=int(syllable_count))

    # ------------------------ READING METRICS (new) ----------------------------
    def analyze_reading_metrics(self, sound, y, sr, file_path, speech_start,
                                speech_end, articulation_time_sec, voiced_intervals,
                                formant_metrics) -> Tuple[ReadingMetrics, float, float]:
        floor = self._adaptive_floor; ceil = self._adaptive_ceiling
        (count, source, passage, speech_rate, artic_rate, speak_frac,
         syll_est, syll_agree) = \
            compute_rate_from_passage(file_path, y, sr, speech_start, speech_end,
                                      articulation_time_sec)
        ems_ratio, ems_peak, ems_syll_slow = envelope_modulation_spectrum(
            y, sr, speech_start, speech_end)
        f1_vals = getattr(formant_metrics, "_f1_values", None)
        f2_vals = getattr(formant_metrics, "_f2_values", None)
        disp_logarea, disp_spread = formant_cloud_dispersion(f1_vals or [], f2_vals or [])
        decay, decay_per_s, _ = intensity_decay_over(
            sound, [(speech_start, speech_end)], floor)
        hnr_med = hnr_median_perinterval(sound, voiced_intervals, floor)
        cpps_val, cpps_src = cpps_hardened(
            sound, y, sr, speech_start, speech_end, floor, ceil,
            intervals=voiced_intervals, min_interval_dur=self.cpps_min_interval_dur)
        notes = []
        if not np.isnan(speak_frac) and speak_frac < 0.65:
            notes.append("much total-time inflation from pausing")
        if not np.isnan(ems_ratio) and ems_ratio < 0.25:
            notes.append("weak syllabic (3-8 Hz) envelope modulation")
        if not np.isnan(decay) and decay < -6.0:
            notes.append("marked loudness decay across passage")
        assessment = "; ".join(notes) if notes else "no salient reading-level deviation flagged"
        rm = ReadingMetrics(
            syllable_count_used=int(count), syllable_source=source, passage_name=passage,
            speaking_time_fraction=speak_frac, ems_3_8hz_ratio=ems_ratio,
            ems_peak_freq_hz=ems_peak, ems_4_to_lowband_ratio=ems_syll_slow,
            formant_dispersion_logarea=disp_logarea, f1f2_cloud_spread=disp_spread,
            intensity_decay_db=decay, intensity_decay_db_per_s=decay_per_s,
            hnr_median_perinterval_db=hnr_med,
            cpps_db=cpps_val, cpps_source=cpps_src,
            syllable_count_estimated=int(syll_est),
            syllable_count_agreement=float(syll_agree), assessment=assessment)
        # v16: warn when the hand-entered passage constant and the audio disagree.
        # This is the only automatic check available on KNOWN_PASSAGE_SYLLABLES.
        if source == "known_passage" and np.isfinite(syll_agree) and not (
                0.70 <= syll_agree <= 1.40):
            print(f"    ! SYLLABLE CONSTANT SUSPECT for passage '{passage}': the table "
                  f"says {count} syllables, the envelope estimate is ~{syll_est} "
                  f"(ratio {syll_agree:.2f}). Verify KNOWN_PASSAGE_SYLLABLES against "
                  f"YOUR wording of this passage - a wrong constant rescales "
                  f"speech_rate and articulation_rate for every file of it. "
                  f"speaking_time_fraction is unaffected.")
        return rm, speech_rate, artic_rate

    # ----------------------------- FLUENCY -------------------------------------
    def compute_fluency(self, pause_metrics, filler_metrics, intensity_metrics,
                        pitch_metrics, voice_quality_metrics, rhythm_metrics,
                        task=None, enabled: Optional[bool] = None) -> FluencyMetrics:
        """
        Composite fluency score. NOT COMPUTED for sustained vowels: every input
        (pause behaviour, filler rate, speech rate, rhythm regularity) is either
        undefined or artifactual in sustained phonation, so the resulting number
        looked like a clinical severity rating while carrying no information.
        """
        task = task or self._task
        # ---- v20: OFF BY DEFAULT ----------------------------------------
        # overall_fluency is an unvalidated weighted sum of eight sub-scores
        # with hand-chosen coefficients, exported next to real measurements and
        # labelled with a "clinical_severity" string. Nothing in this script
        # validates those weights against any outcome, and a number that looks
        # like a severity rating gets read as one. It is retained for
        # continuity but must be switched on deliberately.
        enabled = self.enable_fluency_index if enabled is None else bool(enabled)
        if not enabled:
            return FluencyMetrics(
                intensity_stability=float("nan"), pitch_stability=float("nan"),
                rhythm_regularity=float("nan"), voice_quality_score=float("nan"),
                articulation_score=float("nan"), pause_penalty=float("nan"),
                filler_penalty=float("nan"), voice_break_penalty=float("nan"),
                overall_fluency=float("nan"), clinical_severity="disabled",
                assessment="Composite fluency index disabled (unvalidated); "
                           "use the objective acoustic and temporal measures.")
        if task == TASK_SUSTAINED:
            return FluencyMetrics(
                intensity_stability=float("nan"), pitch_stability=float("nan"),
                rhythm_regularity=float("nan"), voice_quality_score=float("nan"),
                articulation_score=float("nan"), overall_fluency=float("nan"),
                clinical_severity="n/a (sustained vowel)",
                assessment="Fluency score not applicable to sustained phonation; "
                           "use the sustained-vowel metrics instead.")

        def neutral_if_bad(val, fallback=50.0):
            return fallback if (val is None or (isinstance(val, float) and np.isnan(val))) else val
        n_std = getattr(intensity_metrics, "nucleus_std_db", 0.0)
        if not (n_std and n_std > 0 and not np.isnan(n_std)):
            n_std = intensity_metrics.active_std_db
        if n_std and n_std > 0 and not np.isnan(n_std):
            intensity_stability = float(np.clip(100.0 - max(0.0, n_std - 2.0) * 12.0, 0.0, 100.0))
        else:
            intensity_stability = 60.0
        cv = neutral_if_bad(pitch_metrics.coefficient_of_variation, 0.15)
        if cv < 0.05:
            pitch_stability = 60.0
        elif cv <= 0.25:
            pitch_stability = 100.0
        elif cv <= 0.35:
            pitch_stability = float(100.0 - (cv - 0.25) * 300.0)
        else:
            pitch_stability = float(np.clip(70.0 - (cv - 0.35) * 200.0, 30.0, 70.0))
        if rhythm_metrics.npvi_v is not None and not np.isnan(rhythm_metrics.npvi_v):
            npvi = rhythm_metrics.npvi_v
            if 45 <= npvi <= 65:
                rhythm_regularity = 100.0
            else:
                rhythm_regularity = float(np.clip(100.0 - abs(npvi - 55.0) * 2.0, 40.0, 100.0))
        else:
            rhythm_regularity = 65.0
        vq = voice_quality_metrics
        if not bool(getattr(vq, "measured", False)):
            voice_quality_score = 50.0
        elif not bool(getattr(vq, "reliable", True)):
            voice_quality_score = 65.0
        else:
            jl = neutral_if_bad(vq.jitter_local_percent, 1.5)
            sh = neutral_if_bad(vq.shimmer_local_percent, 6.5)
            hnr = neutral_if_bad(vq.hnr_db, 16.0)
            brk = neutral_if_bad(vq.voice_breaks_degree_percent, 0.0)
            jitter_sub = float(np.clip(100.0 * (5.5 - jl) / (5.5 - 1.8), 0.0, 100.0))
            shimmer_sub = float(np.clip(100.0 * (15.0 - sh) / (15.0 - 7.5), 0.0, 100.0))
            hnr_sub = float(np.clip(100.0 * (hnr - 6.0) / (15.0 - 6.0), 0.0, 100.0))
            break_pen = float(min(20.0, max(0.0, brk - 3.0) * 2.0))
            voice_quality_score = float(np.clip(
                0.35 * jitter_sub + 0.30 * shimmer_sub + 0.35 * hnr_sub - break_pen, 0.0, 100.0))
        sr_ = neutral_if_bad(rhythm_metrics.speech_rate_syllables_per_sec, 4.0)
        if 3.5 <= sr_ <= 5.5:
            articulation_score = 100.0
        elif sr_ < 3.5:
            articulation_score = float(np.clip(100.0 - (3.5 - sr_) * 30.0, 20.0, 100.0))
        else:
            articulation_score = float(np.clip(100.0 - (sr_ - 5.5) * 20.0, 40.0, 100.0))
        pause_penalty = float(min(25.0, max(0.0, pause_metrics.ratio_to_speech - 0.20) * 100.0))
        filler_penalty = float(min(15.0, filler_metrics.ratio_to_speech * 75.0))
        voice_break_penalty = 0.0
        components = {"intensity": intensity_stability, "pitch": pitch_stability,
                      "rhythm": rhythm_regularity, "vq": voice_quality_score,
                      "artic": articulation_score}
        weights = {"intensity": 0.15, "pitch": 0.15, "rhythm": 0.15, "vq": 0.30, "artic": 0.25}
        weighted_mean = sum(components[k] * weights[k] for k in components)
        worst = min(components.values())
        blended = 0.70 * weighted_mean + 0.30 * worst
        overall_fluency = max(0.0, min(100.0, blended - pause_penalty - filler_penalty))
        if overall_fluency >= 80:
            clinical_severity = "Normal"
        elif overall_fluency >= 65:
            clinical_severity = "Mild impairment"
        elif overall_fluency >= 50:
            clinical_severity = "Moderate impairment"
        elif overall_fluency >= 35:
            clinical_severity = "Moderate-severe impairment"
        else:
            clinical_severity = "Severe impairment"
        recommendations = []; issues = []
        if pause_metrics.ratio_to_speech > 0.25:
            issues.append("frequent pauses"); recommendations.append("Practice continuous speech exercises")
        if filler_metrics.ratio_to_speech > 0.15:
            issues.append("many filler sounds"); recommendations.append("Work on word retrieval and planning")
        if intensity_stability < 60:
            issues.append("unstable voice intensity"); recommendations.append("Focus on breath support and projection")
        if pitch_stability < 60:
            issues.append("unstable pitch"); recommendations.append("Practice sustained vowel exercises")
        if voice_quality_score < 70:
            issues.append("voice quality concerns"); recommendations.append("Consider voice therapy evaluation")
        if rhythm_regularity < 50:
            issues.append("irregular speech rhythm"); recommendations.append("Practice paced reading exercises")
        if not issues:
            assessment = ("Excellent fluency - natural, smooth speech" if overall_fluency >= 75
                          else "Good fluency - generally smooth with minor variations")
        else:
            assessment = f"Fluency issues detected: {', '.join(issues)}"
        return FluencyMetrics(
            intensity_stability=intensity_stability, pitch_stability=pitch_stability,
            rhythm_regularity=rhythm_regularity, voice_quality_score=voice_quality_score,
            articulation_score=articulation_score, pause_penalty=pause_penalty,
            filler_penalty=filler_penalty, voice_break_penalty=voice_break_penalty,
            overall_fluency=overall_fluency, clinical_severity=clinical_severity,
            assessment=assessment, recommendations=recommendations)

    # ------------------------------ TASK ROUTING -------------------------------
    def resolve_task(self, file_path: str, sound, y, sr) -> Tuple[str, str]:
        """
        Decide whether this file is a sustained vowel or connected speech.
        Order: explicit setting > filename/folder tokens > acoustic classifier.
        """
        if self.task in (TASK_SUSTAINED, TASK_READING):
            return self.task, "forced"
        t = task_from_filename(file_path)
        if t is not None:
            return t, "filename"
        voiced_fraction = float("nan")
        try:
            wide = call(sound, "To Pitch", 0.0, 50.0, 600.0)
            n_all = call(wide, "Get number of frames")
            n_voiced = call(wide, "Count voiced frames")
            if n_all:
                voiced_fraction = float(n_voiced) / float(n_all)
        except Exception:
            pass
        ems_ratio = float("nan")
        try:
            ems_ratio, _, _ = envelope_modulation_spectrum(y, sr, 0.0, len(y) / sr)
        except Exception:
            pass
        centroid_cv = float("nan")
        try:
            c = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            c = c[np.isfinite(c) & (c > 0)]
            if c.size > 10:
                centroid_cv = float(np.std(c) / np.mean(c))
        except Exception:
            pass
        t, reason = task_from_acoustics(voiced_fraction, ems_ratio, centroid_cv)
        return t, f"acoustic ({reason})"

    # --------------------- SUSTAINED-VOWEL ANALYSIS (new in v8) ----------------
    def analyze_sustained_vowel(self, sound, y, sr, pitch, speech_start, speech_end,
                                recording_quality, rms, rms_times,
                                silence_threshold_db=-25.0,
                                given_spans: Optional[List[Tuple[float, float]]] = None):
        """
        Token-wise sustained-phonation analysis.

        The file is split into individual /a/ productions (a 70 s "sustained
        vowel" file is several trials plus breaths, not one token), MPT is taken
        per token, and jitter/shimmer/HNR/CPPS are measured ONLY inside the
        steady-state mid-window of each token. Returns
        (SustainedVowelMetrics, VoiceQualityMetrics_for_the_analysis_window).

        given_spans (v9): token boundaries supplied by the user (see
        read_boundaries_file). When present they are used AS IS - no energy
        segmentation, no splice detection - because if you cut the tokens
        yourself there is nothing to infer.
        """
        empty = SustainedVowelMetrics(assessment="No usable sustained-vowel token found")
        if pitch is None:
            return empty, None
        try:
            n_frames = call(pitch, "Get number of frames")
            times, f0 = [], []
            for i in range(1, n_frames + 1):
                t = call(pitch, "Get time from frame number", i)
                if speech_start - 0.05 <= t <= speech_end + 0.05:
                    v = call(pitch, "Get value in frame", i, "Hertz")
                    times.append(t)
                    f0.append(v if (v is not None and v > 0 and not np.isnan(v)) else np.nan)
            times = np.asarray(times, dtype=float)
            f0 = np.asarray(f0, dtype=float)
            if times.size < 10:
                return empty, None
            f0, _ = repair_octave_jumps(f0)
            voiced = np.isfinite(f0) & (f0 > 0)
            # ---- one file = one production (v10) -----------------------------
            # ---- v20: one file = one sustained phonation --------------------
            # Never infer extra productions and never split at an inferred edit
            # point. An internal dropout stays inside this one production and is
            # reported by the coverage ledger, the break ledger and the
            # unbroken-phonation metric instead of becoming a new token.
            if self.single_token_per_file:
                return self._sustained_from_tokens(
                    sound, y, sr, pitch, times, f0, voiced,
                    [(float(speech_start), float(speech_end))], [],
                    recording_quality, rms, rms_times, "single_production_per_file")
            # ---- boundaries supplied by the user win over any inference ------
            boundaries_source = "inferred"
            if given_spans:
                tokens = [(max(float(a), float(speech_start)),
                           min(float(b), float(speech_end)))
                          for a, b in given_spans
                          if min(b, speech_end) - max(a, speech_start) >= 0.3]
                if tokens:
                    boundaries_source = "file"
                    return self._sustained_from_tokens(
                        sound, y, sr, pitch, times, f0, voiced, tokens, [],
                        recording_quality, rms, rms_times, boundaries_source)
            # Energy-gated segmentation (see segment_phonation_tokens). The
            # voicing-only path is kept only as a fallback.
            tokens = []
            if rms is not None and rms_times is not None:
                try:
                    rms_db_full = librosa.amplitude_to_db(rms, ref=np.max)
                    tokens = segment_phonation_tokens(
                        rms_times, rms_db_full, silence_threshold_db,
                        times, voiced, speech_start, speech_end,
                        min_token_s=self.min_vowel_token_dur,
                        bridge_gap_s=self.token_bridge_gap_s,
                        min_voiced_fraction=self.token_min_voiced_fraction)
                except Exception:
                    tokens = []
            if not tokens:
                tokens = segment_voiced_tokens(
                    times, voiced, min_token_s=self.min_vowel_token_dur,
                    max_bridge_gap_s=self.token_bridge_gap_s)
            if not tokens:
                tokens = [(float(speech_start), float(speech_end))]

            # ---- splice-aware refinement -------------------------------------
            # Concatenated/trimmed recordings hide production boundaries, so
            # look for edit points inside each energy run and split there too.
            splices = []
            # v20: with one production per file an "edit point" found inside the
            # take is a false positive that shortens MPT, so splitting is
            # suppressed when single_token_per_file is set.
            if self.split_at_splices and not self.single_token_per_file:
                try:
                    rms_db_sp = librosa.amplitude_to_db(rms, ref=np.max) \
                        if rms is not None else np.array([])
                    rms_t_sp = rms_times if rms_times is not None else np.array([])
                    splices = detect_splices(
                        times, f0, rms_t_sp, rms_db_sp, tokens,
                        f0_step_st=self.splice_f0_step_st,
                        silence_threshold_db=silence_threshold_db)
                    if splices:
                        tokens = split_spans_at(
                            tokens, splices, min_piece_s=self.min_vowel_token_dur)
                except Exception:
                    splices = []
            return self._sustained_from_tokens(
                sound, y, sr, pitch, times, f0, voiced, tokens, splices,
                recording_quality, rms, rms_times, boundaries_source)
        except Exception as e:
            return SustainedVowelMetrics(
                assessment=f"Error in sustained-vowel analysis: {e}"), None

    def _sustained_from_tokens(self, sound, y, sr, pitch, times, f0, voiced,
                               tokens, splices, recording_quality,
                               rms, rms_times, boundaries_source="inferred"):
        """
        Window tiling, validity screening and robust aggregation for a given set
        of token spans. Split out in v9 so that user-supplied boundaries and
        inferred boundaries follow exactly the same measurement path.
        """
        try:
            durs = [e - s for s, e in tokens]
            if not durs:
                return SustainedVowelMetrics(
                    assessment="No usable sustained-vowel token found"), None
            rms_db = (librosa.amplitude_to_db(rms, ref=np.max)
                      if rms is not None and np.size(rms) else None)
            # ---- v19: longest phonation with no break in the middle ---------
            # Measured per token on its own short-frame envelope, with voicing
            # required and detected splices treated as hard breaks. The old call
            # searched the whole span from the first token's start to the last
            # token's end, so it could - and on real multi-trial files did -
            # return more than mpt_longest_s.
            unbroken = longest_unbroken_phonation(
                y, sr, tokens, pitch_times=times, voiced=voiced, splices=splices,
                min_break_s=self.phonation_break_min_s,
                drop_db=self.phonation_break_drop_db,
                require_voicing=self.phonation_require_voicing,
                env_win_s=self.phonation_env_win_s,
                env_hop_s=self.phonation_env_hop_s)
            uninterrupted_s = float(unbroken["longest_s"])

            def _unbroken_kw() -> Dict[str, Any]:
                """The break ledger, attached on EVERY exit path including the
                failure ones - on a take that broke down repeatedly this is the
                finding, so it must not be the thing that goes missing."""
                return dict(
                    mpt_longest_uninterrupted_s=uninterrupted_s,
                    unbroken_start_s=float(unbroken["start_s"]),
                    unbroken_end_s=float(unbroken["end_s"]),
                    unbroken_token_index=int(unbroken["token_index"]),
                    unbroken_source=str(unbroken["source"]),
                    n_phonation_breaks=int(unbroken["n_breaks_total"]),
                    n_breaks_in_best_token=int(unbroken["n_breaks_in_best"]),
                    unbroken_energy_only_s=float(unbroken["longest_energy_only_s"]),
                    break_threshold_db=float(unbroken["threshold_db"]))

            # ---- MULTI-WINDOW ROBUST ESTIMATION ------------------------------
            # A single 3 s window is far too fragile: on one of these files,
            # moving the window changed jitter from 0.96% to 2.97% and HNR from
            # 11.1 to 6.9 dB. Instead, tile non-overlapping windows across ALL
            # tokens, discard the ones that are not valid steady-state phonation
            # (see window_validity), and report the MEDIAN across the survivors
            # with an IQR as a dispersion/quality indicator. Placement no longer
            # decides the result, and neither does an imperfect token boundary.
            file_median_f0 = float(np.median(f0[voiced])) if np.any(voiced) else float("nan")

            # ---- v18: WINDOW-FREE STEADINESS, computed for EVERY file --------
            # Same criteria as window_validity, asked frame by frame instead of
            # window by window, so it survives the exits below where no window
            # was measured at all. On those files it is the only thing that
            # distinguishes "the voice never held still" from "the voice held
            # still but the take was too short for three 2 s windows".
            frame_prof = steady_frame_profile(
                times, f0, tokens, ref_f0=file_median_f0,
                scale_s=self.vowel_window_max_s,
                max_dev_st=self.window_max_f0_deviation_st,
                max_step_st=self.window_max_internal_step_st,
                min_voiced_fraction=self.window_min_voiced_fraction,
                min_scale_s=self.vowel_window_min_s)

            # ---- TOKEN-LEVEL MEASURES, computed for EVERY file (v11) ---------
            # jitter/shimmer need quasi-periodicity; CPPS and HNR do not, and the
            # voiced fraction least of all. Files whose phonation is too unstable
            # for perturbation analysis used to come out as an all-NaN row and
            # vanished from every plot - which deleted exactly the observation
            # that mattered. Now they still carry a number.
            in_tok = np.zeros(times.shape, dtype=bool)
            for (a, b) in tokens:
                in_tok |= (times >= a) & (times <= b)
            voiced_fraction = (float(np.mean(voiced[in_tok]))
                               if int(np.count_nonzero(in_tok)) > 0 else float("nan"))
            cpps_token = float("nan")
            try:
                cpps_token, _ = cpps_hardened(
                    sound, y, sr, tokens[0][0], tokens[-1][1],
                    self._adaptive_floor, self._adaptive_ceiling,
                    intervals=list(tokens),
                    min_interval_dur=self.cpps_min_interval_dur)
            except Exception:
                pass
            hnr_token = float("nan")
            try:
                hnr_token = hnr_median_perinterval(sound, list(tokens),
                                                   self._adaptive_floor)
            except Exception:
                pass

            # v17: the PROBE now runs BEFORE classification. It used to run after,
            # which meant classify_signal_type() could not see it - so a file whose
            # low voiced_fraction was purely a THRESHOLD artifact (the probe's whole
            # purpose is to detect that) was still labelled "type 3, perturbation
            # UNDEFINED" on the strength of that voiced fraction alone, and the very
            # next console line then said "the THRESHOLD was the limit and the
            # periodicity is strong". Real example from a DBS dataset: voiced 0.54,
            # HNR 17.2 dB, CPPS 13.2 dB, 94% voiced at threshold 0.20 - declared
            # aperiodic, then measured a perfectly good jitter of 0.658%.
            probe: Dict[str, float] = {}
            if voiced_fraction < 0.70:
                # low voicing: check whether the tracker settings are responsible
                # before concluding anything about the voice
                try:
                    probe = voicing_probe(sound, self._adaptive_floor,
                                          self._adaptive_ceiling)
                except Exception:
                    probe = {}

            sig_type, sig_note = classify_signal_type(
                voiced_fraction, hnr_token, cpps_token,
                int(getattr(self, "_f0_subharm_lock", 0)),
                voiced_at_020=probe.get("voiced_at_0.20", float("nan")))

            # Overlapping candidates (hop = win/2) so a fully voiced stretch that
            # straddles a tile boundary is not lost; the survivors are then
            # reduced to a NON-OVERLAPPING set, so nothing is counted twice.
            grid_hop = (self.vowel_window_grid_hop_s
                        or max(0.5, self.vowel_window_max_s / 2.0))
            cands = tile_windows(tokens, win_s=self.vowel_window_max_s,
                                 hop_s=grid_hop,
                                 edge_trim_s=self.vowel_edge_trim_s,
                                 min_window_s=self.vowel_window_min_s)
            n_tiles = 0
            for (a, b) in tokens:
                usable_dur = (b - a) - 2 * self.vowel_edge_trim_s
                if usable_dur >= self.vowel_window_min_s:
                    n_tiles += max(1, int(usable_dur // self.vowel_window_max_s))
            reasons: Dict[str, int] = {}
            valid_wins = []
            for w in cands:
                ok, why = window_validity(
                    w, times, f0, file_median_f0, splices,
                    min_voiced_fraction=self.window_min_voiced_fraction,
                    max_f0_deviation_st=self.window_max_f0_deviation_st,
                    max_internal_step_st=self.window_max_internal_step_st)
                if ok:
                    valid_wins.append(w)
                else:
                    reasons[why] = reasons.get(why, 0) + 1
            # keep only non-overlapping survivors, earliest first
            # v17: the drops here used to be SILENT, so the printed ledger never
            # balanced - e.g. "6 valid of 11" with no rejections listed, inviting
            # the reader to think 5 windows vanished for unknown reasons. They were
            # valid but overlapping (candidates are tiled at hop = win/2). Record
            # them so n_windows_valid + rejections == n_windows_total exactly.
            if valid_wins:
                valid_wins.sort(key=lambda w: w[0])
                picked, last_end = [], -np.inf
                for w in valid_wins:
                    if w[0] >= last_end - 1e-9:
                        picked.append(w); last_end = w[1]
                n_overlap = len(valid_wins) - len(picked)
                if n_overlap:
                    reasons["overlap_dedup"] = reasons.get("overlap_dedup", 0) + n_overlap
                valid_wins = picked
            if len(valid_wins) > self.max_windows_measured:
                idx = np.linspace(0, len(valid_wins) - 1,
                                  self.max_windows_measured).astype(int)
                kept = [valid_wins[i] for i in sorted(set(idx))]
                n_cap = len(valid_wins) - len(kept)
                if n_cap:
                    reasons["over_cap"] = reasons.get("over_cap", 0) + n_cap
                valid_wins = kept
            if not valid_wins:
                return self._fill_coverage(SustainedVowelMetrics(
                    n_tokens=len(tokens), token_durations_s=[float(d) for d in durs],
                    token_spans_s=[(float(s), float(e)) for s, e in tokens],
                    n_splices_detected=len(splices),
                    splice_times_s=[float(c) for c in splices],
                    mpt_longest_s=float(max(durs)), mpt_mean_s=float(np.mean(durs)),
                    **_unbroken_kw(),
                    total_phonation_s=float(np.sum(durs)),
                    analysis_window_s=float(self.vowel_window_max_s),
                    n_windows_total=len(cands), n_windows_valid=0,
                    window_rejections="; ".join(f"{k}:{v}" for k, v in reasons.items()),
                    rej_low_voicing=int(reasons.get("low_voicing", 0)),
                    rej_f0_outlier=int(reasons.get("f0_outlier", 0)),
                    rej_f0_step=int(reasons.get("f0_step", 0)),
                    rej_splice=int(reasons.get("splice", 0)),
                    rej_other=int(reasons.get("no_data", 0) + reasons.get("unmeasurable", 0)),
                    measured_total_s=0.0, boundaries_source=boundaries_source,
                    voiced_fraction=voiced_fraction, window_yield=0.0,
                    cpps_token_db=cpps_token, hnr_token_db=hnr_token,
                    signal_type=sig_type, signal_type_note=sig_note,
                    voiced_at_045=probe.get('voiced_at_0.45', float('nan')),
                    voiced_at_020=probe.get('voiced_at_0.20', float('nan')),
                    voiced_low_floor=probe.get('voiced_low_floor', float('nan')),
                    usable=False, usable_reasons="no valid steady-state window",
                    measured=False,
                    assessment="No window passed the steady-state validity checks "
                               "(" + ", ".join(f"{k}={v}" for k, v in reasons.items())
                               + "); phonation may be creaky, unstable or heavily edited"),
                    tokens, [], frame_prof), None

            rdb = None
            try:
                rdb = librosa.amplitude_to_db(rms, ref=np.max) if rms is not None else None
            except Exception:
                rdb = None

            per_win = []
            for w in valid_wins:
                # v20: CPPS is computed FIRST and a window is discarded only when
                # nothing at all could be measured on it. Under the old order a
                # window whose jitter failed was dropped before CPPS was even
                # attempted - yet CPPS needs no pulse train, so a periodicity
                # failure says nothing about the trustworthiness of the cepstral
                # peak. That discarded the cepstral measures of exactly the
                # roughest voices, selecting on a correlate of the outcome.
                cp, cp_src = cpps_hardened(
                    sound, y, sr, w[0], w[1],
                    self._adaptive_floor, self._adaptive_ceiling,
                    intervals=[w], min_interval_dur=self.cpps_min_interval_dur)
                vq = self.analyze_voice_quality(
                    sound, pitch, w[0], w[1], recording_quality,
                    intervals_override=[w])
                if (not vq.measured) and not np.isfinite(cp):
                    reasons["unmeasurable"] = reasons.get("unmeasurable", 0) + 1
                    continue
                m = (times >= w[0]) & (times <= w[1]) & voiced
                f0w, tw = f0[m], times[m]
                if f0w.size >= 5:
                    st = hz_to_semitones(f0w, float(np.median(f0w)))
                    sd_st = float(np.std(st))
                    slope = float(np.polyfit(tw, st, 1)[0])
                else:
                    sd_st = slope = float("nan")
                tr_rate, tr_ext, tr_ratio = f0_tremor(tw, f0w)
                isd = float("nan")
                if rdb is not None and rms_times is not None:
                    mm = (rms_times >= w[0]) & (rms_times <= w[1])
                    if np.count_nonzero(mm) >= 5:
                        isd = float(np.std(rdb[mm]))
                per_win.append(dict(
                    span=w, token=next((i for i, (a, b) in enumerate(tokens)
                                        if a <= w[0] and w[1] <= b), -1),
                    jitter=vq.jitter_local_percent, shimmer=vq.shimmer_local_percent,
                    hnr=vq.hnr_db, cpps=cp, cpps_src=cp_src,
                    f0=float(np.median(f0w)) if f0w.size else float("nan"),
                    f0_sd_st=sd_st, drift=slope, tremor_rate=tr_rate,
                    tremor_ext=tr_ext, tremor_ratio=tr_ratio, int_sd=isd,
                    vq=vq))
            if not per_win:
                # v18: this exit used to drop the rejection tally and the splice
                # times, so the one file that most needed the audit trail was the
                # only one without it.
                return self._fill_coverage(SustainedVowelMetrics(
                    n_tokens=len(tokens), token_durations_s=[float(d) for d in durs],
                    token_spans_s=[(float(s), float(e)) for s, e in tokens],
                    n_splices_detected=len(splices),
                    splice_times_s=[float(c) for c in splices],
                    mpt_longest_s=float(max(durs)), mpt_mean_s=float(np.mean(durs)),
                    **_unbroken_kw(),
                    total_phonation_s=float(np.sum(durs)),
                    analysis_window_s=float(self.vowel_window_max_s),
                    n_windows_total=len(cands), n_windows_valid=0,
                    window_rejections="; ".join(f"{k}:{v}" for k, v in reasons.items()),
                    rej_low_voicing=int(reasons.get("low_voicing", 0)),
                    rej_f0_outlier=int(reasons.get("f0_outlier", 0)),
                    rej_f0_step=int(reasons.get("f0_step", 0)),
                    rej_splice=int(reasons.get("splice", 0)),
                    rej_other=int(reasons.get("no_data", 0)
                                  + reasons.get("unmeasurable", 0)),
                    measured_total_s=0.0,
                    boundaries_source=boundaries_source, measured=False, usable=False,
                    voiced_fraction=voiced_fraction, window_yield=0.0,
                    cpps_token_db=cpps_token, hnr_token_db=hnr_token,
                    signal_type=sig_type, signal_type_note=sig_note,
                    voiced_at_045=probe.get('voiced_at_0.45', float('nan')),
                    voiced_at_020=probe.get('voiced_at_0.20', float('nan')),
                    voiced_low_floor=probe.get('voiced_low_floor', float('nan')),
                    usable_reasons="no window could be measured by Praat",
                    assessment="No analysis window could be measured "
                               "(Praat returned no perturbation values)"),
                    tokens, [], frame_prof), None

            jitter_med, jitter_iqr = _median_iqr([d["jitter"] for d in per_win])
            shimmer_med, shimmer_iqr = _median_iqr([d["shimmer"] for d in per_win])
            hnr_med, hnr_iqr = _median_iqr([d["hnr"] for d in per_win])
            cpps_val, cpps_iqr = _median_iqr([d["cpps"] for d in per_win])
            cpps_src = per_win[0]["cpps_src"]
            f0_mean, _ = _median_iqr([d["f0"] for d in per_win])
            f0_sd_st, _ = _median_iqr([d["f0_sd_st"] for d in per_win])
            slope, _ = _median_iqr([d["drift"] for d in per_win])
            tremor_rate, _ = _median_iqr([d["tremor_rate"] for d in per_win])
            tremor_extent, _ = _median_iqr([d["tremor_ext"] for d in per_win])
            tremor_ratio, _ = _median_iqr([d["tremor_ratio"] for d in per_win])
            int_sd, _ = _median_iqr([d["int_sd"] for d in per_win])
            # v9: analysis_window_start_s used to report per_win[len//2], a
            # window with no relation to the reported medians. Report the first
            # measured window and, more usefully, how much signal is behind the
            # medians in total.
            main_win = per_win[0]["span"]
            measured_total_s = float(sum(d["span"][1] - d["span"][0] for d in per_win))
            measured_spans = [tuple(d["span"]) for d in per_win]

            # ---- v19: THE WHOLE F0 DISTRIBUTION OVER THE MEASURED WINDOWS ---
            # Taken from the same frames as everything else, so the exported
            # pitch block is internally consistent on a sustained file. Before
            # this, only pitch_mean_hz was swapped for the window value while
            # the SD, median, range and CV stayed whole-span: a reader who
            # checked pitch_cv against pitch_std_hz / pitch_mean_hz got a
            # mismatch, and the range spanned every token plus the tracker's
            # excursions at the token edges.
            f0_win_mask = np.zeros(times.shape, dtype=bool)
            for _a, _b in measured_spans:
                f0_win_mask |= (times >= _a) & (times <= _b)
            f0_win = f0[f0_win_mask & voiced]
            f0_win = f0_win[np.isfinite(f0_win) & (f0_win > 0)]
            if f0_win.size >= 3:
                w_median = float(np.median(f0_win))
                w_std = float(np.std(f0_win))
                w_min = float(np.min(f0_win)); w_max = float(np.max(f0_win))
                w_cv = float(w_std / w_median) if w_median > 0 else float("nan")
                w_range_st = (float(12.0 * np.log2(w_max / w_min))
                              if w_min > 0 else float("nan"))
            else:
                w_median = w_std = w_min = w_max = w_cv = w_range_st = float("nan")

            # ---- FULL VOICE-REPORT PANEL, aggregated over windows (v15) ------
            # These sub-measures used to be blanked out for sustained files
            # because they carried the value of ONE window. The right answer is
            # not to drop them but to aggregate them the same way jitter,
            # shimmer and HNR already are: median across the valid windows
            # (counts are summed, since they scale with measured duration).
            def _vq_med(attr):
                return _median_iqr([getattr(d["vq"], attr, np.nan)
                                    for d in per_win])[0]

            def _vq_iqr(attr):
                return _median_iqr([getattr(d["vq"], attr, np.nan)
                                    for d in per_win])[1]

            def _vq_sum(attr):
                vals = [getattr(d["vq"], attr, np.nan) for d in per_win]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                return float(np.sum(vals)) if vals else float("nan")

            panel_vq = VoiceQualityMetrics(
                jitter_local_percent=jitter_med,
                jitter_local_abs_sec=_vq_med("jitter_local_abs_sec"),
                jitter_rap_percent=_vq_med("jitter_rap_percent"),
                jitter_ppq5_percent=_vq_med("jitter_ppq5_percent"),
                jitter_ddp_percent=_vq_med("jitter_ddp_percent"),
                shimmer_local_percent=shimmer_med,
                shimmer_local_db=_vq_med("shimmer_local_db"),
                shimmer_apq3_percent=_vq_med("shimmer_apq3_percent"),
                shimmer_apq5_percent=_vq_med("shimmer_apq5_percent"),
                shimmer_apq11_percent=_vq_med("shimmer_apq11_percent"),
                shimmer_dda_percent=_vq_med("shimmer_dda_percent"),
                hnr_db=hnr_med, nhr=_vq_med("nhr"),
                mean_autocorrelation=_vq_med("mean_autocorrelation"),
                voice_breaks_count=int(_vq_sum("voice_breaks_count") or 0),
                voice_breaks_degree_percent=_vq_med("voice_breaks_degree_percent"),
                num_pulses=int(_vq_sum("num_pulses") or 0),
                num_periods=int(_vq_sum("num_periods") or 0),
                fraction_unvoiced_percent=0.0,
                # v20: consequence of keeping a window whose CPPS survived but
                # whose perturbation did not. The panel must not claim
                # measured=True when every cycle-based median came back NaN.
                measured=bool(np.isfinite(jitter_med) or np.isfinite(shimmer_med)
                              or np.isfinite(hnr_med)),
                reliable=(recording_quality.vq_reliable
                          if recording_quality is not None else True),
                assessment=(f"median of {len(per_win)} steady-state window(s) of "
                            f"{self.vowel_window_max_s:.1f} s"),
                pathology_indicators=list(per_win[0]["vq"].pathology_indicators))

            # across-token reproducibility, from per-token medians of the windows
            # v17c FAKE-ZERO FIX: this returned 0.0 whenever fewer than two tokens
            # contributed, so a file containing ONE token (the normal case when
            # each repetition is its own file) reported "across-token SD = 0.0" -
            # which reads as perfect token-to-token reproducibility when the truth
            # is that there was nothing to compare. Confirmed on a real 20-file
            # batch: every file had sv_n_tokens=1 and all three columns read 0.0.
            # With one token the quantity does not exist, so it is NaN.
            def _by_token(key):
                vals = []
                for ti in sorted({d["token"] for d in per_win if d["token"] >= 0}):
                    v = [d[key] for d in per_win if d["token"] == ti]
                    mm, _ = _median_iqr(v)
                    if np.isfinite(mm):
                        vals.append(mm)
                return float(np.std(vals)) if len(vals) > 1 else float("nan")
            jl_sd, sh_sd, hn_sd = (_by_token("jitter"), _by_token("shimmer"),
                                   _by_token("hnr"))

            notes = []
            if splices:
                notes.append(f"{len(splices)} edit point(s) inside phonation "
                             "(file appears cut/concatenated)")
            notes.append(f"median of {len(per_win)} valid window(s) of "
                         f"{self.vowel_window_max_s:.1f} s across {len(tokens)} token(s)")
            if reasons:
                notes.append("rejected " + ", ".join(f"{k}={v}" for k, v in reasons.items()))
            if np.isfinite(f0_sd_st) and f0_sd_st > 1.0:
                notes.append(f"unsteady F0 ({f0_sd_st:.2f} st SD)")
            if np.isfinite(slope) and abs(slope) > 0.5:
                notes.append(f"F0 drift {slope:+.2f} st/s")
            if np.isfinite(tremor_extent) and tremor_extent > 0.3:
                notes.append(f"tremor ~{tremor_rate:.1f} Hz, {tremor_extent:.2f} st")
            if np.isfinite(int_sd) and int_sd > 3.0:
                notes.append(f"unsteady loudness ({int_sd:.1f} dB SD)")
            if np.isfinite(jitter_iqr) and np.isfinite(jitter_med) and jitter_med > 0 \
                    and jitter_iqr > jitter_med:
                notes.append("HIGH window-to-window spread: single-number summary is "
                             "unstable for this file, inspect per-window values")

            # ---- USABILITY GATE (v9, tiered in v17) --------------------------
            # Without this, a summary built on 2 valid windows out of 11 lands in
            # the CSV and the bar plots looking exactly like one built on 16.
            #
            # v17: the gate is now TWO gates, because the old single flag was
            # discarding cepstral measures for a periodicity failure. CPPS needs
            # no pulse train and no cycle identification, so "the jitter estimate
            # is unstable" says nothing about whether the CPPS estimate is
            # trustworthy - yet sv_usable=0 removed both from the session table.
            # On real data that biased the headline metric: in one session two of
            # three takes were dropped for jitter instability, and the surviving
            # take happened to have the highest CPPS, moving the session median
            # from 16.8 to 18.9 dB. That is selection on a correlate of the
            # outcome, which is exactly what the robust block exists to prevent.
            #
            #   windows_usable      - enough steady signal to trust ANY window
            #                         median (count / duration / yield). Governs
            #                         cpps_db, cpps_iqr_db and the window-level
            #                         steadiness measures.
            #   perturbation_usable - additionally, the perturbation estimate is
            #                         itself stable and the signal type permits
            #                         cycle-based measures. Governs jitter and
            #                         shimmer only.
            #
            # sv_usable is retained as the AND of the two so existing filters keep
            # their old (conservative) meaning.
            bad_win = []
            if len(per_win) < self.min_valid_windows:
                bad_win.append(f"only {len(per_win)} valid window(s) "
                               f"(< {self.min_valid_windows})")
            if measured_total_s < self.min_measured_s - 1e-6:   # v11: strict-<
                                                               # tolerance, see v11
                bad_win.append(f"only {measured_total_s:.1f} s measured "
                               f"(< {self.min_measured_s:.0f} s)")
            if n_tiles and (len(per_win) / float(n_tiles)) < 0.34:
                bad_win.append(f"only {len(per_win)} of {n_tiles} possible windows are steady "
                               "(<34%): phonation is largely non-steady")
            bad_pert = []
            # v20: shimmer was screened on neither count, so a file could pass
            # perturbation_usable with no shimmer at all, or with a shimmer
            # whose window-to-window spread exceeded its own median.
            if not np.isfinite(jitter_med):
                bad_pert.append("jitter not measurable across the valid windows")
            if not np.isfinite(shimmer_med):
                bad_pert.append("shimmer not measurable across the valid windows")
            if (np.isfinite(jitter_iqr) and np.isfinite(jitter_med)
                    and jitter_med > 0 and jitter_iqr > jitter_med):
                bad_pert.append("window-to-window IQR exceeds the median jitter")
            if (np.isfinite(shimmer_iqr) and np.isfinite(shimmer_med)
                    and shimmer_med > 0 and shimmer_iqr > shimmer_med):
                bad_pert.append("window-to-window IQR exceeds the median shimmer")
            if sig_type == 3:
                bad_pert.append(
                    "signal type 3 (aperiodic): cycle-based perturbation is undefined")
            windows_usable = (len(bad_win) == 0)
            perturbation_usable = windows_usable and (len(bad_pert) == 0)
            bad = bad_win + bad_pert
            usable = (len(bad) == 0)
            if not windows_usable:
                notes.append("NOT USABLE for window-level statistics: "
                             + "; ".join(bad_win))
            if windows_usable and bad_pert:
                notes.append("Window medians usable, but PERTURBATION NOT USABLE ("
                             + "; ".join(bad_pert)
                             + "): CPPS/CPP and the spectral measures for this file "
                               "are still comparable")

            svm = SustainedVowelMetrics(
                n_tokens=len(tokens), token_durations_s=[float(d) for d in durs],
                token_spans_s=[(float(s), float(e)) for s, e in tokens],
                n_splices_detected=len(splices),
                splice_times_s=[float(c) for c in splices],
                mpt_longest_s=float(max(durs)), mpt_mean_s=float(np.mean(durs)),
                **_unbroken_kw(),
                total_phonation_s=float(np.sum(durs)),
                analysis_window_s=float(self.vowel_window_max_s),
                analysis_window_start_s=float(main_win[0]),
                n_windows_total=len(cands), n_windows_valid=len(per_win),
                window_rejections="; ".join(f"{k}:{v}" for k, v in reasons.items()),
                rej_low_voicing=int(reasons.get("low_voicing", 0)),
                rej_f0_outlier=int(reasons.get("f0_outlier", 0)),
                rej_f0_step=int(reasons.get("f0_step", 0)),
                rej_splice=int(reasons.get("splice", 0)),
                rej_other=int(reasons.get("no_data", 0) + reasons.get("unmeasurable", 0)),
                measured_total_s=measured_total_s,
                boundaries_source=boundaries_source,
                voiced_fraction=voiced_fraction,
                window_yield=(len(per_win) / float(n_tiles)) if n_tiles else float("nan"),
                cpps_token_db=cpps_token, hnr_token_db=hnr_token,
                signal_type=sig_type, signal_type_note=sig_note,
                voiced_at_045=probe.get("voiced_at_0.45", float("nan")),
                voiced_at_020=probe.get("voiced_at_0.20", float("nan")),
                voiced_low_floor=probe.get("voiced_low_floor", float("nan")),
                tremor_peak_ratio=tremor_ratio,
                usable=usable, usable_reasons="; ".join(bad),
                windows_usable=windows_usable,
                perturbation_usable=perturbation_usable,
                windows_usable_reasons="; ".join(bad_win),
                perturbation_usable_reasons="; ".join(bad_pert),
                f0_mean_hz=f0_mean, f0_sd_semitones=f0_sd_st, f0_drift_st_per_s=slope,
                f0_median_hz=w_median, f0_std_hz=w_std, f0_min_hz=w_min,
                f0_max_hz=w_max, f0_cv=w_cv, f0_range_semitones=w_range_st,
                measured_spans_s=[(float(a), float(b)) for a, b in measured_spans],
                jitter_local_percent=jitter_med, jitter_iqr=jitter_iqr,
                jitter_ppq5_iqr=_vq_iqr("jitter_ppq5_percent"),
                shimmer_apq11_iqr=_vq_iqr("shimmer_apq11_percent"),
                shimmer_local_percent=shimmer_med, shimmer_iqr=shimmer_iqr,
                hnr_db=hnr_med, hnr_iqr=hnr_iqr,
                cpps_db=cpps_val, cpps_iqr=cpps_iqr, cpps_source=cpps_src,
                intensity_sd_db=int_sd,
                tremor_rate_hz=tremor_rate, tremor_extent_semitones=tremor_extent,
                jitter_sd_across_tokens=jl_sd, shimmer_sd_across_tokens=sh_sd,
                hnr_sd_across_tokens=hn_sd,
                measured=True,
                assessment="; ".join(notes))
            svm = self._fill_coverage(svm, tokens,
                                      [d["span"] for d in per_win], frame_prof)
            if svm.coverage_note:
                svm.assessment = svm.assessment + "; " + svm.coverage_note
            return svm, panel_vq
        except Exception as e:
            return SustainedVowelMetrics(
                assessment=f"Error in sustained-vowel analysis: {e}"), None

    def _fill_coverage(self, svm: SustainedVowelMetrics,
                       tokens: List[Tuple[float, float]],
                       measured_spans: List[Tuple[float, float]],
                       frame_prof: Optional[Dict[str, float]] = None
                       ) -> SustainedVowelMetrics:
        """
        Attach the v18 coverage ledger to a result (any exit path).

        Nothing here changes a measurement: it only accounts for the seconds of
        phonation that went into, or were dropped from, the medians already
        computed. Called on the failure paths too - on those files the ledger IS
        the finding, and sv_steady_frame_fraction is the only column that can
        tell an unsteady voice apart from a take too short for the gate.
        """
        try:
            cov = phonation_coverage(
                tokens, measured_spans,
                win_s=self.vowel_window_max_s,
                edge_trim_s=self.vowel_edge_trim_s,
                min_window_s=self.vowel_window_min_s)
            fp = frame_prof or {}
            svm.analyzable_total_s = cov["analyzable_total_s"]
            svm.analyzed_fraction = cov["analyzed_fraction"]
            svm.window_yield_s = cov["window_yield_s"]
            svm.discard_edge_s = cov["discard_edge_s"]
            svm.discard_quantisation_s = cov["discard_quantisation_s"]
            svm.discard_unsteady_s = cov["discard_unsteady_s"]
            svm.discard_onset_s = cov["discard_onset_s"]
            svm.discard_interior_s = cov["discard_interior_s"]
            svm.discard_offset_s = cov["discard_offset_s"]
            svm.discard_dead_token_s = cov["discard_dead_token_s"]
            svm.n_interior_gaps = int(cov["n_interior_gaps"])
            svm.steady_frame_fraction = fp.get("steady_frame_fraction", float("nan"))
            svm.longest_steady_run_s = fp.get("longest_steady_run_s", float("nan"))
            svm.n_steady_stretches = int(fp.get("n_steady_stretches", 0) or 0)
            svm.coverage_note = describe_coverage(
                cov, svm.steady_frame_fraction, svm.longest_steady_run_s,
                svm.n_steady_stretches)
            # ---- is the GATE the problem, or the voice? ---------------------
            # min_valid_windows x vowel_window_max_s is a geometric requirement:
            # 3 x 2.0 s of NON-OVERLAPPING window needs 6.5 s of flawless
            # phonation before anything about the voice is asked. When the frames
            # are steady and the gate still fails, say so - otherwise the console
            # blames the voice for a setting.
            sff = svm.steady_frame_fraction
            if (np.isfinite(sff) and sff >= 0.80 and svm.measured
                    and not svm.windows_usable):
                svm.coverage_note += (
                    "; NOTE: the frames themselves are steady, so this take failed the "
                    "usability gate on window COUNT/DURATION geometry rather than on the "
                    "voice - check min_valid_windows x vowel_window_max_s against "
                    "sv_total_phonation_s before excluding it")
            elif np.isfinite(sff) and sff >= 0.60 and not svm.measured:
                svm.coverage_note += (
                    "; NOTE: a majority of frames pass the same criteria frame-by-frame, "
                    "yet no whole window survived - the loss is alignment or short "
                    "transients, not a globally unsteady voice; read the rejection tally "
                    "and consider a finer vowel_window_grid_hop_s or a shorter "
                    "vowel_window_max_s")
        except Exception as e:
            svm.coverage_note = f"coverage ledger failed: {e}"
        return svm

    def _load_boundaries(self, file_path: str) -> List[Tuple[float, float]]:
        """
        Look for user-supplied token boundaries for this file (v9).
        Searched, in order:
            <boundaries_dir>/<stem><boundaries_suffix>
            <same folder as the wav>/<stem><boundaries_suffix>
            <same folder as the wav>/<stem>.TextGrid
        """
        stem = Path(file_path).stem
        cands = []
        if self.boundaries_dir:
            cands.append(Path(self.boundaries_dir) / f"{stem}{self.boundaries_suffix}")
        cands.append(Path(file_path).with_name(f"{stem}{self.boundaries_suffix}"))
        cands.append(Path(file_path).with_suffix(".TextGrid"))
        for c in cands:
            try:
                if c.exists():
                    spans = read_boundaries_file(c)
                    if spans:
                        return spans
                    print(f"    ! boundaries file found but not parseable: {c}")
            except Exception:
                continue
        return []

    # ------------------------- MAIN ANALYSIS METHOD ----------------------------
    def analyze(self, file_path: str) -> SpeechAnalysisResult:
        print(f"\nAnalyzing: {file_path}")
        print("-" * 60)
        sound, y, sr = self.load_audio(file_path)
        duration_total = len(y) / sr
        # v13: per-file state must not leak into the next file
        self._current_file = str(file_path)
        self._pp_cache_key, self._pp_cache = None, None
        self._snr_reference = "none"
        self._digital_silence = False
        # v16: _f0_wide_median was initialised in __init__ only, so if
        # estimate_f0_range() raised for this file the PREVIOUS file's value
        # stayed in place and was used for the subharmonic-lock test and written
        # to the CSV as this file's f0_wide_median_hz. Reset it with the rest of
        # the per-file state.
        self._f0_wide_median = float("nan")
        self._f0_subharm_lock = 0
        self._f0_lock_st = float("nan")
        self._octave = {}
        self._silence_threshold_used = float("nan")

        # ---- 0. TASK ROUTING: decides which metrics are even meaningful -----
        self._task, self._task_source = self.resolve_task(file_path, sound, y, sr)
        is_sust = (self._task == TASK_SUSTAINED)
        print(f"  Task: {self._task}  [{self._task_source}]")

        # v9: the envelope, the silence line and the speech boundaries do NOT
        # depend on F0, so they are computed first and the octave check can then
        # run on the actual phonation instead of on the whole file.
        rms, rms_times = self.compute_rms_envelope(y, sr)
        silence_threshold = self.get_silence_threshold(rms, task=self._task)
        self._silence_threshold_used = float(silence_threshold)
        print(f"  Silence threshold: {silence_threshold:.1f} dB")
        speech_start, speech_end = self.find_speech_boundaries(rms, rms_times, silence_threshold)
        duration_speech = speech_end - speech_start
        print(f"  Duration: {duration_total:.2f}s total, {duration_speech:.2f}s speech")

        self._adaptive_floor, self._adaptive_ceiling, self._adaptive_median = \
            self.estimate_f0_range(sound, task=self._task)
        # ---- OCTAVE VERIFICATION (v9) ---------------------------------------
        self._octave = {}
        if self.octave_check:
            self._octave = resolve_f0_octave(
                y, sr, [(speech_start, speech_end)], self._adaptive_median,
                strong_db=self.octave_strong_db)
            v = self._octave.get("verdict", "inconclusive")
            if v in ("tracker_halved", "tracker_doubled"):
                print(f"    !! OCTAVE ERROR: {self._octave['note']}")
                # v13: the comb test assumes a steady harmonic spectrum. On a
                # reading passage F0 moves constantly and the "loudest chunks" are
                # whatever vowels happen to be loud, so the verdict is far less
                # trustworthy - and an automatic octave shift on connected speech
                # would be a silent, hard-to-notice change. For reading the check
                # therefore stays a DIAGNOSTIC: it reports and never corrects.
                if not is_sust:
                    print("       -> connected speech: reported only, range NOT changed "
                          "(the comb test is designed for steady phonation). If this "
                          "fires, check the file by hand.")
                elif self.octave_autocorrect and self.pitch_range is None:
                    factor = float(self._octave["factor"])
                    self._adaptive_median *= factor
                    self._adaptive_floor = max(40.0, self._adaptive_floor * factor)
                    self._adaptive_ceiling = min(700.0, self._adaptive_ceiling * factor)
                    print(f"       -> analysis range corrected by x{factor:g} "
                          f"(now {self._adaptive_floor:.0f}-{self._adaptive_ceiling:.0f} Hz). "
                          "VERIFY BY EAR before using this file.")
                elif self.pitch_range is not None:
                    print("       -> a fixed pitch_range is in force, so the range was NOT "
                          "changed. If this fires on several files, the fixed range is "
                          "probably set an octave off.")
                else:
                    print("       -> octave_autocorrect=False, range left unchanged.")
            elif v == "ambiguous":
                print(f"    ! F0 OCTAVE AMBIGUOUS: {self._octave['note']}")
            elif v == "inconclusive":
                print(f"    ! F0 octave check inconclusive: {self._octave.get('note', '')}")
        # ---- SUBHARMONIC LOCK (v11) -----------------------------------------
        # How far an UNCONSTRAINED tracker falls below the in-range F0. An octave
        # gap means the signal really does carry a strong subharmonic: with a
        # free floor the tracker locks onto it. That is not merely a settings
        # problem, it is period doubling / diplophonia, and it turns out to be
        # the cleanest single index of phonatory instability in this dataset.
        self._f0_subharm_lock = 0
        self._f0_lock_st = float("nan")
        if (np.isfinite(self._f0_wide_median) and self._f0_wide_median > 0
                and np.isfinite(self._adaptive_median) and self._adaptive_median > 0):
            self._f0_lock_st = float(12.0 * np.log2(
                self._adaptive_median / self._f0_wide_median))
            if self._f0_lock_st >= 7.0:
                self._f0_subharm_lock = 1
        print(f"  Adaptive F0 range: {self._adaptive_floor:.0f}-{self._adaptive_ceiling:.0f} Hz "
              f"(median {self._adaptive_median:.0f} Hz"
              + (f", octave: {self._octave.get('verdict')}" if self._octave else "") + ")")
        if self._f0_subharm_lock:
            print(f"    ! SUBHARMONIC LOCK: an unconstrained tracker sits at "
                  f"{self._f0_wide_median:.0f} Hz vs {self._adaptive_median:.0f} Hz in-range "
                  f"({self._f0_lock_st:.1f} st). The fixed range is handling it, but the "
                  f"subharmonic energy is real (period doubling) - report it as a finding.")

        recording_quality = self.assess_recording_quality(
            y, sr, speech_start, speech_end, silence_threshold,
            rms=rms, rms_times=rms_times, task=self._task)
        snr_txt = (f"{recording_quality.snr_db:.0f} dB"
                   if np.isfinite(recording_quality.snr_db) else "n/e")
        print(f"  Recording quality: {recording_quality.quality_label} "
              f"(score {recording_quality.quality_score:.0f}, SNR {snr_txt}, "
              f"BW {recording_quality.effective_bandwidth_hz:.0f} Hz, "
              f"VQ reliable: {recording_quality.vq_reliable})")
        for w in recording_quality.warnings:
            print(f"    ! {w}")

        pause_metrics, active_speech_mask = self.detect_pauses(
            rms, rms_times, speech_start, speech_end, silence_threshold)
        pause_word = "Breath breaks" if is_sust else "Pauses"
        print(f"  {pause_word}: {pause_metrics.count} ({pause_metrics.pauses_per_minute:.1f}/min)")

        intensity_metrics = self.analyze_intensity(
            sound, speech_start, speech_end, active_speech_mask, rms_times)
        print(f"  Intensity (active speech): {intensity_metrics.active_mean_db:.1f} dB "
              f"[uncalibrated] (active std: {intensity_metrics.active_std_db:.1f}, "
              f"nucleus std: {intensity_metrics.nucleus_std_db:.1f})")

        pitch_metrics, pitch_obj = self.analyze_pitch(sound, speech_start, speech_end)
        oct_txt = (f", octave-repaired {pitch_metrics.octave_repair_fraction*100:.1f}% of frames"
                   if pitch_metrics.octave_repair_fraction > 0.005 else "")
        print(f"  Pitch: {pitch_metrics.mean_hz:.1f} Hz "
              f"(CV: {pitch_metrics.coefficient_of_variation:.3f}{oct_txt})")

        # Fillers are a connected-speech construct only.
        if is_sust:
            filler_metrics = FillerMetrics()
        else:
            filler_metrics = self.detect_fillers(
                y, sr, speech_start, speech_end, rms, rms_times, silence_threshold,
                sound, pitch_obj)
            print(f"  Fillers: {filler_metrics.count} ({filler_metrics.fillers_per_minute:.1f}/min)")

        articulation_time = (speech_end - speech_start) - pause_metrics.total_duration
        voiced_intervals = self.get_voiced_intervals(sound, pitch_obj, self._adaptive_floor) \
            if pitch_obj is not None else []

        # ---- v19: THE TASK DECIDES WHICH SIGNAL IS MEASURED ------------------
        # Formants and the spectral block used to run BEFORE the task branch, on
        # the whole speech span, for both tasks. On a multi-token sustained file
        # that span includes the inter-token silence and the breaths, so the LTAS
        # shape measures, cpp_db and the F1-F2 cloud were all measured partly on
        # signal that is not phonation - and those are exactly the measures this
        # script falls back on when jitter and shimmer are undefined. The
        # sustained analysis therefore runs first, and its token spans and valid
        # windows are handed to the two blocks below.
        sustained_metrics = SustainedVowelMetrics()
        window_vq = None
        if is_sust:
            given_spans = self._load_boundaries(file_path)
            if given_spans:
                print(f"  Token boundaries read from file: {len(given_spans)} span(s) "
                      "- energy segmentation and splice detection skipped")
            sustained_metrics, window_vq = self.analyze_sustained_vowel(
                sound, y, sr, pitch_obj, speech_start, speech_end,
                recording_quality, rms, rms_times,
                silence_threshold_db=silence_threshold,
                given_spans=given_spans)
            # Formants from the SAME windows the perturbation medians came from;
            # tokens if no window survived; the whole span only as a last resort.
            formant_spans = (list(sustained_metrics.measured_spans_s)
                             or list(sustained_metrics.token_spans_s) or None)
            spectral_spans = list(sustained_metrics.token_spans_s) or None
        else:
            formant_spans = None
            spectral_spans = None

        formant_metrics = self.analyze_formants(
            sound, speech_start, speech_end, pitch_obj, intervals=formant_spans)
        f3_txt = (f"{formant_metrics.f3_mean_hz:.0f} Hz"
                  if np.isfinite(formant_metrics.f3_mean_hz) else "n/a (tracking failed)")
        fsrc = ("valid windows" if (is_sust and sustained_metrics.measured_spans_s)
                else ("tokens" if is_sust else "speech span"))
        print(f"  Formants: F1={formant_metrics.f1_mean_hz:.0f} Hz, "
              f"F2={formant_metrics.f2_mean_hz:.0f} Hz, F3={f3_txt} "
              f"(ceiling {formant_metrics.max_formant_hz_used:.0f} Hz, from {fsrc})")

        spectral_metrics = self.analyze_spectral(
            sound, y, sr, speech_start, speech_end,
            intervals=spectral_spans, task=self._task)
        print(f"  Spectral: centroid={spectral_metrics.spectral_centroid_mean_hz:.0f} Hz, "
              f"CPPS(LTAS span)={spectral_metrics.cpp_db:.1f} dB "
              f"[{spectral_metrics.cpp_source}], tilt over "
              f"{spectral_metrics.tilt_band_hz[0]:.0f}-{spectral_metrics.tilt_band_hz[1]:.0f} Hz "
              f"({spectral_metrics.span_source})")

        if is_sust:
            # ---- SUSTAINED VOWEL BRANCH -------------------------------------
            # Voice quality from the steady-state window, NOT the whole file.
            voice_quality_metrics = window_vq if window_vq is not None else \
                self.analyze_voice_quality(sound, pitch_obj, speech_start, speech_end,
                                           recording_quality)
            print(f"  Sustained: {sustained_metrics.n_tokens} token(s), "
                f"MPT(longest token)={sustained_metrics.mpt_longest_s:.2f}s, "
                f"total phonation={sustained_metrics.total_phonation_s:.1f}s")
            # ---- v19: the unbroken-phonation ledger -------------------------
            _sm = sustained_metrics
            if np.isfinite(_sm.mpt_longest_uninterrupted_s):
                _pct = (100.0 * _sm.mpt_longest_uninterrupted_s / _sm.mpt_longest_s
                        if np.isfinite(_sm.mpt_longest_s) and _sm.mpt_longest_s > 0
                        else float("nan"))
                print(f"             Longest UNBROKEN phonation: "
                      f"{_sm.mpt_longest_uninterrupted_s:.2f}s "
                      f"({_pct:.0f}% of the longest token) at "
                      f"{_sm.unbroken_start_s:.2f}-{_sm.unbroken_end_s:.2f}s "
                      f"in token {_sm.unbroken_token_index + 1}; "
                      f"{_sm.n_breaks_in_best_token} break(s) inside that token, "
                      f"{_sm.n_phonation_breaks} across all tokens "
                      f"[break = >={self.phonation_break_min_s*1000:.0f} ms below "
                      f"{_sm.break_threshold_db:.0f} dB or unvoiced; {_sm.unbroken_source}]")
                # A large voiced/energy gap is a different clinical finding from
                # a short take: the sound continues but the voice stops.
                if (np.isfinite(_sm.unbroken_energy_only_s)
                        and np.isfinite(_sm.mpt_longest_uninterrupted_s)
                        and _sm.unbroken_energy_only_s
                        > 1.5 * max(_sm.mpt_longest_uninterrupted_s, 0.2) + 0.3):
                    print(f"    ! APHONIC BREAKS: ignoring voicing, the sound runs "
                          f"unbroken for {_sm.unbroken_energy_only_s:.2f}s against "
                          f"{_sm.mpt_longest_uninterrupted_s:.2f}s of unbroken "
                          f"PHONATION. The airflow does not stop but the fold "
                          f"vibration does - report this rather than the shorter "
                          f"number alone (sv_unbroken_energy_only_s in the audit block).")
            else:
                print("             Longest UNBROKEN phonation: not measurable "
                      "(no voiced stretch above the break threshold inside any token)")
            rej = (f"  |  rejected: {sustained_metrics.window_rejections}"
                   if sustained_metrics.window_rejections else "")
            # v17: n_windows_total counts OVERLAPPING candidates (hop = win/2)
            # while window_yield uses the NON-OVERLAPPING count, so the two
            # percentages differ; state both denominators instead of leaving the
            # reader to reconcile them.
            print(f"             Windows: {sustained_metrics.n_windows_valid} valid of "
                  f"{sustained_metrics.n_windows_total} overlapping candidates "
                  f"({sustained_metrics.analysis_window_s:.1f}s each, hop "
                  f"{sustained_metrics.analysis_window_s/2:.1f}s){rej}")
            if sustained_metrics.signal_type:
                print(f"             Signal type: {sustained_metrics.signal_type_note}")
            if np.isfinite(sustained_metrics.voiced_at_020):
                verdict = interpret_probe(
                    {"voiced_at_0.20": sustained_metrics.voiced_at_020},
                    sustained_metrics.hnr_token_db, sustained_metrics.cpps_token_db)
                print(f"             Voicing probe: {sustained_metrics.voiced_at_045*100:.0f}% "
                      f"voiced at threshold 0.45, "
                      f"{sustained_metrics.voiced_at_020*100:.0f}% at 0.20, "
                      f"{sustained_metrics.voiced_low_floor*100:.0f}% with a 60 Hz floor")
                print(f"                            -> {verdict}")
            print(f"             Token level (no periodicity required): "
                  f"voiced {sustained_metrics.voiced_fraction*100:.0f}%, "
                  f"CPPS {sustained_metrics.cpps_token_db:.1f} dB, "
                  f"HNR {sustained_metrics.hnr_token_db:.1f} dB, "
                  f"window yield {sustained_metrics.window_yield*100:.0f}%")
            # ---- v18 COVERAGE LEDGER ---------------------------------------
            # "The windows were not good enough" is not actionable on its own.
            # These two lines say how many seconds were kept, and whether the
            # discarded seconds came from the voice or from the tiling geometry.
            sm = sustained_metrics
            print(f"             Coverage: analysed {sm.analyzed_fraction*100:.0f}% of "
                  f"{sm.total_phonation_s:.1f} s phonated "
                  f"({sm.measured_total_s:.1f} s in {sm.n_windows_valid} window(s)); "
                  f"steady frames {sm.steady_frame_fraction*100:.0f}%, longest steady "
                  f"run {sm.longest_steady_run_s:.1f} s "
                  f"({sm.n_steady_stretches} stretch(es))")
            print(f"             Discarded: by cause - unsteady "
                  f"{sm.discard_unsteady_s:.1f} s, geometry "
                  f"{(sm.discard_edge_s + sm.discard_quantisation_s):.1f} s "
                  f"(edge trim {sm.discard_edge_s:.1f} s + remainder "
                  f"{sm.discard_quantisation_s:.1f} s)  |  by position - onset "
                  f"{sm.discard_onset_s:.1f} s, interior {sm.discard_interior_s:.1f} s "
                  f"in {sm.n_interior_gaps} gap(s), offset {sm.discard_offset_s:.1f} s")
            print(f"             -> {sm.coverage_note}")
            if sustained_metrics.n_splices_detected:
                cuts = ", ".join(f"{c:.1f}" for c in sustained_metrics.splice_times_s)
                print(f"    ! {sustained_metrics.n_splices_detected} EDIT POINT(S) detected "
                      f"inside phonation at (s): {cuts}. This file looks cut/concatenated; "
                      "tokens were split there and the analysis window kept off the joins.")
            if sustained_metrics.token_spans_s:
                spans = ", ".join(f"{a:.1f}-{b:.1f}"
                                  for a, b in sustained_metrics.token_spans_s)
                print(f"             Token spans (s): {spans}")
            # Cross-check against the independent energy-gap count. These should
            # agree: n productions leave n-1 gaps. A mismatch means the token
            # segmentation is splitting or merging productions.
            # v20: the cross-check depends on where the boundaries came from.
            # Under one-production-per-file the expectation is exactly 1, not
            # "breath gaps + 1" - an internal breath is expected there and must
            # not be reported as a segmentation error. User-supplied boundaries
            # are authoritative and are never second-guessed.
            if sustained_metrics.boundaries_source == "single_production_per_file":
                expected = 1
                sustained_metrics.n_tokens_expected = 1.0
                sustained_metrics.token_count_mismatch = (
                    sustained_metrics.n_tokens != 1)
            elif sustained_metrics.boundaries_source == "file":
                expected = sustained_metrics.n_tokens
                sustained_metrics.n_tokens_expected = float(expected)
                sustained_metrics.token_count_mismatch = False
            else:
                expected = pause_metrics.count + 1 + sustained_metrics.n_splices_detected
                sustained_metrics.n_tokens_expected = float(expected)
                sustained_metrics.token_count_mismatch = (
                    sustained_metrics.n_tokens != expected)
            if sustained_metrics.token_count_mismatch:
                print(f"    ! Token count ({sustained_metrics.n_tokens}) disagrees with "
                      f"breath gaps + edit points + 1 ({expected}). Check the spans above "
                      f"against the audio. MPT depends directly on these boundaries; "
                      f"supply them via a <stem>_tokens.csv file to remove the guesswork.")
            if sustained_metrics.measured:
                vq_tag = "" if voice_quality_metrics.reliable else \
                    "  [recording flagged: interpret with caution]"
                print(f"             Jitter: {sustained_metrics.jitter_local_percent:.3f}% "
                      f"(IQR {sustained_metrics.jitter_iqr:.3f}), "
                      f"Shimmer: {sustained_metrics.shimmer_local_percent:.3f}% "
                      f"(IQR {sustained_metrics.shimmer_iqr:.3f})")
                print(f"             HNR: {sustained_metrics.hnr_db:.1f} dB "
                      f"(IQR {sustained_metrics.hnr_iqr:.1f}), "
                      f"CPPS: {sustained_metrics.cpps_db:.1f} dB "
                      f"(IQR {sustained_metrics.cpps_iqr:.1f}){vq_tag}")
            else:
                print("             Jitter/Shimmer/HNR: NOT MEASURABLE (undefined for this "
                      "signal type, not missing data)")
                print("             STILL COMPARABLE for this file: sv_cpps_token_db, "
                      "sv_hnr_token_db, voiced_fraction, sv_window_yield, cpp_db, "
                      "alpha_ratio_db, hammarberg_index, spectral_tilt_db_per_khz, "
                      "sv_mpt_longest_s, intensity_active_mean_db, f0_subharmonic_lock")
            print(f"             F0={sustained_metrics.f0_mean_hz:.1f} Hz, "
                  f"SD={sustained_metrics.f0_sd_semitones:.2f} st, "
                  f"drift={sustained_metrics.f0_drift_st_per_s:+.2f} st/s, "
                  f"tremor={sustained_metrics.tremor_rate_hz:.1f} Hz/"
                  f"{sustained_metrics.tremor_extent_semitones:.2f} st, "
                  f"intensity SD={sustained_metrics.intensity_sd_db:.1f} dB")
            print(f"             {sustained_metrics.assessment}")
            if sustained_metrics.measured and not sustained_metrics.windows_usable:
                print(f"    !! WINDOW MEDIANS NOT USABLE for group statistics "
                      f"({sustained_metrics.windows_usable_reasons}). The numbers above "
                      f"are reported for inspection only; sv_windows_usable=0 in the CSV.")
            elif (sustained_metrics.measured
                  and sustained_metrics.windows_usable
                  and not sustained_metrics.perturbation_usable):
                print(f"    !  PERTURBATION NOT USABLE "
                      f"({sustained_metrics.perturbation_usable_reasons}): exclude "
                      f"jitter/shimmer for this file, but its CPPS/CPP, token HNR and "
                      f"spectral measures ARE comparable (sv_perturbation_usable=0, "
                      f"sv_windows_usable=1 in the CSV).")
            # v15: two measures that ARE meaningful on a sustained vowel and had
            # been suppressed with the connected-speech block:
            #   - intensity decay across the vowel (vocal fatigue / breath support)
            #   - F1-F2 cloud dispersion, which on ONE vowel means articulatory
            #     STEADINESS (so the good direction is reversed vs reading, hence
            #     the separate sv_ column names)
            # v19: measured on the LONGEST TOKEN, not on the whole file. The
            # console has always called this "decay across the vowel", but the
            # fit ran from speech_start to speech_end, so on a three-trial file
            # it described the trend across the recording session - trial 3
            # quieter than trial 1 - with the breaths in between contributing
            # frames as well. It also now ships as a rate, because the old value
            # is dB across the analysed span and therefore scales with how long
            # the take happened to be.
            try:
                _tok = list(sustained_metrics.token_spans_s)
                _span = (max(_tok, key=lambda ab: ab[1] - ab[0]) if _tok
                         else (speech_start, speech_end))
                _across, _per_s, _ = intensity_decay_over(
                    sound, [_span], self._adaptive_floor)
                sustained_metrics.intensity_decay_db = _across
                sustained_metrics.intensity_decay_db_per_s = _per_s
            except Exception:
                pass
            try:
                la, sp_hz = formant_cloud_dispersion(
                    getattr(formant_metrics, "_f1_values", []) or [],
                    getattr(formant_metrics, "_f2_values", []) or [])
                sustained_metrics.f1f2_dispersion_logarea = la
                sustained_metrics.f1f2_cloud_spread_hz = sp_hz
            except Exception:
                pass
            # v17b: this panel used to print unconditionally, so a file with ZERO
            # valid windows showed "Jitter/Shimmer/HNR: NOT MEASURABLE" and then,
            # four lines later, a full panel of ppq5 / apq11 / NHR / autocorr taken
            # from the whole-file fallback. The CSV now blanks those (see _pert() in
            # _raw_metric_table), so the console has to agree - otherwise the
            # printout invites you to read numbers the export deliberately withheld.
            # The label was wrong too: with no windows they were never window medians.
            if sustained_metrics.measured:
                print(f"             Voice-report panel (median over "
                      f"{sustained_metrics.n_windows_valid} valid window(s)): "
                      f"ppq5={voice_quality_metrics.jitter_ppq5_percent:.3f}%, "
                      f"rap={voice_quality_metrics.jitter_rap_percent:.3f}%, "
                      f"apq11={voice_quality_metrics.shimmer_apq11_percent:.3f}%, "
                      f"shim_dB={voice_quality_metrics.shimmer_local_db:.3f}, "
                      f"NHR={voice_quality_metrics.nhr:.4f}, "
                      f"autocorr={voice_quality_metrics.mean_autocorrelation:.3f}, "
                      f"breaks={voice_quality_metrics.voice_breaks_degree_percent:.1f}%")
            else:
                print("             Voice-report panel: NOT REPORTED. No steady-state "
                      "window was measurable, so every cycle-based sub-measure would "
                      "come from the whole-file fallback, i.e. from the tracker's "
                      "behaviour on the few frames it accepted. Blank in the CSV.")
            print(f"             In-vowel steadiness: decay="
                  f"{sustained_metrics.intensity_decay_db:+.1f} dB across the vowel, "
                  f"F1F2 spread={sustained_metrics.f1f2_cloud_spread_hz:.0f} Hz")
            # Connected-speech blocks are NOT computed for this task.
            reading_metrics = ReadingMetrics(
                syllable_source="n/a (sustained vowel)",
                assessment="Reading/connected-speech metrics not applicable to sustained phonation")
            rhythm_metrics = RhythmMetrics(
                total_duration_sec=duration_speech, speech_duration_sec=duration_speech,
                articulation_time_sec=articulation_time,
                phonation_time_sec=sustained_metrics.total_phonation_s,
                speech_rate_syllables_per_sec=float("nan"),
                articulation_rate_syllables_per_sec=float("nan"),
                npvi_v=None, rpvi_c=None,
                percent_v=pitch_metrics.voiced_frames_percent,
                varco_v=float("nan"), delta_v=float("nan"), delta_c=float("nan"),
                assessment="Rate/rhythm not applicable to sustained phonation",
                estimated_syllable_count=0)
        else:
            # ---- CONNECTED-SPEECH BRANCH ------------------------------------
            voice_quality_metrics = self.analyze_voice_quality(
                sound, pitch_obj, speech_start, speech_end, recording_quality)
            if voice_quality_metrics.measured:
                vq_tag = "" if voice_quality_metrics.reliable else \
                    "  [recording flagged: interpret with caution]"
                print(f"  Jitter: {voice_quality_metrics.jitter_local_percent:.3f}%, "
                      f"Shimmer: {voice_quality_metrics.shimmer_local_percent:.3f}%, "
                      f"HNR: {voice_quality_metrics.hnr_db:.1f} dB{vq_tag}")
            else:
                print("  Jitter/Shimmer/HNR: NOT MEASURABLE "
                      "(insufficient reliable voiced segments)")
            reading_metrics, exact_speech_rate, exact_artic_rate = self.analyze_reading_metrics(
                sound, y, sr, file_path, speech_start, speech_end,
                articulation_time, voiced_intervals, formant_metrics)
            print(f"  Reading: passage={reading_metrics.passage_name or 'unknown'} "
                  f"({reading_metrics.syllable_source}), "
                  f"syllables={reading_metrics.syllable_count_used}, "
                  f"speaking_time_frac={reading_metrics.speaking_time_fraction:.2f}")
            if reading_metrics.syllable_source == "estimated":
                print("    ! Syllable count ESTIMATED from the envelope: rate and nPVI are "
                      "approximate. Add a passage token to the filename "
                      f"({', '.join(list(KNOWN_PASSAGE_SYLLABLES)[:4])}, ...) for exact rates.")
            print(f"           EMS(3-8Hz)={reading_metrics.ems_3_8hz_ratio:.3f} "
                  f"(peak {reading_metrics.ems_peak_freq_hz:.1f} Hz), "
                  f"vowel-dispersion(log-area)={reading_metrics.formant_dispersion_logarea:.2f}, "
                  f"decay={reading_metrics.intensity_decay_db:.1f} dB, "
                  f"CPPS={reading_metrics.cpps_db:.1f} dB ({reading_metrics.cpps_source})")
            rhythm_metrics = self.analyze_rhythm(
                y, sr, speech_start, speech_end, pause_metrics, pitch_metrics,
                syllable_count=reading_metrics.syllable_count_used)
            if rhythm_metrics.npvi_v:
                print(f"  Rhythm: {rhythm_metrics.speech_rate_syllables_per_sec:.2f} syll/s, "
                      f"nPVI={rhythm_metrics.npvi_v:.1f} "
                      f"(rate source: {reading_metrics.syllable_source})")
            else:
                print(f"  Rhythm: {rhythm_metrics.speech_rate_syllables_per_sec:.2f} syll/s "
                      f"(rate source: {reading_metrics.syllable_source})")

        fluency_metrics = self.compute_fluency(
            pause_metrics, filler_metrics, intensity_metrics,
            pitch_metrics, voice_quality_metrics, rhythm_metrics,
            task=self._task, enabled=self.enable_fluency_index)
        if np.isfinite(fluency_metrics.overall_fluency):
            print(f"  Fluency: {fluency_metrics.overall_fluency:.1f}/100 "
                  f"({fluency_metrics.clinical_severity})")
        else:
            print(f"  Fluency: {fluency_metrics.clinical_severity}")

        settings = dict(self._settings)
        settings["task_resolved"] = self._task
        settings["task_source"] = self._task_source
        # v9 audit trail: the exact settings this file was measured with
        settings["pitch_floor_used"] = float(self._adaptive_floor)
        settings["pitch_ceiling_used"] = float(self._adaptive_ceiling)
        settings["pitch_median_used"] = float(self._adaptive_median)
        settings["silence_threshold_db_used"] = float(self._silence_threshold_used)
        settings["snr_reference"] = self._snr_reference
        settings["digital_silence"] = bool(self._digital_silence)
        settings["octave_verdict"] = self._octave.get("verdict", "not_checked")
        settings["octave_note"] = self._octave.get("note", "")
        settings["octave_score_ref_db"] = float(self._octave.get("score_ref_db", np.nan))
        settings["octave_score_half_db"] = float(self._octave.get("score_half_db", np.nan))
        settings["octave_factor"] = float(self._octave.get("factor", 1.0))
        settings["f0_wide_median"] = float(self._f0_wide_median)
        settings["f0_subharm_lock"] = int(getattr(self, "_f0_subharm_lock", 0))
        settings["f0_lock_st"] = float(getattr(self, "_f0_lock_st", float("nan")))
        return SpeechAnalysisResult(
            file_path=file_path, file_name=Path(file_path).name,
            duration_total=duration_total, duration_speech=duration_speech,
            speech_start=speech_start, speech_end=speech_end,
            pause_metrics=pause_metrics, filler_metrics=filler_metrics,
            intensity_metrics=intensity_metrics, pitch_metrics=pitch_metrics,
            voice_quality_metrics=voice_quality_metrics, formant_metrics=formant_metrics,
            spectral_metrics=spectral_metrics, rhythm_metrics=rhythm_metrics,
            fluency_metrics=fluency_metrics, reading_metrics=reading_metrics,
            sustained_metrics=sustained_metrics,
            recording_quality=recording_quality,
            task=self._task, task_source=self._task_source,
            analysis_settings=settings)

    # --------------- ROBUST F0 RANGE + VOICED-SEGMENT EXTRACTION ---------------
    def estimate_f0_range(self, sound, task=None) -> Tuple[float, float, float]:
        """
        Adaptive F0 search range.

        FIXED in v8: for a SUSTAINED VOWEL the range is tied tightly to the
        median (roughly +/- one fifth), because a wide ceiling invites octave
        DOUBLING - which is exactly what produced "median 101 Hz" alongside
        "mean 140.8 Hz, CV 0.354" on a steady /a/. For connected speech the
        wider range is kept, since real prosody spans it.

        v9: `pitch_range=(floor, ceiling)` short-circuits all of this. For a
        LONGITUDINAL design that is the correct choice: a range re-estimated per
        file also changes the jitter/shimmer period bounds and the
        PowerCepstrogram window, so a PRE/POST difference partly reflects a
        settings difference. The measured median is still returned (and still
        octave-checked in analyze()) for reporting.
        """
        task = task or self._task
        if self.pitch_range is not None:
            floor, ceiling = float(self.pitch_range[0]), float(self.pitch_range[1])
            med = 0.5 * (floor + ceiling)
            # FIXED in v11. This used to return the median of an UNCONSTRAINED
            # wide pass, which on a subharmonic-prone voice is F0/2 - so with a
            # correct fixed range the log read "range 110-330 Hz (median 95 Hz)"
            # and the octave check, fed that 95 Hz, raised OCTAVE ERROR on files
            # it had just analysed correctly at 186 Hz. The reference is now the
            # median measured INSIDE the fixed range; the unconstrained median is
            # kept separately, because the gap between the two is informative in
            # its own right (see _f0_wide_median below).
            try:
                inr = call(sound, "To Pitch", 0.0, floor, ceiling)
                m = call(inr, "Get quantile", 0.0, 0.0, 0.50, "Hertz")
                if m and not np.isnan(m) and m > 0:
                    med = float(m)
            except Exception:
                pass
            self._f0_wide_median = float("nan")
            try:
                wide = call(sound, "To Pitch", 0.0, max(40.0, floor * 0.55),
                            min(800.0, ceiling * 1.6))
                mw = call(wide, "Get quantile", 0.0, 0.0, 0.50, "Hertz")
                if mw and not np.isnan(mw) and mw > 0:
                    self._f0_wide_median = float(mw)
            except Exception:
                pass
            return floor, ceiling, med
        try:
            wide = call(sound, "To Pitch", 0.0, 50.0, 600.0)
            q05 = call(wide, "Get quantile", 0.0, 0.0, 0.05, "Hertz")
            q95 = call(wide, "Get quantile", 0.0, 0.0, 0.95, "Hertz")
            med = call(wide, "Get quantile", 0.0, 0.0, 0.50, "Hertz")
            if (q05 is None or np.isnan(q05) or q05 <= 0 or
                    q95 is None or np.isnan(q95) or q95 <= 0):
                return 75.0, 500.0, (med if med and not np.isnan(med) else 150.0)
            if not (med and not np.isnan(med) and med > 0):
                med = 0.5 * (q05 + q95)
            self._f0_wide_median = float(med)
            if task == TASK_SUSTAINED:
                # narrow, median-anchored: blocks 2x/0.5x tracking outright
                floor = max(50.0, med / 1.7)
                ceiling = min(600.0, med * 1.7)
                if ceiling <= floor * 1.4:
                    floor, ceiling = max(50.0, med / 2.0), min(600.0, med * 2.0)
                return float(floor), float(ceiling), float(med)
            floor = max(50.0, q05 * 0.83)
            ceiling = q95 * 1.45
            abs_cap = max(300.0, med * 2.2)
            ceiling = min(ceiling, abs_cap, 600.0)
            if ceiling <= floor * 1.5:
                floor, ceiling = max(50.0, floor * 0.8), floor * 3.0
            return float(floor), float(ceiling), float(med)
        except Exception:
            return 75.0, 500.0, 150.0

    def get_voiced_intervals(self, sound, pitch, f0_floor) -> List[Tuple[float, float]]:
        n_frames = call(pitch, "Get number of frames")
        if n_frames < 2:
            return []
        dt = call(pitch, "Get time step")
        if dt is None or dt <= 0 or np.isnan(dt):
            dt = 0.01
        voiced_flags = np.zeros(n_frames, dtype=bool); times = np.zeros(n_frames)
        for i in range(1, n_frames + 1):
            t = call(pitch, "Get time from frame number", i)
            v = call(pitch, "Get value in frame", i, "Hertz")
            times[i - 1] = t
            voiced_flags[i - 1] = (v is not None) and (v > 0) and (not np.isnan(v))
        runs = []; i = 0
        while i < n_frames:
            if voiced_flags[i]:
                j = i
                while j + 1 < n_frames and voiced_flags[j + 1]:
                    j += 1
                runs.append((times[i] - dt / 2, times[j] + dt / 2)); i = j + 1
            else:
                i += 1
        if not runs:
            return []
        max_gap = max(0.090, 1.25 / max(f0_floor, 1.0))
        merged = [list(runs[0])]
        for s, e in runs[1:]:
            if s - merged[-1][1] <= max_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        min_len = 0.06
        return [(s, e) for s, e in merged if (e - s) >= min_len]

    def estimate_syllables(self, y, sr, speech_start, speech_end) -> int:
        """
        FIXED: un-clamped peak-picking estimate. The previous version applied
        max(dur*1.0, min(count, dur*8.0)), coupling the count to duration and
        able to MASK bradylalia. Kept as a method for backward compatibility;
        the analysis path now prefers the known-passage count when available.
        """
        return estimate_syllables_unclamped(y, sr, speech_start, speech_end)


# =============================================================================
# AUTOMATIC PER-SUBJECT PITCH-RANGE CALIBRATION  (v18)
# =============================================================================
#
# WHY THIS EXISTS. A fixed pitch_range is required for a longitudinal design -
# re-estimating it per file changes the jitter/shimmer period bounds and the
# PowerCepstrogram window, so part of any PRE/POST difference becomes a settings
# difference. But a HAND-ENTERED constant has to be remembered, transcribed and
# kept matched to the subject, and when it is wrong the failure is silent and
# total: with the floor above the subject's real period the tracker cannot find
# periodicity, voiced_fraction collapses, HNR reads ~1 dB, every window fails
# 'low_voicing' and the whole subject comes out as "aperiodic". Nothing in the
# per-file output says "your floor is wrong" - it says "this voice is severely
# dysphonic", which is a clinical conclusion drawn from a typo.
#
# So the range is derived from the audio ONCE PER SUBJECT, written to a JSON
# file, and re-used verbatim on every later run. Automatic where it can be,
# fixed where it must be, and recorded either way.
#
# THE OCTAVE IS NOT AUTOMATABLE, AND THIS DOES NOT PRETEND OTHERWISE. An
# autocorrelation tracker cannot distinguish F0 from F0/2, and neither can a
# median over files. The comb test (resolve_f0_octave) settles the clear cases
# and flags the rest: when a subject's phonation is period-doubled or
# diplophonic, a full harmonic comb genuinely exists at both F0 and F0/2 and the
# two readings mean different things clinically. In that case the calibrator
# prints BOTH candidate ranges and refuses to choose - confidence='check'.
# Choose by listening to one file, then pass the choice in once.


def _subject_id_from_path(path: str) -> str:
    """Best-effort subject label from a path, for the calibration record only."""
    parts = [p for p in re.split(r"[\\/]+", str(path)) if p]
    for p in reversed(parts[:-1]):
        if re.search(r"[A-Za-z]{2,}\s?\d{2,}", p):
            return p
    return parts[-2] if len(parts) > 1 else "unknown"


def probe_file_f0(analyzer: 'CompleteSpeechAnalyzer', wav_path: str,
                  wide_floor: float = 40.0, wide_ceiling: float = 600.0,
                  voicing_threshold: float = 0.40) -> Dict[str, Any]:
    """
    One file, one question: what period does this voice actually have?

    Deliberately run with a WIDE range (40-600 Hz by default) so the answer is
    not conditioned on the very setting being calibrated, and with the comb test
    on top so a halving/doubling is caught rather than averaged in. Nothing here
    is reported as a measurement of the voice; it exists only to choose the
    analysis range.
    """
    out: Dict[str, Any] = dict(file=str(wav_path), ok=False, median_hz=float("nan"),
                               voiced_fraction=float("nan"), phonation_s=float("nan"),
                               verdict="", factor=1.0, corrected_hz=float("nan"),
                               score_ref_db=float("nan"), score_half_db=float("nan"),
                               note="")
    try:
        sound, y, sr = analyzer.load_audio(str(wav_path))
        rms, rms_times = analyzer.compute_rms_envelope(y, sr)
        thr = analyzer.get_silence_threshold(rms, task=TASK_SUSTAINED)
        t0, t1 = analyzer.find_speech_boundaries(rms, rms_times, thr)
        pitch = call(sound, "To Pitch (ac)", 0.0, float(wide_floor), 15, "no",
                     analyzer.silence_threshold, float(voicing_threshold),
                     0.01, 0.35, 0.14, float(wide_ceiling))
        n = call(pitch, "Get number of frames")
        vals, n_in = [], 0
        for i in range(1, int(n) + 1):
            t = call(pitch, "Get time from frame number", i)
            if t0 <= t <= t1:
                n_in += 1
                v = call(pitch, "Get value in frame", i, "Hertz")
                if v is not None and v > 0 and not np.isnan(v):
                    vals.append(float(v))
        if not vals or n_in == 0:
            out["note"] = "no voiced frame in the phonation span"
            return out
        med = float(np.median(vals))
        out.update(ok=True, median_hz=med, corrected_hz=med,
                   voiced_fraction=float(len(vals)) / float(n_in),
                   phonation_s=float(t1 - t0))
        try:
            oc = resolve_f0_octave(y, sr, [(t0, t1)], med,
                                   strong_db=analyzer.octave_strong_db)
            out.update(verdict=str(oc.get("verdict", "")),
                       factor=float(oc.get("factor", 1.0)),
                       corrected_hz=float(oc.get("f0_true", med)),
                       score_ref_db=float(oc.get("score_ref_db", float("nan"))),
                       score_half_db=float(oc.get("score_half_db", float("nan"))),
                       note=str(oc.get("note", "")))
        except Exception as e:
            out["note"] = f"comb test failed: {e}"
    except Exception as e:
        out["note"] = f"probe failed: {e}"
    return out


def calibrate_pitch_range(wav_files: List[str],
                          margin: float = 1.7,
                          wide_floor: float = 40.0,
                          wide_ceiling: float = 600.0,
                          min_voiced_fraction: float = 0.30,
                          floor_hard_min: float = 40.0,
                          ceiling_hard_max: float = 600.0,
                          max_spread_st: float = 2.0,
                          subject_id: Optional[str] = None,
                          cache_path: Optional[str] = None,
                          recalibrate: bool = False,
                          on_ambiguous: str = "as_measured",
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Derive ONE pitch floor/ceiling for a whole subject from the audio itself.

    Every file is probed with a wide-range tracker, each probe is octave-checked
    against the harmonic comb, and the SUBJECT median of the corrected values
    sets the range:  floor = median/margin, ceiling = median*margin - the same
    rule estimate_f0_range() uses per file, but computed once over all sessions
    so the setting is identical for every take and cannot drift between them.

    cache_path: JSON written on the first run and re-read afterwards, so later
        runs reproduce the earlier numbers exactly instead of re-deriving them
        from whichever files happen to be in the folder. Point it at the
        SUBJECT's folder, not the session's. recalibrate=True overwrites it.

    on_ambiguous: what to do when the comb test cannot settle the octave for the
        majority of files (real period doubling / diplophonia does this).
          'as_measured' (default) keep the tracked value, flag confidence='check'
          'upper'                 assume the tracked value is the subharmonic
                                  and use 2x it
          'lower'                 assume the tracked value is the doubled one
        There is no correct default here. Listen to one file, then set it.

    Returns a dict with pitch_range, the per-file evidence, and confidence in
    {'high','check','failed'}. confidence != 'high' means READ THE REPORT before
    using the numbers - the range is still returned, because a flagged range you
    can see is safer than a hand-typed one you cannot.
    """
    files = [str(f) for f in wav_files]
    subject = subject_id or (_subject_id_from_path(files[0]) if files else "unknown")
    if cache_path and not recalibrate:
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if cached.get("pitch_range"):
                if verbose:
                    print(f"\n  Pitch range read from calibration file "
                          f"{cache_path}: {cached['pitch_range'][0]:.0f}-"
                          f"{cached['pitch_range'][1]:.0f} Hz "
                          f"(subject {cached.get('subject_id', '?')}, "
                          f"confidence {cached.get('confidence', '?')}, "
                          f"calibrated {cached.get('calibrated_on', '?')} on "
                          f"{cached.get('n_files_used', '?')} file(s))")
                    print("    Delete the file or pass recalibrate=True to derive it again.")
                cached["source"] = "cache"
                return cached
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"    ! calibration cache unreadable ({e}); re-deriving")

    analyzer = CompleteSpeechAnalyzer(task=TASK_SUSTAINED, octave_check=True)
    if verbose:
        print(f"\nCALIBRATING PITCH RANGE for subject '{subject}' "
              f"over {len(files)} file(s) - wide probe {wide_floor:.0f}-"
              f"{wide_ceiling:.0f} Hz, then harmonic-comb octave check")
    probes = []
    for f in files:
        if not Path(f).exists():
            continue
        p = probe_file_f0(analyzer, f, wide_floor, wide_ceiling)
        probes.append(p)
        if verbose and p["ok"]:
            print(f"    {Path(f).name:42s} {p['median_hz']:6.1f} Hz "
                  f"(voiced {p['voiced_fraction']*100:3.0f}%)  "
                  f"comb: {p['verdict'] or 'n/a'}"
                  + (f" -> {p['corrected_hz']:.0f} Hz"
                     if abs(p["factor"] - 1.0) > 1e-9 else ""))
        elif verbose:
            print(f"    {Path(f).name:42s} probe failed: {p['note']}")

    usable = [p for p in probes
              if p["ok"] and np.isfinite(p["corrected_hz"]) and p["corrected_hz"] > 0
              and p["voiced_fraction"] >= min_voiced_fraction]
    relaxed = False
    if len(usable) < max(2, int(0.2 * max(1, len(probes)))):
        usable = [p for p in probes if p["ok"] and np.isfinite(p["corrected_hz"])
                  and p["corrected_hz"] > 0]
        relaxed = True
    out: Dict[str, Any] = dict(
        subject_id=subject, source="calibrated", pitch_range=None,
        median_hz=float("nan"), spread_st=float("nan"),
        n_files_probed=len(probes), n_files_used=len(usable),
        verdicts={}, confidence="failed", warnings=[], notes=[], report="",
        calibrated_on=datetime.now().strftime("%Y-%m-%d %H:%M"),
        on_ambiguous=on_ambiguous, margin=float(margin),
        per_file=[{k: p[k] for k in ("file", "median_hz", "corrected_hz",
                                     "voiced_fraction", "verdict",
                                     "score_ref_db", "score_half_db")}
                  for p in probes])
    if not usable:
        out["warnings"].append("no file produced a usable F0 probe; range NOT calibrated")
        if verbose:
            print("    !! CALIBRATION FAILED: no usable probe. Check the audio.")
        return out

    cor = np.array([p["corrected_hz"] for p in usable], dtype=float)
    med = float(np.median(cor))
    st = 12.0 * np.log2(cor / med)
    spread = float(np.percentile(st, 90) - np.percentile(st, 10)) if cor.size > 2 \
        else float(np.ptp(st))
    verdicts: Dict[str, int] = {}
    for p in usable:
        verdicts[p["verdict"] or "none"] = verdicts.get(p["verdict"] or "none", 0) + 1
    n_amb = verdicts.get("ambiguous", 0)
    n_inc = verdicts.get("inconclusive", 0)
    n_moved = verdicts.get("tracker_halved", 0) + verdicts.get("tracker_doubled", 0)
    amb_share = float(n_amb + n_inc) / float(len(usable))
    moved_share = float(n_moved) / float(len(usable))

    # the octave question, stated explicitly rather than absorbed into a median
    if amb_share >= 0.5:
        out["warnings"].append(
            f"the comb test could not settle the octave on {n_amb + n_inc} of "
            f"{len(usable)} file(s): this is what genuine period doubling / "
            f"diplophonia looks like, and {med:.0f} Hz and {2 * med:.0f} Hz are "
            f"both defensible readings of the same signal")
        if on_ambiguous == "upper":
            med *= 2.0
            out["warnings"].append("on_ambiguous='upper': using 2x the tracked value")
        elif on_ambiguous == "lower":
            med *= 0.5
            out["warnings"].append("on_ambiguous='lower': using 0.5x the tracked value")
    # The comb OVERRULING the tracker on most files is itself a finding: the
    # calibrated range then sits an octave away from everything the tracker
    # reported, which is too large a change to make silently.
    if moved_share >= 0.5:
        out["warnings"].append(
            f"the comb test moved the octave on {n_moved} of {len(usable)} file(s), so "
            f"the calibrated range is an octave away from what the tracker itself "
            f"reported. This is usually right when it is unanimous, but it rewrites "
            f"every F0 number for this subject: verify one file by ear before trusting it")
    elif n_moved:
        out["notes"].append(
            f"the comb test corrected the octave on {n_moved} of {len(usable)} file(s); "
            f"the pooled value uses the corrected readings")
    if spread > max_spread_st:
        out["warnings"].append(
            f"F0 spread across files is {spread:.1f} st (> {max_spread_st:.1f}): the "
            f"subject's F0 is not stable across sessions, or the files are not all "
            f"the same subject. A single fixed range may not suit all of them")
    # A 2:1 split SURVIVING the comb correction means the octave is unresolved per
    # file, so the pooled median is a mixture of two populations rather than an
    # estimate of one. Detected pairwise, because with a 50/50 split the median
    # lands between the two clusters and is close to neither of them.
    involved = set()
    for i in range(cor.size):
        for j in range(i + 1, cor.size):
            if abs(abs(12.0 * np.log2(cor[i] / cor[j])) - 12.0) < 1.5:
                involved.add(i); involved.add(j)
    if len(involved) >= max(2, int(0.25 * len(usable))):
        out["warnings"].append(
            f"{len(involved)} of {len(usable)} files sit a full octave apart from each "
            f"other AFTER the comb check: the octave is unresolved per file, so the "
            f"pooled median is a mixture of two populations. Verify by ear and consider "
            f"analysing the two groups separately")

    floor = max(float(floor_hard_min), med / float(margin))
    ceiling = min(float(ceiling_hard_max), med * float(margin))
    if ceiling <= floor * 1.4:
        floor = max(float(floor_hard_min), med / 2.0)
        ceiling = min(float(ceiling_hard_max), med * 2.0)
    out.update(pitch_range=[float(round(floor, 1)), float(round(ceiling, 1))],
               median_hz=med, spread_st=spread, verdicts=verdicts,
               confidence=("high" if (not out["warnings"] and not relaxed) else "check"))
    if relaxed:
        out["warnings"].append(
            f"only {len(usable)} file(s) reached the voicing floor of "
            f"{min_voiced_fraction:.2f}; the range rests on weak evidence")

    lines = [f"subject {subject}: F0 {med:.1f} Hz over {len(usable)} of "
             f"{len(probes)} file(s), spread {spread:.1f} st",
             f"  -> pitch_range = ({floor:.0f}, {ceiling:.0f}) Hz  "
             f"[median / {margin:g} .. median * {margin:g}]",
             "  comb verdicts: " + ", ".join(f"{k}={v}" for k, v in verdicts.items())]
    for nt in out["notes"]:
        lines.append(f"  - {nt}")
    for w in out["warnings"]:
        lines.append(f"  ! {w}")
    if amb_share >= 0.5 and on_ambiguous == "as_measured":
        lo2, hi2 = max(floor_hard_min, 2 * med / margin), min(ceiling_hard_max, 2 * med * margin)
        lines.append(f"  The two candidate ranges are ({floor:.0f}, {ceiling:.0f}) Hz "
                     f"for F0 = {med:.0f} Hz and ({lo2:.0f}, {hi2:.0f}) Hz for "
                     f"F0 = {2 * med:.0f} Hz. Listen to one file, then pass "
                     f"on_ambiguous='upper' or set pitch_range by hand once.")
    out["report"] = "\n".join(lines)
    if verbose:
        print("\n" + out["report"])
        print(f"  confidence: {out['confidence'].upper()}")
    if cache_path:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
            if verbose:
                print(f"  calibration saved to {cache_path} - later runs will re-use it "
                      f"verbatim, so the setting stays identical across sessions")
        except Exception as e:
            print(f"  ! could not write the calibration file: {e}")
    return out


def expand_to_wav_files(paths: List[str], recursive: bool = True) -> List[str]:
    """Accept a mix of .wav file paths and folder paths; expand folders."""
    out = []
    for pth in paths:
        p = Path(pth)
        if p.is_dir():
            pattern = "**/*.wav" if recursive else "*.wav"
            found = sorted(str(f) for f in p.glob(pattern))
            if not found:
                print(f"WARNING: no .wav files found in folder: {p}")
            out.extend(found)
        elif p.is_file():
            out.append(str(p))
        else:
            print(f"WARNING: path not found: {p}")
    return out


# =============================================================================
# METRIC EXTRACTION (single source of truth for plots + CSV)
# =============================================================================

# -----------------------------------------------------------------------------
# CANONICAL METRIC SET  (v16)
# -----------------------------------------------------------------------------
# The union over BOTH tasks is exactly 97 metrics - the same count the original
# (v6) script produced - but the members were re-chosen for ROBUSTNESS, because
# a metric that is not computable on a bad day cannot be used to compare days.
#
# What changed, and why:
#
#  * PERTURBATION FAMILIES COLLAPSED TO THEIR MOST-SMOOTHED MEMBER.
#    jitter_ddp == 3 x jitter_rap and shimmer_dda == 3 x shimmer_apq3 BY
#    DEFINITION in Praat, so those two carried no information at all. Of the
#    remainder, the wider the smoothing window the less a single mis-marked
#    pulse moves the result, so PPQ5 (5 periods) and APQ11 (11 periods) are
#    kept, RAP/APQ3/APQ5 dropped. jitter_local and shimmer_local_dB stay because
#    they are the conventional reporting pair and keep the output comparable with
#    the published literature and with the earlier runs of this script.
#
#  * EVERY PERTURBATION MEASURE NOW SHIPS WITH ITS DISPERSION.
#    jitter_iqr, shimmer_iqr, hnr_iqr_db and cpps_iqr_db are the window-to-window
#    IQR behind each median. This is the direct answer to "some days they are not
#    computable": a value whose IQR approaches the value itself is noise, and now
#    you can see that in the same table instead of guessing.
#
#  * THE ROBUST BLOCK IS COMPLETE AND PROMOTED.
#    CPPS needs no pulse train and no periodicity decision - it is a property of
#    the cepstrum - so it survives exactly the rough/breathy/diplophonic voices
#    that break jitter and shimmer, and it is the best-validated single
#    correlate of perceived dysphonia severity. It leads the set, alongside CPP,
#    HNR (+ the per-interval median), the LTAS shape measures (alpha ratio,
#    Hammarberg, tilt) and the always-computable phonatory indices
#    (voiced_fraction, f0_subharmonic_lock, f0_lock_semitones). On a file where
#    jitter/shimmer come back empty, these still produce a comparable number.
#
#  * sv_ TWINS OF GENERIC COLUMNS REMOVED.
#    On a sustained file the generic columns already carry the robust
#    across-window medians, so sv_jitter_local_percent, sv_shimmer_local_percent,
#    sv_hnr_db, sv_cpps_db, sv_f0_mean_hz, sv_intensity_decay_db and the four
#    sv_*_iqr twins were literally the same numbers under a second name. The
#    sv_ prefix is now reserved for what only a sustained vowel HAS: MPT, token
#    structure, steady-state F0 behaviour, tremor, and the token-level
#    (periodicity-free) fallbacks.
#
#  * DERIVABLE / MARGINAL COLUMNS DROPPED (6).
#    rpvi and varco_v (interval proxies without real V/C segmentation - nPVI
#    kept as the one representative), estimated_syllable_count (superseded by
#    syllable_count_used), f4_mean_hz (LPC F4 is not reliably estimable at these
#    ceilings), f2_f1_ratio (algebraic function of two retained columns),
#    num_periods (num_pulses minus the interval count).
#
#  * PROVENANCE MOVED OUT OF THE METRIC TABLE, NOT DELETED (see AUDIT_COLUMNS).
#    The settings-actually-used, octave verdict, window-rejection counts,
#    usability gate and recording-property columns are the reason v8/v9 can be
#    trusted at all, but they are not measurements of the voice and they should
#    not be counted as metrics or drawn as bar plots. They now go to their own
#    clearly separated block in the CSV.
#
# NOTE ON READING THE COUNT (corrected in v19): the union across tasks is 103.
# Any single file emits the 60 both-task metrics plus the 22-column block for its
# own task, so a sustained file carries 82 and a reading file 81; the other
# task's block is blank by design. The old note said 73 and 78, which had not
# been true since v16 moved the provenance columns into AUDIT_COLUMNS.
#
# FOUR OF THE 60 'BOTH' COLUMNS ARE SUSTAINED-ONLY IN PRACTICE: cpps_iqr_db,
# hnr_iqr_db, jitter_iqr and shimmer_iqr are window-to-window dispersions, and
# only the sustained path measures in windows. They are structurally blank on a
# reading file. They stay in METRICS_BOTH so the CSV keeps one fixed shape, but
# do not read an empty cell there as a failed measurement.

_M_BOTH_CONTEXT = [
    "duration_total_s", "duration_speech_s", "articulation_time_s", "phonation_time_s",
]
_M_BOTH_INTENSITY = [
    "intensity_active_mean_db", "intensity_active_std_db", "intensity_nucleus_std_db",
    "intensity_range_db", "intensity_decay_db",
]
_M_BOTH_PITCH = [
    "pitch_mean_hz", "pitch_median_hz", "pitch_std_hz", "pitch_min_hz", "pitch_max_hz",
    "pitch_range_hz", "pitch_range_semitones", "pitch_cv",
    "voiced_percent", "unvoiced_percent",
]
_M_BOTH_VQ_ROBUST = [
    "cpps_db", "cpps_iqr_db", "cpp_db",
    "hnr_db", "hnr_iqr_db", "hnr_median_perinterval_db", "nhr", "mean_autocorrelation",
    "voice_breaks_count", "voice_breaks_degree_percent",
]
_M_BOTH_VQ_PERTURB = [
    "jitter_local_percent", "jitter_ppq5_percent", "jitter_iqr",
    "shimmer_local_db", "shimmer_apq11_percent", "shimmer_iqr",
    "num_pulses",
]
_M_BOTH_SPECTRAL = [
    "spectral_centroid_hz", "spectral_spread_hz", "spectral_skewness",
    "spectral_kurtosis", "spectral_tilt_db_per_khz", "alpha_ratio_db",
    "hammarberg_index",
]
_M_BOTH_FORMANT = [
    "f1_mean_hz", "f1_std_hz", "f2_mean_hz", "f2_std_hz", "f3_mean_hz", "f3_std_hz",
]
_M_BOTH_STABILITY = [
    "voiced_fraction", "f0_subharmonic_lock", "f0_lock_semitones",
]
_M_BOTH_PAUSE = [
    "pause_count", "pauses_per_minute", "pause_total_duration_s",
    "pause_mean_duration_s", "pause_ratio_to_speech",
]
_M_BOTH_REC = [
    "rec_quality_score", "rec_snr_db", "rec_clipping_fraction",
]
_M_READING_ONLY = [
    "speech_rate_syll_per_s", "articulation_rate_syll_per_s", "syllable_count_used",
    "speaking_time_fraction",
    "filler_count", "fillers_per_minute", "filler_total_duration_s",
    "filler_ratio_to_speech",
    "ems_3_8hz_ratio", "ems_peak_freq_hz", "ems_syllabic_to_slow_ratio", "npvi",
    "vowel_space_area", "vowel_dispersion_logarea", "vowel_cloud_spread_hz",
    "fluency_intensity_stability", "fluency_pitch_stability",
    "fluency_rhythm_regularity", "fluency_voice_quality", "fluency_articulation",
    "fluency_overall",
]
_M_SUSTAINED_ONLY = [
    "sv_mpt_longest_s", "sv_mpt_longest_uninterrupted_s", "sv_mpt_mean_s",
    "sv_total_phonation_s", "sv_n_tokens",
    "sv_f0_sd_semitones", "sv_f0_drift_st_per_s", "sv_intensity_sd_db",
    "sv_tremor_rate_hz", "sv_tremor_extent_st",
    "sv_cpps_token_db", "sv_hnr_token_db", "sv_window_yield",
    # v19: duration-free loudness fade across the longest token. The paired
    # intensity_decay_db is dB across the analysed span and therefore scales
    # with take length, which for sustained phonation varies by design.
    "sv_intensity_decay_db_per_s",
    # ---- v18: how much of the vowel is behind the numbers ------------------
    "sv_analyzed_fraction", "sv_window_yield_s",
    "sv_steady_frame_fraction", "sv_longest_steady_run_s",
    "sv_f1f2_cloud_spread_hz",
    "sv_jitter_sd_across_tokens", "sv_shimmer_sd_across_tokens",
    "sv_hnr_sd_across_tokens",
]

METRICS_BOTH = (_M_BOTH_CONTEXT + _M_BOTH_INTENSITY + _M_BOTH_PITCH
                + _M_BOTH_VQ_ROBUST + _M_BOTH_VQ_PERTURB + _M_BOTH_SPECTRAL
                + _M_BOTH_FORMANT + _M_BOTH_STABILITY + _M_BOTH_PAUSE + _M_BOTH_REC)
METRICS_READING_ONLY = list(_M_READING_ONLY)
METRICS_SUSTAINED_ONLY = list(_M_SUSTAINED_ONLY)
CANONICAL_METRICS = METRICS_BOTH + METRICS_READING_ONLY + METRICS_SUSTAINED_ONLY

# Provenance / QC. Written to the CSV in a separate block, never plotted, never
# counted as a metric, never aggregated as if it were a measurement.
AUDIT_COLUMNS = [
    "task_is_sustained_vowel", "one_production_per_file", "vq_measured", "cpps_is_praat",
    "syllable_source_known", "syllable_count_estimated", "syllable_count_agreement",
    "formant_track_ok", "formant_ceiling_hz_used",
    "pitch_octave_repair_fraction",
    "pitch_floor_used_hz", "pitch_ceiling_used_hz", "silence_threshold_db_used",
    "f0_octave_ok", "f0_octave_corrected",
    "f0_octave_score_ref_db", "f0_octave_score_half_db", "f0_wide_median_hz",
    "sv_measured", "sv_usable", "sv_windows_usable", "sv_perturbation_usable",
    "sv_signal_type",
    "sv_n_windows_valid", "sv_n_windows_total", "sv_measured_total_s",
    "sv_n_tokens_expected", "sv_token_count_mismatch", "sv_boundaries_from_file",
    "sv_n_splices_detected", "sv_analysis_window_s",
    "sv_rej_low_voicing", "sv_rej_f0_outlier", "sv_rej_f0_step",
    "sv_rej_splice", "sv_rej_other",
    # ---- v18: where the discarded phonation went ---------------------------
    "sv_analyzable_total_s", "sv_discard_unsteady_s", "sv_discard_edge_s",
    "sv_discard_quantisation_s", "sv_discard_onset_s", "sv_discard_interior_s",
    "sv_discard_offset_s", "sv_discard_dead_token_s", "sv_n_interior_gaps",
    "sv_n_steady_stretches",
    "sv_voiced_at_045", "sv_voiced_at_020", "sv_voiced_low_floor",
    "sv_tremor_peak_ratio", "sv_f1f2_dispersion_logarea",
    # ---- v19: the unbroken-phonation ledger --------------------------------
    # sv_mpt_longest_uninterrupted_s is the measurement; these say how it was
    # obtained and what the rest of the take looked like. A take with the same
    # unbroken duration but six breaks is a different finding from one with
    # none, and only these columns separate them.
    "sv_n_phonation_breaks", "sv_n_breaks_in_best_token",
    "sv_unbroken_start_s", "sv_unbroken_end_s", "sv_unbroken_token_index",
    "sv_unbroken_energy_only_s", "sv_unbroken_voicing_required",
    "sv_break_threshold_db", "sv_break_min_duration_s",
    # ---- v19: provenance of the spectral / formant blocks ------------------
    "cpp_is_praat_cpps", "spectral_span_intervals", "formant_span_windows",
    "spectral_tilt_band_lo_hz", "spectral_tilt_band_hi_hz",
    "pitch_block_from_windows",
    "rec_effective_bandwidth_hz", "rec_spectral_edge_hz", "rec_sample_rate_hz",
    "rec_is_bandlimited", "rec_vq_reliable", "rec_snr_estimable",
    "rec_digital_silence",
]

assert len(CANONICAL_METRICS) == len(set(CANONICAL_METRICS)), "duplicate metric name"
# v18: 97 + the four coverage metrics (sv_analyzed_fraction, sv_window_yield_s,
# sv_steady_frame_fraction, sv_longest_steady_run_s). They are measurements of
# the VOICE - how much of the phonation was steady enough to measure - not
# provenance, so they belong here and not in AUDIT_COLUMNS. The v6 parity count
# is recorded in the message for continuity.
assert len(CANONICAL_METRICS) == 103, (
    f"canonical metric set must be 103 (97 v6-parity + 4 v18 coverage + "
    f"1 uninterrupted MPT + 1 v19 decay rate), "
    f"got {len(CANONICAL_METRICS)}")


def _raw_metric_table(result: 'SpeechAnalysisResult') -> Dict[str, float]:
    """
    Every number the analyser produced, under its canonical name, before the
    task gate is applied. Kept separate from get_metric_table() so the gating
    rules are readable in one place.
    """
    p = result.pause_metrics; f = result.filler_metrics; i = result.intensity_metrics
    pi = result.pitch_metrics; vq = result.voice_quality_metrics; fm = result.formant_metrics
    sp = result.spectral_metrics; rh = result.rhythm_metrics; fl = result.fluency_metrics
    rq = result.recording_quality; rd = result.reading_metrics
    sv = result.sustained_metrics
    is_sust = (result.task == TASK_SUSTAINED)
    st = result.analysis_settings

    def _g(key, default=np.nan):
        v = st.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    # v17 - DO NOT EXPORT PERTURBATION FROM THE WHOLE-FILE FALLBACK.
    # In analyze(), when the sustained analysis finds ZERO valid steady-state
    # windows, voice_quality_metrics falls back to the whole-file Praat Voice
    # Report. The console correctly prints "Jitter/Shimmer/HNR: NOT MEASURABLE",
    # but the fallback values were still landing in the CSV - so on a 22%-voiced,
    # 3.3 dB-HNR signal the table carried a jitter, a shimmer, an NHR and an
    # autocorrelation that are tracker noise over the few frames the tracker
    # happened to accept. That is the precise failure this whole design exists to
    # prevent: it makes the metric look computable on exactly the days it is not,
    # which is what makes cross-day comparison unsound.
    #
    # On a sustained file, cycle-based measures are exported ONLY when the
    # windowed analysis actually measured them. Everything that does not need a
    # pulse train (CPPS, CPP, token HNR, spectral shape, voiced fraction) is
    # unaffected and still exported - an unmeasurable voice remains a FINDING with
    # numbers attached, not an empty row.
    _pert_ok = (not is_sust) or bool(sv.measured)

    def _pert(v):
        """Cycle-based measure: blanked on a sustained file with no valid window."""
        return v if _pert_ok else np.nan

    # v19: is a window-restricted F0 distribution available? All seven pitch
    # columns switch on this together (see the pitch block below).
    _win_pitch = bool(is_sust and np.isfinite(sv.f0_mean_hz)
                      and np.isfinite(sv.f0_median_hz))

    t = {
        # ---- context -------------------------------------------------------
        "duration_total_s": result.duration_total,
        "duration_speech_s": result.duration_speech,
        "articulation_time_s": rh.articulation_time_sec,
        "phonation_time_s": rh.phonation_time_sec,
        # ---- intensity -----------------------------------------------------
        "intensity_active_mean_db": i.active_mean_db,
        "intensity_active_std_db": i.active_std_db,
        "intensity_nucleus_std_db": i.nucleus_std_db,
        "intensity_range_db": i.range_db,
        "intensity_decay_db": sv.intensity_decay_db if is_sust else rd.intensity_decay_db,
        # ---- pitch ---------------------------------------------------------
        # v19: THE WHOLE BLOCK SWITCHES TOGETHER. Only pitch_mean_hz used to be
        # replaced by the window value on a sustained file; the median, SD, min,
        # max, range and CV stayed whole-span. So the exported row mixed two
        # populations - pitch_cv no longer equalled pitch_std_hz / pitch_mean_hz,
        # and pitch_range_semitones spanned every token plus the tracker's
        # excursions at the token edges, i.e. it measured the segmentation
        # rather than the voice. All seven now come from the same valid windows,
        # and pitch_block_from_windows in the audit block records that they did.
        # voiced_percent stays whole-span on purpose: inside a valid window it
        # would be ~100% by construction (windows require
        # window_min_voiced_fraction), so it would carry no information.
        "pitch_mean_hz": (sv.f0_mean_hz if (is_sust and _win_pitch)
                          else pi.mean_hz),
        "pitch_median_hz": sv.f0_median_hz if (is_sust and _win_pitch) else pi.median_hz,
        "pitch_std_hz": sv.f0_std_hz if (is_sust and _win_pitch) else pi.std_hz,
        "pitch_min_hz": sv.f0_min_hz if (is_sust and _win_pitch) else pi.min_hz,
        "pitch_max_hz": sv.f0_max_hz if (is_sust and _win_pitch) else pi.max_hz,
        "pitch_range_hz": ((sv.f0_max_hz - sv.f0_min_hz) if (is_sust and _win_pitch)
                           else pi.range_hz),
        "pitch_range_semitones": (sv.f0_range_semitones if (is_sust and _win_pitch)
                                  else pi.range_semitones),
        "pitch_cv": sv.f0_cv if (is_sust and _win_pitch) else pi.coefficient_of_variation,
        "voiced_percent": pi.voiced_frames_percent,
        "unvoiced_percent": pi.unvoiced_frames_percent,
        # ---- voice quality: ROBUST block ------------------------------------
        # CPPS leads: no pulse train, no periodicity decision, so it survives
        # the voices that break jitter/shimmer.
        "cpps_db": sv.cpps_db if is_sust else rd.cpps_db,
        "cpps_iqr_db": sv.cpps_iqr,
        "cpp_db": sp.cpp_db,
        "hnr_db": _pert(vq.hnr_db),
        "hnr_iqr_db": sv.hnr_iqr,
        "hnr_median_perinterval_db": (sv.hnr_token_db if is_sust
                                      else rd.hnr_median_perinterval_db),
        "nhr": _pert(vq.nhr),
        "mean_autocorrelation": _pert(vq.mean_autocorrelation),
        "voice_breaks_count": _pert(vq.voice_breaks_count),
        "voice_breaks_degree_percent": _pert(vq.voice_breaks_degree_percent),
        # ---- voice quality: PERTURBATION (most-smoothed members only) -------
        "jitter_local_percent": _pert(vq.jitter_local_percent),
        "jitter_ppq5_percent": _pert(vq.jitter_ppq5_percent),
        "jitter_iqr": sv.jitter_iqr,
        "shimmer_local_db": _pert(vq.shimmer_local_db),
        "shimmer_apq11_percent": _pert(vq.shimmer_apq11_percent),
        "shimmer_iqr": sv.shimmer_iqr,
        "num_pulses": _pert(vq.num_pulses),
        # ---- spectral ------------------------------------------------------
        "spectral_centroid_hz": sp.spectral_centroid_mean_hz,
        "spectral_spread_hz": sp.spectral_spread_mean_hz,
        "spectral_skewness": sp.spectral_skewness_mean,
        "spectral_kurtosis": sp.spectral_kurtosis_mean,
        "spectral_tilt_db_per_khz": sp.spectral_tilt_db,
        "alpha_ratio_db": sp.alpha_ratio,
        "hammarberg_index": sp.hammarberg_index,
        # ---- formants ------------------------------------------------------
        "f1_mean_hz": fm.f1_mean_hz, "f1_std_hz": fm.f1_std_hz,
        "f2_mean_hz": fm.f2_mean_hz, "f2_std_hz": fm.f2_std_hz,
        "f3_mean_hz": fm.f3_mean_hz, "f3_std_hz": fm.f3_std_hz,
        # ---- always-computable phonatory indices ---------------------------
        # These are the ones to fall back on when perturbation analysis returns
        # nothing: an unmeasurable voice is a finding, not missing data.
        "voiced_fraction": (sv.voiced_fraction if (is_sust and np.isfinite(sv.voiced_fraction))
                            else (pi.voiced_frames_percent / 100.0
                                  if np.isfinite(pi.voiced_frames_percent) else np.nan)),
        "f0_subharmonic_lock": _g("f0_subharm_lock"),
        "f0_lock_semitones": _g("f0_lock_st"),
        # ---- pause / breath structure --------------------------------------
        "pause_count": p.count, "pauses_per_minute": p.pauses_per_minute,
        "pause_total_duration_s": p.total_duration,
        "pause_mean_duration_s": p.avg_duration,
        "pause_ratio_to_speech": p.ratio_to_speech,
        # ---- recording -----------------------------------------------------
        "rec_quality_score": rq.quality_score, "rec_snr_db": rq.snr_db,
        "rec_clipping_fraction": rq.clipping_fraction,
        # ---- reading-only --------------------------------------------------
        "speech_rate_syll_per_s": rh.speech_rate_syllables_per_sec,
        "articulation_rate_syll_per_s": rh.articulation_rate_syllables_per_sec,
        "syllable_count_used": rd.syllable_count_used,
        "speaking_time_fraction": rd.speaking_time_fraction,
        "filler_count": f.count, "fillers_per_minute": f.fillers_per_minute,
        "filler_total_duration_s": f.total_duration,
        "filler_ratio_to_speech": f.ratio_to_speech,
        "ems_3_8hz_ratio": rd.ems_3_8hz_ratio,
        "ems_peak_freq_hz": rd.ems_peak_freq_hz,
        "ems_syllabic_to_slow_ratio": rd.ems_4_to_lowband_ratio,
        "npvi": rh.npvi_v,
        "vowel_space_area": fm.vowel_space_area,
        "vowel_dispersion_logarea": rd.formant_dispersion_logarea,
        "vowel_cloud_spread_hz": rd.f1f2_cloud_spread,
        "fluency_intensity_stability": fl.intensity_stability,
        "fluency_pitch_stability": fl.pitch_stability,
        "fluency_rhythm_regularity": fl.rhythm_regularity,
        "fluency_voice_quality": fl.voice_quality_score,
        "fluency_articulation": fl.articulation_score,
        "fluency_overall": fl.overall_fluency,
        # ---- sustained-only ------------------------------------------------
        "sv_mpt_longest_s": sv.mpt_longest_s,
        "sv_mpt_longest_uninterrupted_s": sv.mpt_longest_uninterrupted_s,
        "sv_mpt_mean_s": sv.mpt_mean_s,
        "sv_total_phonation_s": sv.total_phonation_s, "sv_n_tokens": sv.n_tokens,
        "sv_f0_sd_semitones": sv.f0_sd_semitones,
        "sv_f0_drift_st_per_s": sv.f0_drift_st_per_s,
        "sv_intensity_sd_db": sv.intensity_sd_db,
        "sv_tremor_rate_hz": sv.tremor_rate_hz,
        "sv_tremor_extent_st": sv.tremor_extent_semitones,
        "sv_cpps_token_db": sv.cpps_token_db, "sv_hnr_token_db": sv.hnr_token_db,
        "sv_window_yield": sv.window_yield,
        "sv_intensity_decay_db_per_s": sv.intensity_decay_db_per_s,
        # ---- v18 coverage ---------------------------------------------------
        "sv_analyzed_fraction": sv.analyzed_fraction,
        "sv_window_yield_s": sv.window_yield_s,
        "sv_steady_frame_fraction": sv.steady_frame_fraction,
        "sv_longest_steady_run_s": sv.longest_steady_run_s,
        "sv_f1f2_cloud_spread_hz": sv.f1f2_cloud_spread_hz,
        "sv_jitter_sd_across_tokens": sv.jitter_sd_across_tokens,
        "sv_shimmer_sd_across_tokens": sv.shimmer_sd_across_tokens,
        "sv_hnr_sd_across_tokens": sv.hnr_sd_across_tokens,
    }
    return t


def get_audit_table(result: 'SpeechAnalysisResult') -> Dict[str, float]:
    """
    Provenance and quality-control columns: how the file was measured and
    whether the measurement can be trusted. Deliberately NOT part of the metric
    set - none of these is a property of the voice, and treating them as metrics
    is how a settings difference gets reported as a treatment effect.
    """
    vq = result.voice_quality_metrics; fm = result.formant_metrics
    pi = result.pitch_metrics; rq = result.recording_quality
    rd = result.reading_metrics; sv = result.sustained_metrics
    sp_ = result.spectral_metrics
    is_sust = (result.task == TASK_SUSTAINED)
    st = result.analysis_settings

    def _g(key, default=np.nan):
        v = st.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    def _sv(v):
        return v if is_sust else np.nan

    t = {
        "task_is_sustained_vowel": 1.0 if is_sust else 0.0,
        "one_production_per_file": _g("one_production_per_file", 0.0),
        "vq_measured": 1.0 if getattr(vq, "measured", False) else 0.0,
        "cpps_is_praat": 1.0 if (sv.cpps_source if is_sust
                                 else rd.cpps_source).startswith("praat") else 0.0,
        "syllable_source_known": (np.nan if is_sust else
                                  (1.0 if rd.syllable_source == "known_passage" else 0.0)),
        "syllable_count_estimated": np.nan if is_sust else rd.syllable_count_estimated,
        "syllable_count_agreement": np.nan if is_sust else rd.syllable_count_agreement,
        "formant_track_ok": 1.0 if fm.track_ok else 0.0,
        "formant_ceiling_hz_used": fm.max_formant_hz_used,
        "pitch_octave_repair_fraction": pi.octave_repair_fraction,
        "pitch_floor_used_hz": _g("pitch_floor_used"),
        "pitch_ceiling_used_hz": _g("pitch_ceiling_used"),
        "silence_threshold_db_used": _g("silence_threshold_db_used"),
        "f0_octave_ok": {"ok": 1.0, "ambiguous": 0.5}.get(
            str(st.get("octave_verdict", "")), 0.0),
        "f0_octave_corrected": 0.0 if _g("octave_factor", 1.0) == 1.0 else 1.0,
        "f0_octave_score_ref_db": _g("octave_score_ref_db"),
        "f0_octave_score_half_db": _g("octave_score_half_db"),
        "f0_wide_median_hz": _g("f0_wide_median"),
        "sv_measured": _sv(1.0 if sv.measured else 0.0),
        "sv_usable": _sv(1.0 if sv.usable else 0.0),
        "sv_windows_usable": _sv(1.0 if getattr(sv, "windows_usable", False) else 0.0),
        "sv_perturbation_usable": _sv(
            1.0 if getattr(sv, "perturbation_usable", False) else 0.0),
        "sv_signal_type": _sv(sv.signal_type),
        "sv_n_windows_valid": _sv(sv.n_windows_valid),
        "sv_n_windows_total": _sv(sv.n_windows_total),
        "sv_measured_total_s": _sv(sv.measured_total_s),
        "sv_n_tokens_expected": _sv(sv.n_tokens_expected),
        "sv_token_count_mismatch": _sv(1.0 if sv.token_count_mismatch else 0.0),
        "sv_boundaries_from_file": _sv(1.0 if sv.boundaries_source == "file" else 0.0),
        "sv_n_splices_detected": _sv(sv.n_splices_detected),
        "sv_analysis_window_s": _sv(sv.analysis_window_s),
        "sv_rej_low_voicing": _sv(sv.rej_low_voicing),
        "sv_rej_f0_outlier": _sv(sv.rej_f0_outlier),
        "sv_rej_f0_step": _sv(sv.rej_f0_step),
        "sv_rej_splice": _sv(sv.rej_splice),
        "sv_rej_other": _sv(sv.rej_other),
        "sv_analyzable_total_s": _sv(sv.analyzable_total_s),
        "sv_discard_unsteady_s": _sv(sv.discard_unsteady_s),
        "sv_discard_edge_s": _sv(sv.discard_edge_s),
        "sv_discard_quantisation_s": _sv(sv.discard_quantisation_s),
        "sv_discard_onset_s": _sv(sv.discard_onset_s),
        "sv_discard_interior_s": _sv(sv.discard_interior_s),
        "sv_discard_offset_s": _sv(sv.discard_offset_s),
        "sv_discard_dead_token_s": _sv(sv.discard_dead_token_s),
        "sv_n_interior_gaps": _sv(sv.n_interior_gaps),
        "sv_n_steady_stretches": _sv(sv.n_steady_stretches),
        "sv_voiced_at_045": _sv(sv.voiced_at_045),
        "sv_voiced_at_020": _sv(sv.voiced_at_020),
        "sv_voiced_low_floor": _sv(sv.voiced_low_floor),
        "sv_tremor_peak_ratio": _sv(sv.tremor_peak_ratio),
        "sv_f1f2_dispersion_logarea": _sv(sv.f1f2_dispersion_logarea),
        # ---- v19 unbroken-phonation ledger ---------------------------------
        "sv_n_phonation_breaks": _sv(sv.n_phonation_breaks),
        "sv_n_breaks_in_best_token": _sv(sv.n_breaks_in_best_token),
        "sv_unbroken_start_s": _sv(sv.unbroken_start_s),
        "sv_unbroken_end_s": _sv(sv.unbroken_end_s),
        "sv_unbroken_token_index": _sv(sv.unbroken_token_index),
        "sv_unbroken_energy_only_s": _sv(sv.unbroken_energy_only_s),
        "sv_unbroken_voicing_required": _sv(
            1.0 if str(sv.unbroken_source).startswith("voiced") else 0.0),
        "sv_break_threshold_db": _sv(sv.break_threshold_db),
        "sv_break_min_duration_s": _sv(_g("phonation_break_min_s")),
        # ---- v19 provenance of the spectral / formant / pitch blocks -------
        "cpp_is_praat_cpps": 1.0 if str(
            getattr(sp_, "cpp_source", "")).startswith("praat") else 0.0,
        "spectral_span_intervals": 1.0 if "interval" in str(
            getattr(sp_, "span_source", "")) else 0.0,
        "formant_span_windows": _sv(1.0 if sv.measured_spans_s else 0.0),
        "spectral_tilt_band_lo_hz": float(getattr(sp_, "tilt_band_hz",
                                                  (np.nan, np.nan))[0]),
        "spectral_tilt_band_hi_hz": float(getattr(sp_, "tilt_band_hz",
                                                  (np.nan, np.nan))[1]),
        "pitch_block_from_windows": _sv(
            1.0 if np.isfinite(sv.f0_median_hz) else 0.0),
        "rec_effective_bandwidth_hz": rq.effective_bandwidth_hz,
        "rec_spectral_edge_hz": rq.spectral_edge_hz,
        "rec_sample_rate_hz": rq.sample_rate_hz,
        "rec_is_bandlimited": 1.0 if rq.is_bandlimited else 0.0,
        "rec_vq_reliable": 1.0 if rq.vq_reliable else 0.0,
        "rec_snr_estimable": 0.0 if not np.isfinite(
            float(rq.snr_db) if rq.snr_db is not None else np.nan) else 1.0,
        "rec_digital_silence": 1.0 if st.get("digital_silence") else 0.0,
    }

    def _num(x):
        try:
            return float(x) if x is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    return {k: _num(t.get(k, np.nan)) for k in AUDIT_COLUMNS}


def get_metric_table(result: SpeechAnalysisResult) -> Dict[str, float]:
    """
    The 97 canonical metrics, task-gated. Column order and membership are
    identical for every file, so the CSV is a rectangle; metrics that cannot
    mean anything for this file's task are NaN (blank in the CSV, skipped in the
    plots) rather than a number that would silently correlate an artifact with
    stimulation condition.
    """
    is_sust = (result.task == TASK_SUSTAINED)
    raw = _raw_metric_table(result)

    def _num(x):
        try:
            return float(x) if x is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    out = {}
    for k in CANONICAL_METRICS:
        if is_sust and k in METRICS_READING_ONLY:
            out[k] = np.nan
        elif (not is_sust) and k in METRICS_SUSTAINED_ONLY:
            out[k] = np.nan
        else:
            out[k] = _num(raw.get(k, np.nan))
    return out


# =============================================================================
# PER-METRIC DIRECTIONAL GUIDANCE
# =============================================================================
METRIC_DIRECTION: Dict[str, Tuple[str, str]] = {
    "duration_total_s": ("neutral", "recording length; context only"),
    "duration_speech_s": ("neutral", "detected speech length; context only"),
    "articulation_time_s": ("neutral", "speaking time excluding pauses; context only"),
    "phonation_time_s": ("neutral", "voiced time; context only"),
    "pause_count": ("lower", "fewer pauses = more fluent"),
    "pauses_per_minute": ("lower", "fewer pauses/min = more fluent"),
    "pause_total_duration_s": ("lower", "less total pause time = more fluent"),
    "pause_mean_duration_s": ("lower", "shorter pauses = more fluent"),
    "pause_ratio_to_speech": ("lower", "less pause relative to speech = more fluent"),
    "filler_count": ("lower", "fewer fillers = more fluent"),
    "fillers_per_minute": ("lower", "fewer fillers/min = more fluent"),
    "filler_total_duration_s": ("lower", "less filler time = more fluent"),
    "filler_ratio_to_speech": ("lower", "less filler relative to speech = more fluent"),
    "intensity_range_db": ("around", "moderate range is normal; extreme values atypical"),
    "task_is_sustained_vowel": ("neutral", "1 = sustained vowel, 0 = connected speech"),
    "intensity_active_mean_db": ("neutral", "UNCALIBRATED active level; within-setup only"),
    "intensity_active_std_db": ("around", "lower = steadier loudness; some variation is normal"),
    "intensity_nucleus_std_db": ("lower", "lower = better intensity CONTROL on stressed vowels"),
    "pitch_mean_hz": ("neutral", "speaker characteristic (sex/age); not better/worse"),
    "pitch_median_hz": ("neutral", "speaker characteristic; not better/worse"),
    "pitch_std_hz": ("around", "some F0 variation = natural prosody; near-zero = monotone"),
    "pitch_min_hz": ("neutral", "context only"), "pitch_max_hz": ("neutral", "context only"),
    "pitch_range_hz": ("around", "moderate range healthy; very small = monotone"),
    "pitch_cv": ("around", "natural read-speech F0 CV is moderate; very low = monotone"),
    "pitch_range_semitones": ("around", "moderate semitone range = expressive"),
    "pitch_octave_repair_fraction": ("lower", "fraction of frames corrected for octave errors; high value = check F0 range settings"),
    "voiced_percent": ("higher", "more voicing = more continuous phonation (within reason)"),
    "unvoiced_percent": ("lower", "less unvoiced = more continuous phonation"),
    "jitter_local_percent": ("lower", "lower = more stable pitch periods"),
    "jitter_ppq5_percent": ("lower", "lower = more stable pitch periods"),
    "shimmer_local_db": ("lower", "lower = more stable amplitude"),
    "shimmer_apq11_percent": ("lower", "lower = more stable amplitude"),
    "hnr_db": ("higher", "higher = clearer, less noisy voice"),
    "nhr": ("lower", "lower noise-to-harmonics = clearer voice"),
    "mean_autocorrelation": ("higher", "higher = more periodic/clean phonation"),
    "voice_breaks_count": ("lower", "fewer genuine phonation breaks = steadier voice"),
    "voice_breaks_degree_percent": ("lower", "less time in breaks = steadier voice"),
    "num_pulses": ("neutral", "depends on voiced duration; sanity-check only"),
    "vq_measured": ("higher", "1 = voice-quality actually measured"),
    "f1_mean_hz": ("neutral", "vowel/speaker dependent; not better/worse"),
    "f1_std_hz": ("lower", "lower = more stable articulation"),
    "f2_mean_hz": ("neutral", "vowel/speaker dependent; not better/worse"),
    "f2_std_hz": ("lower", "lower = more stable articulation"),
    "f3_mean_hz": ("neutral", "speaker dependent; not better/worse"),
    "f3_std_hz": ("lower", "lower = more stable articulation"),
    "vowel_space_area": ("higher", "larger vowel space = clearer articulation (reduced in dysarthria)"),
    "formant_track_ok": ("higher", "1 = F3 above F2 (valid track); 0 = LPC failure, F3/F4 discarded"),
    "formant_ceiling_hz_used": ("neutral", "formant ceiling chosen from the speaker's F0; context only"),
    "spectral_centroid_hz": ("neutral", "timbre/brightness; speaker & content dependent"),
    "spectral_spread_hz": ("neutral", "spectral width; context only"),
    "spectral_skewness": ("neutral", "spectral shape; context only"),
    "spectral_kurtosis": ("neutral", "spectral shape; context only"),
    "spectral_tilt_db_per_khz": ("neutral", "vocal effort/phonation type; within-subject"),
    "alpha_ratio_db": ("neutral", "vocal effort/brightness; within-subject"),
    "hammarberg_index": ("neutral", "vocal effort; within-subject"),
    "cpp_db": ("higher", "higher CPP = clearer, less breathy voice"),
    "speech_rate_syll_per_s": ("around", "typical read speech mid range; much slower may indicate impairment"),
    "articulation_rate_syll_per_s": ("around", "typical mid range; very slow may indicate impairment"),
    "npvi": ("around", "onset-interval PVI proxy; mid range for English read speech (within-subject)"),
    # --- NEW reading metrics ---
    "syllable_count_used": ("neutral", "syllables used for rate (exact if known passage)"),
    "syllable_source_known": ("higher", "1 = rate from KNOWN passage count (exact); 0 = estimated"),
    "syllable_count_estimated": ("neutral", "envelope-based syllable estimate, kept as a "
                                            "cross-check on the hand-entered constant; too "
                                            "noisy to use as a rate on its own"),
    "syllable_count_agreement": ("around", "estimate / KNOWN_PASSAGE_SYLLABLES constant; "
                                           "~1.0 = the constant matches the audio, outside "
                                           "0.7-1.4 means one of the two is wrong (check the "
                                           "constant against your passage wording)"),
    "speaking_time_fraction": ("higher", "1.0 = no pauses; lower = more total time lost to PAUSING"),
    "ems_3_8hz_ratio": ("higher", "stronger syllabic (3-8 Hz) envelope modulation = better rhythm/intelligibility; reduced in dysarthria"),
    "ems_peak_freq_hz": ("around", "syllabic peak typically ~4-5 Hz; far from this is atypical"),
    "ems_syllabic_to_slow_ratio": ("higher", "more syllabic vs slow modulation energy = crisper rhythm"),
    "vowel_dispersion_logarea": ("higher", "larger F1-F2 cloud = more articulatory expansion; SHRINKS with hypokinetic centralization"),
    "vowel_cloud_spread_hz": ("higher", "larger spread = more vowel differentiation; reduced in centralization"),
    "intensity_decay_db": ("around", "near 0 = steady loudness; strongly negative = hypokinetic loudness DECAY across passage"),
    "hnr_median_perinterval_db": ("higher", "higher = clearer voice (robust median complement to HNR)"),
    "cpps_db": ("higher", "higher CPPS = clearer, less dysphonic voice (robust severity correlate)"),
    "cpps_is_praat": ("higher", "1 = CPPS from Praat (trusted); 0 = FFT fallback (interpret cautiously)"),
    "fluency_intensity_stability": ("higher", "higher = steadier loudness control (indicative)"),
    "fluency_pitch_stability": ("higher", "higher = healthier prosody (indicative)"),
    "fluency_rhythm_regularity": ("higher", "higher = more typical rhythm (indicative)"),
    "fluency_voice_quality": ("higher", "higher = cleaner voice (indicative)"),
    "fluency_articulation": ("higher", "higher = more typical rate (indicative)"),
    "fluency_overall": ("higher", "higher = more fluent overall (INDICATIVE only, not validated)"),
    # --- sustained-vowel metrics ---
    "sv_n_tokens": ("neutral", "number of separate /a/ productions found in the file"),
    "sv_n_splices_detected": ("lower", "edit points found inside phonation; >0 means the file is cut/concatenated - prefer analysing the original clips separately"),
    "sv_n_windows_valid": ("higher", "steady-state windows that passed validity checks; more = a better-supported median"),
    "sv_n_windows_total": ("neutral", "windows tried before validity checks; context only"),
    "sv_mpt_longest_s": ("higher", "maximum phonation time: longer = better respiratory/phonatory support"),
    "sv_mpt_longest_uninterrupted_s": ("higher",
        "longest stretch of CONTINUOUS PHONATION inside a single token, with no "
        "break (>=50 ms below the phonation threshold or unvoiced) in the middle. "
        "Cannot exceed sv_mpt_longest_s; the gap between the two is how much of "
        "the best token was actually held. Read it with sv_n_breaks_in_best_token "
        "and sv_unbroken_energy_only_s in the audit block."),
    "sv_intensity_decay_db_per_s": ("higher",
        "loudness fade across the longest token in dB PER SECOND (negative = fading). "
        "Unlike intensity_decay_db this does not scale with take length, so it is "
        "the one to compare when phonation duration varies between sessions."),
    "sv_mpt_mean_s": ("higher", "mean token duration; longer = better sustained support"),
    "sv_total_phonation_s": ("neutral", "total phonated time in the file; context only"),
    "sv_analysis_window_s": ("neutral", "length of the steady-state window actually measured; sanity-check only"),
    "sv_f0_sd_semitones": ("lower", "lower = steadier pitch on sustained phonation (typ. < 0.5 st)"),
    "sv_f0_drift_st_per_s": ("around", "near 0 = no drift; large magnitude = failing pitch control across the token"),
    "sv_intensity_sd_db": ("lower", "lower = steadier loudness within the window"),
    "sv_tremor_rate_hz": ("neutral", "dominant 2-12 Hz F0 modulation rate; ~4-7 Hz typical of vocal tremor"),
    "sv_tremor_extent_st": ("lower", "lower = less frequency tremor; large extent is a tremor sign"),
    "sv_jitter_sd_across_tokens": ("lower", "lower = more reproducible across repetitions of the same task"),
    "sv_shimmer_sd_across_tokens": ("lower", "lower = more reproducible across repetitions"),
    "sv_hnr_sd_across_tokens": ("lower", "lower = more reproducible across repetitions"),
    "sv_measured": ("higher", "1 = steady-state perturbation measures actually obtained"),
    # ---- v9 audit / usability ----
    "sv_windows_usable": ("higher", "1 = enough steady signal to trust the WINDOW "
                                    "medians (CPPS, window HNR, steadiness); governs "
                                    "the cepstral block"),
    "sv_perturbation_usable": ("higher", "1 = jitter/shimmer are additionally stable "
                                         "and the signal type permits cycle-based "
                                         "measures; governs ONLY jitter and shimmer"),
    "sv_usable": ("higher", "1 = summary rests on enough valid steady-state signal; "
                            "EXCLUDE files with 0 from group statistics"),
    "sv_measured_total_s": ("higher", "seconds of steady phonation behind the medians; "
                                      "under ~10 s treat the numbers as provisional"),
    "sv_n_tokens_expected": ("neutral", "tokens implied by breath gaps + edit points + 1"),
    "sv_token_count_mismatch": ("lower", "1 = segmentation disagrees with the gap count; "
                                         "MPT is suspect, supply boundaries instead"),
    "sv_boundaries_from_file": ("higher", "1 = token boundaries were supplied, not inferred"),
    "sv_rej_low_voicing": ("lower", "windows discarded for insufficient voicing"),
    "sv_rej_f0_outlier": ("lower", "windows discarded as creak/subharmonic (off the speaker's F0)"),
    "sv_rej_f0_step": ("lower", "windows discarded for an internal F0 discontinuity"),
    "sv_rej_splice": ("lower", "windows discarded for overlapping a detected edit point"),
    "sv_rej_other": ("lower", "windows discarded for other reasons (no data / unmeasurable)"),
    "sv_f1f2_dispersion_logarea": ("lower", "on ONE sustained vowel a SMALLER F1-F2 cloud means "
                                            "steadier articulation (opposite meaning to the "
                                            "connected-speech version of this metric)"),
    "sv_f1f2_cloud_spread_hz": ("lower", "same as above in Hz: lower = steadier vowel"),
    "sv_signal_type": ("lower", "1 = nearly periodic (jitter/shimmer valid), 2 = "
                                "subharmonics (unreliable), 3 = aperiodic (perturbation "
                                "UNDEFINED - use cepstral/spectral measures)"),
    "sv_voiced_at_045": ("higher", "voiced fraction at Praat's default voicing threshold"),
    "sv_voiced_at_020": ("higher", "voiced fraction at a very permissive threshold; still low "
                                   "= the signal really is aperiodic, not a settings problem"),
    "sv_voiced_low_floor": ("higher", "voiced fraction with a 60 Hz floor; rules out the pitch "
                                      "range as the cause of low voicing"),
    "sv_window_yield": ("higher", "steady 2-3 s windows found / windows that would fit; "
                                  "low = the voice cannot hold a steady state"),
    # ---- v18: how much of the vowel is behind the numbers -------------------
    "sv_analyzed_fraction": ("higher", "seconds measured / seconds phonated. 1.0 = the "
                                       "whole vowel is behind the medians; 0.4 = 60% was "
                                       "discarded. Read it WITH sv_discard_interior_s: "
                                       "losing time at the onset and offset of a long take "
                                       "is normal, losing it in the middle is not. Also "
                                       "capped by geometry - about 0.8-0.95 is the best a "
                                       "clean take can reach, since the edge trim and the "
                                       "last part-window are always dropped"),
    "sv_window_yield_s": ("higher", "seconds measured / seconds the tiling COULD measure. "
                                    "The duration-weighted twin of sv_window_yield with the "
                                    "geometry divided out, so 1.0 means nothing was lost to "
                                    "the voice and a low value is a real steadiness finding "
                                    "rather than a short take"),
    "sv_steady_frame_fraction": ("higher", "fraction of phonated frames on which a valid "
                                           "analysis window could be CENTRED, computed frame "
                                           "by frame with no window grid. Unaffected by take "
                                           "length, and defined even when no window survived, "
                                           "so this is the column that separates 'the voice "
                                           "never held still' from 'the take was too short for "
                                           "the usability gate'. Jitter, tremor and slow drift "
                                           "do NOT lower it - they are measured elsewhere"),
    "sv_longest_steady_run_s": ("higher", "longest uninterrupted steady stretch in seconds. "
                                          "80% steady frames in one block and 80% scattered "
                                          "over ten islands are different findings and only "
                                          "this column separates them; it is also the honest "
                                          "upper bound on how long a window you could place"),
    "sv_analyzable_total_s": ("neutral", "seconds the tiling geometry could measure at best "
                                         "(after edge trim and the part-window remainder); "
                                         "the denominator of sv_window_yield_s"),
    "sv_discard_unsteady_s": ("lower", "seconds discarded by the validity screen: the part of "
                                       "the loss that is about the VOICE"),
    "sv_discard_edge_s": ("neutral", "seconds removed by the fixed onset/offset trim; a "
                                     "setting, not a finding"),
    "sv_discard_quantisation_s": ("neutral", "seconds left over because the remainder was "
                                             "shorter than one window; a setting, not a "
                                             "finding (shorten vowel_window_max_s to recover "
                                             "it)"),
    "sv_discard_onset_s": ("lower", "phonated seconds before the first measured window: "
                                    "onset instability plus the edge trim"),
    "sv_discard_interior_s": ("lower", "phonated seconds discarded BETWEEN measured windows. "
                                       "The clearest single sign of a take that kept breaking "
                                       "down mid-vowel"),
    "sv_discard_offset_s": ("lower", "phonated seconds after the last measured window: offset "
                                     "decay, the edge trim and the part-window remainder"),
    "sv_discard_dead_token_s": ("lower", "seconds in tokens that yielded no measured window "
                                         "at all"),
    "sv_n_interior_gaps": ("lower", "number of separate unmeasured gaps inside the phonation; "
                                    "0 = one continuous measured block"),
    "sv_n_steady_stretches": ("lower", "how many separate steady stretches the frame-level "
                                       "analysis found; 1 = the voice held once and kept it"),
    "sv_cpps_token_db": ("higher", "CPPS over the whole token - computable even when the voice "
                                   "is too aperiodic for jitter/shimmer, so use it when they are blank"),
    "sv_hnr_token_db": ("higher", "per-interval median HNR over the whole token; same role as above"),
    "f0_wide_median_hz": ("neutral", "F0 an UNCONSTRAINED tracker would report; an octave below "
                                     "the in-range value means real subharmonic energy"),
    "f0_subharmonic_lock": ("lower", "1 = a free tracker falls to F0/2 on this file: period "
                                     "doubling / diplophonia present"),
    "f0_lock_semitones": ("lower", "in-range F0 minus unconstrained F0, in semitones; ~12 = full "
                                   "octave lock, ~0 = clean periodicity"),
    "sv_tremor_peak_ratio": ("neutral", "prominence of the 2-12 Hz peak; below ~2.5 there is "
                                        "no measurable tremor and rate/extent are blank"),
    "pitch_floor_used_hz": ("neutral", "pitch floor this file was measured with; must be "
                                       "IDENTICAL across sessions of one subject"),
    "pitch_ceiling_used_hz": ("neutral", "pitch ceiling this file was measured with; must be "
                                         "IDENTICAL across sessions of one subject"),
    "silence_threshold_db_used": ("neutral", "silence line used for segmentation (dB rel. peak)"),
    "f0_octave_ok": ("higher", "1 = harmonic comb confirms the tracked F0, 0.5 = ambiguous "
                               "(check by ear), 0 = octave error or inconclusive"),
    "f0_octave_corrected": ("lower", "1 = the analysis range was doubled/halved by the comb "
                                     "check; verify this file by ear before using it"),
    "f0_octave_score_ref_db": ("higher", "odd-vs-even harmonic contrast at the tracked F0; "
                                         "near 0 dB = real fundamental, very negative = subharmonic"),
    "f0_octave_score_half_db": ("lower", "same contrast one octave below; near 0 dB means a full "
                                         "comb also fits there, i.e. the tracker doubled"),
    "rec_digital_silence": ("lower", "1 = the silent reference is pasted/edited silence, so SNR "
                                     "describes the edit and not the recording"),
    "rec_quality_score": ("higher", "higher = better recording (about the signal, not the voice)"),
    "rec_snr_db": ("higher", "higher SNR = cleaner recording"),
    "rec_effective_bandwidth_hz": ("neutral", "informational; low value alone does NOT mean bad voice"),
    "rec_spectral_edge_hz": ("neutral", "informational; where the spectrum dies"),
    "rec_is_bandlimited": ("lower", "0 = full-band (good); 1 = telephone/codec cutoff (VQ unreliable)"),
    "rec_clipping_fraction": ("lower", "lower = less clipping = cleaner recording"),
    "rec_sample_rate_hz": ("neutral", "recording property; context only"),
    "rec_vq_reliable": ("higher", "1 = voice-quality metrics trustworthy on this recording"),
    "rec_snr_estimable": ("higher", "1 = a silent reference existed so SNR is real; 0 = SNR not estimable (record room tone)"),
    # ---- dispersion of the robust medians (v16) -----------------------------
    # The IQR behind each window-median. This is the column that answers
    # "is this number stable enough to compare across days?": when the IQR
    # approaches the value itself, the single-number summary is noise.
    "cpps_iqr_db": ("lower", "window-to-window IQR of CPPS; lower = more repeatable estimate"),
    "hnr_iqr_db": ("lower", "window-to-window IQR of HNR; lower = more repeatable estimate"),
    "jitter_iqr": ("lower", "window-to-window IQR of jitter; if it approaches the jitter value, the median is not usable"),
    "shimmer_iqr": ("lower", "window-to-window IQR of shimmer; if it approaches the shimmer value, the median is not usable"),
    "voiced_fraction": ("higher", "fraction of frames the tracker calls voiced; always computable, so it still separates files when jitter/shimmer cannot be measured"),
}


# Metrics shown in the overview grid, per task. Ordered so the robust,
# always-computable measures come first: on a bad recording those are the panels
# that still carry a number.
OVERVIEW_METRICS_READING = [
    "cpps_db", "hnr_median_perinterval_db", "cpp_db",
    "speech_rate_syll_per_s", "speaking_time_fraction", "ems_3_8hz_ratio",
    "vowel_dispersion_logarea", "intensity_decay_db", "intensity_nucleus_std_db",
    "pitch_cv", "pause_ratio_to_speech", "fluency_overall",
]
OVERVIEW_METRICS_SUSTAINED = [
    # robust first: these four are defined for every take, usable or not
    "cpps_db", "sv_cpps_token_db", "sv_hnr_token_db", "voiced_fraction",
    "sv_analyzed_fraction", "sv_steady_frame_fraction",
    "sv_window_yield", "f0_subharmonic_lock",
    # then the steady-state block, which needs a measurable voice
    "sv_mpt_longest_s", "jitter_ppq5_percent", "shimmer_apq11_percent",
    "hnr_db", "pitch_mean_hz", "sv_f0_sd_semitones",
    "sv_f0_drift_st_per_s", "sv_intensity_sd_db", "sv_tremor_extent_st",
]


# Curated per-task metric sets for the individual bar plots. The full 97-metric
# table always goes to the CSV; plotting all of it buries the ~20 numbers that
# carry the clinical signal. Use plot_scope="all" to plot every metric.
#
# v16: each perturbation median is listed immediately next to its IQR, so the
# plot pair answers "did it change?" and "is the change bigger than the
# measurement spread?" side by side instead of in two different folders.
CORE_METRICS_SUSTAINED = [
    # --- robust block: computable even when perturbation analysis fails ------
    "cpps_db", "cpps_iqr_db", "sv_cpps_token_db", "cpp_db",
    "hnr_db", "hnr_iqr_db", "hnr_median_perinterval_db", "sv_hnr_token_db",
    "voiced_fraction", "sv_window_yield", "sv_window_yield_s",
    "sv_analyzed_fraction", "sv_steady_frame_fraction", "sv_longest_steady_run_s",
    "f0_subharmonic_lock", "f0_lock_semitones",
    "alpha_ratio_db", "hammarberg_index", "spectral_tilt_db_per_khz",
    # --- perturbation, each with its dispersion ------------------------------
    "jitter_local_percent", "jitter_ppq5_percent", "jitter_iqr",
    "shimmer_local_db", "shimmer_apq11_percent", "shimmer_iqr",
    "nhr", "mean_autocorrelation", "voice_breaks_degree_percent", "num_pulses",
    # --- phonatory capacity and steadiness ----------------------------------
    "sv_mpt_longest_s", "sv_mpt_longest_uninterrupted_s", "sv_mpt_mean_s",
    "sv_total_phonation_s", "sv_n_tokens",
    "pitch_mean_hz", "sv_f0_sd_semitones", "sv_f0_drift_st_per_s",
    "sv_intensity_sd_db", "intensity_decay_db", "sv_intensity_decay_db_per_s",
    "sv_tremor_rate_hz", "sv_tremor_extent_st",
    "sv_jitter_sd_across_tokens", "sv_shimmer_sd_across_tokens",
    "sv_hnr_sd_across_tokens",
    # --- vowel identity / steadiness ----------------------------------------
    "f1_mean_hz", "f2_mean_hz", "f1_std_hz", "f2_std_hz",
    "sv_f1f2_cloud_spread_hz",
    "intensity_active_mean_db", "intensity_nucleus_std_db",
    # --- recording ----------------------------------------------------------
    "rec_quality_score", "rec_snr_db", "rec_clipping_fraction",
]
CORE_METRICS_READING = [
    # --- robust voice quality ----------------------------------------------
    "cpps_db", "cpp_db", "hnr_db", "hnr_median_perinterval_db",
    "alpha_ratio_db", "hammarberg_index", "spectral_tilt_db_per_khz",
    "voiced_fraction", "f0_subharmonic_lock",
    # --- perturbation (secondary on connected speech) -----------------------
    "jitter_local_percent", "jitter_ppq5_percent",
    "shimmer_local_db", "shimmer_apq11_percent",
    "nhr", "voice_breaks_degree_percent",
    # --- rate and timing ----------------------------------------------------
    "duration_speech_s", "articulation_time_s", "phonation_time_s",
    "speech_rate_syll_per_s", "articulation_rate_syll_per_s",
    "syllable_count_used", "speaking_time_fraction",
    "pause_count", "pauses_per_minute", "pause_ratio_to_speech",
    "pause_mean_duration_s",
    "filler_count", "fillers_per_minute", "filler_ratio_to_speech",
    # --- rhythm / envelope --------------------------------------------------
    "ems_3_8hz_ratio", "ems_peak_freq_hz", "ems_syllabic_to_slow_ratio", "npvi",
    # --- loudness and prosody ----------------------------------------------
    "intensity_active_mean_db", "intensity_active_std_db",
    "intensity_nucleus_std_db", "intensity_decay_db",
    "pitch_mean_hz", "pitch_cv", "pitch_range_semitones", "voiced_percent",
    # --- articulation -------------------------------------------------------
    "f1_mean_hz", "f2_mean_hz", "vowel_space_area", "vowel_dispersion_logarea",
    "vowel_cloud_spread_hz",
    # --- composites and recording ------------------------------------------
    "fluency_overall", "fluency_voice_quality", "fluency_articulation",
    "rec_quality_score", "rec_snr_db", "rec_clipping_fraction",
]


def core_metrics_for(results) -> List[str]:
    """Curated bar-plot metric list matching the task of the analysed files."""
    tasks = {getattr(r, "task", TASK_READING) for r in results}
    if tasks == {TASK_SUSTAINED}:
        return list(CORE_METRICS_SUSTAINED)
    if tasks == {TASK_READING}:
        return list(CORE_METRICS_READING)
    seen, out = set(), []
    for m in CORE_METRICS_SUSTAINED + CORE_METRICS_READING:
        if m not in seen:
            seen.add(m); out.append(m)
    return out


def overview_metrics_for(results) -> List[str]:
    """Overview metric list matching the task of the analysed files."""
    tasks = {getattr(r, "task", TASK_READING) for r in results}
    if tasks == {TASK_SUSTAINED}:
        return list(OVERVIEW_METRICS_SUSTAINED)
    if tasks == {TASK_READING}:
        return list(OVERVIEW_METRICS_READING)
    return OVERVIEW_METRICS_SUSTAINED[:6] + OVERVIEW_METRICS_READING[:6]


# -----------------------------------------------------------------------------
# TASK-DEPENDENT DIRECTION OVERRIDES  (v16 fix)
# -----------------------------------------------------------------------------
# Some metrics are computed for BOTH tasks but do not mean the same thing in
# each, so a single global direction mislabels one of them. The pause block is
# the clear case: on connected speech those really are hesitation pauses and
# fewer is better, but on a sustained-vowel file the "pauses" are the gaps
# BETWEEN TRIALS, so pause_count is essentially the trial count minus one.
# Labelling it "fewer pauses = more fluent" told you that recording two /a/
# tokens instead of three was a clinical improvement. These metrics are
# descriptive on sustained files, and _pick_best_on_per_metric() must not
# optimise them either.
TASK_DIRECTION_OVERRIDES: Dict[str, Dict[str, Tuple[str, str]]] = {
    TASK_SUSTAINED: {
        "pause_count": ("neutral", "breath gaps BETWEEN tokens = trial count - 1; "
                                   "structural, not a fluency measure"),
        "pauses_per_minute": ("neutral", "breath-gap rate; depends on how many trials "
                                         "were recorded, not on the voice"),
        "pause_total_duration_s": ("neutral", "total inter-trial silence; recording "
                                              "protocol, not a fluency measure"),
        "pause_mean_duration_s": ("neutral", "mean inter-trial breath pause; context only"),
        "pause_ratio_to_speech": ("neutral", "inter-trial silence relative to phonation; "
                                             "context only"),
        # On ONE vowel a tighter F1-F2 cloud means steadier articulation, i.e.
        # the good direction is the OPPOSITE of the connected-speech reading.
        # (The sustained file reports this as sv_f1f2_cloud_spread_hz, which
        # already carries the correct direction; this entry guards the generic
        # name in case a mixed batch plots it.)
        "vowel_cloud_spread_hz": ("lower", "on a single sustained vowel a SMALLER F1-F2 "
                                           "cloud means steadier articulation"),
    },
}


def metric_direction(metric: str, task: Optional[str] = None) -> Tuple[str, str]:
    """Direction + guidance for a metric, honouring per-task overrides."""
    if task:
        ov = TASK_DIRECTION_OVERRIDES.get(task, {})
        if metric in ov:
            return ov[metric]
    return METRIC_DIRECTION.get(metric, ("neutral", "no directional guidance"))


def _task_of(results) -> Optional[str]:
    """The single task shared by these results, or None if the batch is mixed."""
    tasks = {getattr(r, "task", None) for r in results}
    tasks.discard(None)
    return next(iter(tasks)) if len(tasks) == 1 else None


def _direction_subtitle(metric: str, task: Optional[str] = None) -> str:
    direction, note = metric_direction(metric, task)
    arrow = {
        "higher": "\u2191 higher is generally better",
        "lower":  "\u2193 lower is generally better",
        "around": "\u2248 expected around a typical range",
        "neutral": "\u2013 descriptive (no inherent good direction)",
    }.get(direction, "\u2013 descriptive")
    return f"{arrow}  |  {note}"


# =============================================================================
# PLOTTING
# =============================================================================

def make_bar_plots(results, output_folder, output_prefix, dpi=130,
                   plot_scope="core", metrics=None):
    """
    plot_scope: "core" (default) plots the curated task-appropriate metric set;
                "all" plots every metric in the table.
    """
    plots_dir = output_folder / f"{output_prefix}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [Path(r.file_path).stem for r in results]
    tables = [get_metric_table(r) for r in results]
    _batch_task = _task_of(results)
    if metrics is not None:
        metric_names = [m for m in metrics if m in tables[0]]
    elif plot_scope == "core":
        metric_names = [m for m in core_metrics_for(results) if m in tables[0]]
    else:
        metric_names = list(tables[0].keys())
    saved = []
    for idx, metric in enumerate(metric_names, start=1):
        values = np.array([t.get(metric, np.nan) for t in tables], dtype=float)
        if np.all(np.isnan(values)):
            print(f"  [skip] {metric}: no numeric data for any file")
            continue
        direction = metric_direction(metric, _batch_task)[0]
        bar_color = {"higher": "#4C78A8", "lower": "#4C78A8",
                     "around": "#8A6FB0", "neutral": "#9AA0A6"}.get(direction, "#4C78A8")
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.8))
        x = np.arange(len(labels))
        # v20: was np.nan_to_num(..., nan=0.0), which DREW a missing value as
        # a real zero-height bar. Passing NaN leaves the slot empty instead.
        bars = ax.bar(x, np.where(np.isfinite(values), values, np.nan), color=bar_color,
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                txt = "n/a"
            elif abs(v) >= 100:
                txt = f"{v:.0f}"
            elif abs(v) >= 1:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    txt, ha="center", va="bottom", fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.margins(y=0.15)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.text(0.5, 0.025, _direction_subtitle(metric, _batch_task), ha="center",
                 va="bottom", fontsize=9, color="#444444", style="italic", wrap=True)
        out_path = plots_dir / f"{idx:02d}_{metric}.png"
        fig.savefig(out_path, dpi=dpi); plt.close(fig); saved.append(out_path)
    print(f"\n{len(saved)} metric plots saved to: {plots_dir}")
    return saved


def make_overview_grid(results, output_folder, output_prefix, metrics=None, dpi=130):
    if metrics is None:
        metrics = overview_metrics_for(results)
    labels = [Path(r.file_path).stem for r in results]
    tables = [get_metric_table(r) for r in results]
    _batch_task = _task_of(results)
    n = len(metrics); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    compact_arrow = {"higher": "\u2191 better", "lower": "\u2193 better",
                     "around": "\u2248 typical range", "neutral": "\u2013 descriptive"}
    x = np.arange(len(labels))
    for ax, metric in zip(axes, metrics):
        values = np.array([t.get(metric, np.nan) for t in tables], dtype=float)
        direction = metric_direction(metric, _batch_task)[0]
        bar_color = {"higher": "#72B7B2", "lower": "#72B7B2",
                     "around": "#B79CD6", "neutral": "#BDC1C6"}.get(direction, "#72B7B2")
        ax.bar(x, np.where(np.isfinite(values), values, np.nan), color=bar_color,
               edgecolor="black", linewidth=0.5)
        hint = compact_arrow.get(direction, "\u2013")
        ax.set_title(f"{metric}\n({hint})", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.margins(y=0.2)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Speech Metrics Overview  (\u2191/\u2193 = direction generally associated "
                 "with healthier voice/speech; indicative, not diagnostic)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_folder / f"{output_prefix}_overview.png"
    fig.savefig(out_path, dpi=dpi); plt.close(fig)
    print(f"Overview grid saved to: {out_path}")
    return out_path


def write_consolidated_csv(results, output_folder, output_prefix):
    """
    Wide matrix: one row per metric, one column per file.

    v16: the file is now written in two clearly labelled blocks - the 97
    canonical METRICS, then the AUDIT columns (settings used, octave verdict,
    window-rejection counts, usability gate, recording properties). Mixing the
    two, as earlier versions did, invites a provenance column into a statistical
    model as though it were an outcome.

    It no longer calls write_long_csv() as a side effect; analyze_files() calls
    both explicitly, so calling this one alone no longer writes and announces a
    second file you did not ask for.
    """
    tables = [get_metric_table(r) for r in results]
    audit_tables = [get_audit_table(r) for r in results]
    _batch_task = _task_of(results)
    file_labels = [Path(r.file_path).stem for r in results]
    metric_names = list(tables[0].keys())
    csv_path = output_folder / f"{output_prefix}_all_metrics.csv"

    def _s(r, key, default=""):
        v = r.analysis_settings.get(key, default)
        return "" if v is None else str(v)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "direction", "guidance"] + file_labels)
        # ---- text audit rows (v9): every gating decision, per file ----------
        writer.writerow(["task", "neutral", "analysis task used for this file"]
                        + [getattr(r, "task", "") for r in results])
        writer.writerow(["task_source", "neutral", "how the task was decided"]
                        + [getattr(r, "task_source", "") for r in results])
        writer.writerow(["octave_verdict", "neutral",
                         "harmonic-comb octave check: ok | ambiguous | tracker_halved "
                         "| tracker_doubled | inconclusive"]
                        + [_s(r, "octave_verdict") for r in results])
        writer.writerow(["octave_note", "neutral", "why the octave check decided that"]
                        + [_s(r, "octave_note") for r in results])
        writer.writerow(["snr_reference", "neutral",
                         "where the noise floor came from (none | leading_trailing | "
                         "internal_below_threshold)"]
                        + [_s(r, "snr_reference") for r in results])
        writer.writerow(["sv_boundaries_source", "neutral",
                         "token boundaries: inferred | file"]
                        + [getattr(r.sustained_metrics, "boundaries_source", "")
                           for r in results])
        writer.writerow(["sv_window_rejections", "neutral",
                         "why analysis windows were discarded"]
                        + [getattr(r.sustained_metrics, "window_rejections", "")
                           for r in results])
        writer.writerow(["sv_coverage_note", "neutral",
                         "v18: how much of the phonation was analysed and where the "
                         "rest went"]
                        + [getattr(r.sustained_metrics, "coverage_note", "")
                           for r in results])
        writer.writerow(["sv_usable_reasons", "neutral",
                         "blank = usable; otherwise why this file should not enter "
                         "group statistics"]
                        + [getattr(r.sustained_metrics, "usable_reasons", "")
                           for r in results])
        writer.writerow(["sv_token_spans_s", "neutral",
                         "token boundaries actually used (s)"]
                        + ["; ".join(f"{a:.1f}-{b:.1f}" for a, b in
                                     getattr(r.sustained_metrics, "token_spans_s", []))
                           for r in results])
        writer.writerow(["rec_warnings", "neutral", "recording-quality warnings"]
                        + [" | ".join(r.recording_quality.warnings) for r in results])
        def _emit(names, source_tables):
            for metric in names:
                direction, note = metric_direction(metric, _batch_task)
                row = [metric, direction, note]
                for t in source_tables:
                    v = t.get(metric, np.nan)
                    # v20: isinstance(v, float) is FALSE for np.float64 and
                    # every other NumPy scalar, so a NumPy NaN slipped past this
                    # test and was written as the literal string "nan".
                    row.append("" if (v is None or not np.isfinite(float(v)))
                               else f"{v:.6g}")
                writer.writerow(row)

        # ---- BLOCK 1: the 97 canonical metrics ------------------------------
        writer.writerow([f"### METRICS ({len(metric_names)}) - measurements of the voice",
                         "", ""] + [""] * len(file_labels))
        _emit(metric_names, tables)
        # ---- BLOCK 2: provenance / QC (v16) ---------------------------------
        # Separated so nothing downstream can treat a settings value or a
        # rejection count as if it were an outcome measure. This is the block to
        # read FIRST when two sessions disagree: a change in pitch_floor_used_hz
        # or f0_octave_corrected explains a "finding" that is not one.
        writer.writerow([f"### AUDIT ({len(AUDIT_COLUMNS)}) - how it was measured, "
                         "NOT measurements of the voice", "", ""]
                        + [""] * len(file_labels))
        _emit(AUDIT_COLUMNS, audit_tables)
    print(f"Consolidated CSV saved to: {csv_path} "
          f"({len(metric_names)} metrics + {len(AUDIT_COLUMNS)} audit columns)")
    return csv_path


def write_long_csv(results, output_folder, output_prefix):
    """
    Tidy/long export (v9): one row per file x metric, with the grouping columns
    already parsed. The wide matrix above is convenient to read but awkward for
    mixed models; this one loads straight into R/pandas:

        file, date, condition, condition_class, task, usable, metric, value

    `usable` is sv_usable for sustained files (1 = the summary rests on enough
    valid steady-state signal), empty for connected speech.
    """
    csv_path = output_folder / f"{output_prefix}_long.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "date", "condition", "condition_class", "repetition",
                         "task", "usable", "metric", "value", "direction"])
        for r in results:
            table = get_metric_table(r)
            stem = Path(r.file_path).stem
            date = extract_date_from_filename(r.file_path)
            cond = parse_condition_from_filename(r.file_path)
            cls = classify_condition(cond)
            rep = parse_repetition_from_filename(r.file_path)
            usable = ""
            if getattr(r, "task", "") == TASK_SUSTAINED:
                usable = "1" if getattr(r.sustained_metrics, "usable", False) else "0"
            for metric, v in table.items():
                # v20: NumPy-scalar-safe, see note in write_consolidated_csv.
                if v is None or not np.isfinite(float(v)):
                    continue
                direction = metric_direction(metric, getattr(r, "task", None))[0]
                writer.writerow([stem, date, cond, cls, "" if rep is None else rep,
                                 getattr(r, "task", ""), usable, metric,
                                 f"{float(v):.6g}", direction])
    print(f"Long-format CSV saved to: {csv_path}")
    return csv_path


# =============================================================================
# PER-STIMULATION AGGREGATION + PLOTS  ("plots_per_stim")
# =============================================================================

_SPECIAL_CONDITIONS = ("pre", "post", "followup", "follow_up", "follow-up")


_REP_SUFFIX_RE = re.compile(r"^(?P<cond>[A-Za-z][A-Za-z_]*)-(?P<rep>\d{1,3})$")


def parse_condition_from_filename(file_path: str) -> str:
    """
    Condition token from the file stem.

    FIXED in v10. With one file per repetition the stems became
    "audio_12_20260309_PRE-01", so the last underscore token is "PRE-01", which
    matched none of pre/post/off/followup and fell through to "other" - i.e.
    every PRE, POST, OFF and FOLLOWUP file was silently dropped into one
    meaningless bucket and only ON survived (via startswith("on")). A trailing
    "-<digits>" repetition index is now stripped.

    The separator must be a HYPHEN: "ON-02" is repetition 2 of condition ON,
    while "ON2" is left intact, because that form is used for a second
    stimulation SETTING and must stay a distinct condition.
    """
    stem = Path(file_path).stem
    token = (stem.split("_")[-1] if "_" in stem else stem).strip()
    m = _REP_SUFFIX_RE.match(token)
    return m.group("cond") if m else token


def parse_repetition_from_filename(file_path: str) -> Optional[int]:
    """Repetition index from a trailing '-<digits>', else None."""
    stem = Path(file_path).stem
    token = (stem.split("_")[-1] if "_" in stem else stem).strip()
    m = _REP_SUFFIX_RE.match(token)
    return int(m.group("rep")) if m else None


def classify_condition(condition: str) -> str:
    c = condition.strip().lower()
    if c == "off":
        return "OFF"
    if c.startswith("on"):
        return "ON"
    if c in ("followup", "follow_up", "follow-up"):
        return "followup"
    if c in ("pre", "post"):
        return c
    return "other"


def _nanmean_tables(tables, how="median"):
    """
    Combine the repetition tables of one session.

    v10: the default is the MEDIAN, not the mean. With three repetitions the
    median is what you want (one bad take cannot drag the session), and it is
    what write_session_csv reports, so the two outputs agree.
    """
    if not tables:
        return {}
    keys = tables[0].keys(); out = {}
    for k in keys:
        vals = np.array([t.get(k, np.nan) for t in tables], dtype=float)
        if np.all(np.isnan(vals)):
            out[k] = np.nan
        else:
            out[k] = float(np.nanmedian(vals) if how == "median" else np.nanmean(vals))
    return out


def _usable_tables(entries, usable_only=True):
    """entries: list of (table, usable). Drops unusable reps, keeps all if none survive."""
    keep = [t for t, u in entries if u] if usable_only else [t for t, _ in entries]
    if keep:
        return keep, len(keep), False
    return [t for t, _ in entries], 0, True


def _pick_best_on_per_metric(on_tables, off_table, task=None):
    """
    WARNING (v9): this takes the best value of EACH METRIC INDEPENDENTLY, so the
    resulting column is a composite that no single recording ever produced - the
    HNR may come from one stimulation setting and the jitter from another. It is
    a screening device, not a result, and it is no longer the default
    (on_aggregation="all"). Keep it only for exploratory plots.
    """
    if not on_tables:
        return {}
    keys = on_tables[0].keys(); best = {}
    for k in keys:
        vals = np.array([t.get(k, np.nan) for t in on_tables], dtype=float)
        if np.all(np.isnan(vals)):
            best[k] = np.nan; continue
        direction = metric_direction(k, task)[0]
        finite = vals[~np.isnan(vals)]
        if direction == "higher":
            best[k] = float(np.nanmax(vals))
        elif direction == "lower":
            best[k] = float(np.nanmin(vals))
        else:
            ref = None
            if off_table is not None:
                ov = off_table.get(k, np.nan)
                if ov is not None and not (isinstance(ov, float) and np.isnan(ov)):
                    ref = float(ov)
            if ref is not None:
                idx = int(np.nanargmin(np.abs(vals - ref))); best[k] = float(vals[idx])
            else:
                best[k] = float(np.median(finite))
    return best


def build_per_stim_columns(results, on_aggregation="all", usable_only=True):
    """
    on_aggregation:
      "all"    (default, v9) one column per ON recording - nothing is invented
      "median" per-metric median across the ON recordings of that day
      "best"   the old per-metric best (a composite; see _pick_best_on_per_metric)
    """
    by_day = {}; day_order = []
    for r in results:
        date = extract_date_from_filename(r.file_path)
        cls = classify_condition(parse_condition_from_filename(r.file_path))
        if date not in by_day:
            by_day[date] = {}; day_order.append(date)
        usable = (getattr(r.sustained_metrics, "usable", False)
                  if getattr(r, "task", "") == TASK_SUSTAINED else True)
        by_day[date].setdefault(cls, []).append((get_metric_table(r), usable))
    on_names = {}
    for r in results:
        date = extract_date_from_filename(r.file_path)
        if classify_condition(parse_condition_from_filename(r.file_path)) == "ON":
            on_names.setdefault(date, []).append(
                parse_condition_from_filename(r.file_path))
    day_order.sort()
    labels = []; tables = []
    special_counts = {}
    for date in day_order:
        for cond in _SPECIAL_CONDITIONS:
            cls = classify_condition(cond)
            special_counts[cls] = special_counts.get(cls, 0) + (
                1 if cls in by_day.get(date, {}) else 0)
    for date in day_order:
        groups = by_day[date]
        for special in ("pre", "post", "followup"):
            if special in groups:
                keep, n_ok, none_ok = _usable_tables(groups[special], usable_only)
                avg = _nanmean_tables(keep)
                base = special if special_counts.get(special, 0) <= 1 else f"{special}\n{date}"
                tag = f"\nn={n_ok}" if n_ok else "\n(NO usable rep)"
                labels.append(base + tag); tables.append(avg)
        off_table = None
        if "OFF" in groups:
            keep, n_ok, none_ok = _usable_tables(groups["OFF"], usable_only)
            off_table = _nanmean_tables(keep)
            labels.append(f"OFF\n{date}" + (f"\nn={n_ok}" if n_ok else "\n(NO usable rep)"))
            tables.append(off_table)
        on_entries = groups.get("ON", [])
        on_tables = [t for t, u in on_entries if u] or [t for t, _ in on_entries] \
            if on_entries else []
        if on_tables:
            n_on = len(on_tables)
            if on_aggregation == "all" or n_on == 1:
                names = on_names.get(date, [])
                for k, t in enumerate(on_tables):
                    nm = names[k] if k < len(names) else f"ON{k+1}"
                    lbl = f"{nm}\n{date}" if n_on > 1 else f"ON\n{date}"
                    labels.append(lbl); tables.append(t)
            elif on_aggregation == "median":
                med = {}
                for k in on_tables[0].keys():
                    vals = np.array([t.get(k, np.nan) for t in on_tables], dtype=float)
                    med[k] = np.nan if np.all(np.isnan(vals)) else float(np.nanmedian(vals))
                labels.append(f"ON median (n={n_on})\n{date}"); tables.append(med)
            else:                       # "best": composite, exploratory only
                labels.append(f"best ON [composite]\n{date}")
                tables.append(_pick_best_on_per_metric(on_tables, off_table,
                                                       task=_task_of(results)))
        if "other" in groups:
            keep, n_ok, _ = _usable_tables(groups["other"], usable_only)
            labels.append(f"other\n{date}" + (f"\nn={n_ok}" if n_ok else "\n(NO usable rep)"))
            tables.append(_nanmean_tables(keep))
    return labels, tables


def _make_bar_plots_from_tables(labels, tables, plots_dir, dpi=130, metrics=None,
                                task=None):
    _batch_task = task
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not tables:
        print("  [plots_per_stim] no columns to plot"); return []
    if metrics is not None:
        metric_names = [m for m in metrics if m in tables[0]]
    else:
        metric_names = list(tables[0].keys())
    saved = []
    for idx, metric in enumerate(metric_names, start=1):
        values = np.array([t.get(metric, np.nan) for t in tables], dtype=float)
        if np.all(np.isnan(values)):
            print(f"  [skip] {metric}: no numeric data for any column"); continue
        direction = metric_direction(metric, _batch_task)[0]
        bar_color = {"higher": "#4C78A8", "lower": "#4C78A8",
                     "around": "#8A6FB0", "neutral": "#9AA0A6"}.get(direction, "#4C78A8")
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.8))
        x = np.arange(len(labels))
        # v20: was np.nan_to_num(..., nan=0.0), which DREW a missing value as
        # a real zero-height bar. Passing NaN leaves the slot empty instead.
        bars = ax.bar(x, np.where(np.isfinite(values), values, np.nan), color=bar_color,
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                txt = "n/a"
            elif abs(v) >= 100:
                txt = f"{v:.0f}"
            elif abs(v) >= 1:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    txt, ha="center", va="bottom", fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight="bold"); ax.set_ylabel(metric)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.margins(y=0.15)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.text(0.5, 0.025, _direction_subtitle(metric), ha="center", va="bottom",
                 fontsize=9, color="#444444", style="italic", wrap=True)
        out_path = plots_dir / f"{idx:02d}_{metric}.png"
        fig.savefig(out_path, dpi=dpi); plt.close(fig); saved.append(out_path)
    print(f"\n{len(saved)} per-stim metric plots saved to: {plots_dir}")
    return saved


def _make_overview_grid_from_tables(labels, tables, out_path, metrics=None, dpi=130,
                                    task=None):
    _batch_task = task
    if not tables:
        return None
    if metrics is None:
        metrics = list(OVERVIEW_METRICS_READING)
    n = len(metrics); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    compact_arrow = {"higher": "\u2191 better", "lower": "\u2193 better",
                     "around": "\u2248 typical range", "neutral": "\u2013 descriptive"}
    x = np.arange(len(labels))
    for ax, metric in zip(axes, metrics):
        values = np.array([t.get(metric, np.nan) for t in tables], dtype=float)
        direction = metric_direction(metric, _batch_task)[0]
        bar_color = {"higher": "#72B7B2", "lower": "#72B7B2",
                     "around": "#B79CD6", "neutral": "#BDC1C6"}.get(direction, "#72B7B2")
        ax.bar(x, np.where(np.isfinite(values), values, np.nan), color=bar_color,
               edgecolor="black", linewidth=0.5)
        hint = compact_arrow.get(direction, "\u2013")
        ax.set_title(f"{metric}\n({hint})", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.margins(y=0.2)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-stim Metrics Overview  (one column per condition/recording; "
                 "\u2191/\u2193 = direction generally associated with healthier voice)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi); plt.close(fig)
    print(f"Per-stim overview grid saved to: {out_path}")
    return out_path


def _write_per_stim_csv(labels, tables, csv_path, task=None):
    if not tables:
        return csv_path
    metric_names = list(tables[0].keys())
    flat_labels = [lbl.replace("\n", " ") for lbl in labels]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "direction", "guidance"] + flat_labels)
        for metric in metric_names:
            direction, note = metric_direction(metric, task)
            row = [metric, direction, note]
            for t in tables:
                v = t.get(metric, np.nan)
                row.append("" if (v is None or not np.isfinite(float(v)))
                           else f"{v:.6g}")
            writer.writerow(row)
    print(f"Per-stim consolidated CSV saved to: {csv_path}")
    return csv_path


SESSION_METRICS = [
    # ---- ROBUST BLOCK -------------------------------------------------------
    # Defined for EVERY repetition, so a session whose phonation is too unstable
    # for perturbation analysis still produces numbers instead of dropping out of
    # the table altogether. These are the only rows you can compare between
    # sessions without selection bias, because nothing was excluded to get them.
    "sv_cpps_token_db", "sv_hnr_token_db", "voiced_fraction", "sv_window_yield",
    "sv_analyzed_fraction", "sv_steady_frame_fraction", "sv_longest_steady_run_s",
    "f0_subharmonic_lock", "f0_lock_semitones", "cpp_db",
    "alpha_ratio_db", "hammarberg_index", "spectral_tilt_db_per_khz",
    "spectral_centroid_hz", "sv_total_phonation_s",
    "sv_mpt_longest_uninterrupted_s",
    # ---- STEADY-STATE BLOCK -------------------------------------------------
    # From usable repetitions only. Each perturbation median is followed by its
    # window-to-window IQR: a condition difference smaller than that IQR is not
    # a difference, it is placement noise.
    "cpps_db", "cpps_iqr_db", "hnr_db", "hnr_iqr_db",
    "jitter_ppq5_percent", "jitter_iqr",
    "shimmer_apq11_percent", "shimmer_iqr",
    "pitch_mean_hz", "sv_f0_sd_semitones", "sv_mpt_longest_s",
    "sv_intensity_sd_db", "sv_tremor_extent_st",
    "intensity_active_mean_db", "f1_mean_hz", "f2_mean_hz",
]


def write_session_csv(results, output_folder, output_prefix,
                      metrics=None, usable_only=True, min_usable_reps=2):
    """
    Session-level summary (v10) - THE table to run statistics on.

    With one file per repetition, the individual file is no longer the unit of
    analysis: three repetitions of /a/ recorded minutes apart are three measures
    of ONE state. This writes, per (date x condition):

        n_reps, n_usable, median, sd, min, max, cv  for each key metric

    Two things follow that no per-file table can give you:
      1. the MEDIAN over repetitions is far more stable than any single take
         (jitter and shimmer especially have poor test-retest);
      2. the SD ACROSS REPETITIONS is your measurement noise. A PRE/POST or
         ON/OFF difference is only interpretable if it is large relative to it -
         a rough smallest-detectable-change is ~2.8 x that SD. Without this
         number, "0.485% vs 0.428%" cannot be called a change at all.

    usable_only: repetitions with sv_usable = 0 are excluded (and counted). If a
    session has fewer than min_usable_reps survivors it is written with
    enough_reps = 0 - report it, do not silently average it.
    """
    metrics = metrics or SESSION_METRICS
    # v11: the two blocks must NOT share a denominator.
    # Perturbation measures only exist on usable repetitions, so their median is
    # necessarily computed on a SELECTED subset - and the takes that get dropped
    # are dropped because the voice was unstable, i.e. the selection is on the
    # outcome. That biases every session towards "healthier". The robust block is
    # therefore aggregated over ALL repetitions, so each session has at least one
    # unbiased line, and rep_basis records which rule was used.
    ROBUST_ALL_REPS = {
        "sv_cpps_token_db", "sv_hnr_token_db", "voiced_fraction",
        "sv_window_yield", "f0_subharmonic_lock", "f0_lock_semitones",
        # v18: defined for every take, including the ones the gate rejects -
        # aggregating them over usable reps only would hide exactly the takes
        # they exist to describe
        "sv_analyzed_fraction", "sv_window_yield_s",
        "sv_steady_frame_fraction", "sv_longest_steady_run_s",
        "cpp_db", "intensity_active_mean_db", "sv_mpt_longest_s",
        "sv_mpt_longest_uninterrupted_s",
        "pitch_mean_hz", "f1_mean_hz", "f2_mean_hz",
        "alpha_ratio_db", "hammarberg_index", "spectral_tilt_db_per_khz",
        "spectral_centroid_hz", "sv_total_phonation_s"}
    # v17 - THIRD TIER. These are window medians, so they need enough steady
    # signal (sv_windows_usable), but they are CEPSTRAL: they need no pulse train
    # and no cycle identification. Gating them on the perturbation criterion, as
    # the two-tier version did, discarded a file's CPPS because its JITTER was
    # unstable - and since jitter instability correlates with voice quality, the
    # surviving takes were a biased sample. Observed on a real DBS session: two of
    # three takes dropped for jitter IQR, the survivor happened to have the highest
    # CPPS, and the session median moved from 16.8 to 18.9 dB - about 45% of the
    # entire between-session range in that dataset, produced purely by selection.
    CEPSTRAL_WINDOW_METRICS = {
        "cpps_db", "cpps_iqr_db", "hnr_median_perinterval_db",
        "sv_f0_sd_semitones", "sv_intensity_sd_db", "sv_tremor_extent_st"}
    by_session: Dict[Tuple[str, str], List[Tuple[str, Dict[str, float], bool, bool]]] = {}
    for r in results:
        date = extract_date_from_filename(r.file_path)
        cls = classify_condition(parse_condition_from_filename(r.file_path))
        is_sust = (getattr(r, "task", "") == TASK_SUSTAINED)
        sv = r.sustained_metrics
        win_ok = getattr(sv, "windows_usable", getattr(sv, "usable", False)) \
            if is_sust else True
        pert_ok = getattr(sv, "perturbation_usable", getattr(sv, "usable", False)) \
            if is_sust else True
        by_session.setdefault((date, cls), []).append(
            (Path(r.file_path).stem, get_metric_table(r), win_ok, pert_ok))

    csv_path = output_folder / f"{output_prefix}_by_session.csv"
    rows = []
    for (date, cond) in sorted(by_session.keys()):
        reps = by_session[(date, cond)]
        all_reps = [t for _, t, _, _ in reps]
        keep_win = [t for _, t, w, _ in reps if w] if usable_only else list(all_reps)
        keep_pert = [t for _, t, _, p in reps if p] if usable_only else list(all_reps)
        n_usable = sum(1 for _, _, _, p in reps if p)
        n_win_usable = sum(1 for _, _, w, _ in reps if w)
        enough = 1 if n_usable >= min_usable_reps else 0
        for metric in metrics:
            if metric in ROBUST_ALL_REPS:
                use, basis = all_reps, "all"
            elif metric in CEPSTRAL_WINDOW_METRICS:
                if not keep_win:
                    continue
                use, basis = keep_win, "windows_usable"
            elif not keep_pert:
                # v14: this used to fall back to ALL repetitions, so a session
                # with zero usable takes still printed a perturbation median -
                # e.g. the follow-up row showed jitter 5.05%, which came from a
                # single 2 s window of a type-3 signal. That is precisely the
                # number that must never reach a summary table. The steady-state
                # block is now simply absent for such a session; the robust block
                # still describes it.
                continue
            else:
                use, basis = keep_pert, "perturbation_usable"
            vals = np.array([t.get(metric, np.nan) for t in use], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            rows.append(dict(
                date=date, condition=cond, metric=metric, rep_basis=basis,
                n_reps=len(reps), n_usable=n_usable, n_win_usable=n_win_usable,
                enough_reps=enough,
                n_used=int(vals.size), median=med, sd=sd,
                min=float(np.min(vals)), max=float(np.max(vals)),
                cv_percent=(100.0 * sd / abs(med)) if (np.isfinite(sd) and med) else float("nan"),
                sdc_95=(2.77 * sd) if np.isfinite(sd) else float("nan")))
    fields = ["date", "condition", "metric", "rep_basis", "n_reps", "n_usable",
              "n_win_usable", "enough_reps", "n_used", "median", "sd", "min",
              "max", "cv_percent", "sdc_95"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) and np.isfinite(v)
                             else ("" if isinstance(v, float) else v))
                         for k, v in row.items()})
    print(f"Session-level CSV saved to: {csv_path}")
    print_session_summary(rows)
    return csv_path


def print_session_summary(rows, headline=("cpps_db", "hnr_db",
                                          "jitter_ppq5_percent", "pitch_mean_hz")):
    """Compact console table: median (SD over repetitions) per session."""
    if not rows:
        return
    sessions = []
    for r in rows:
        key = (r["date"], r["condition"])
        if key not in sessions:
            sessions.append(key)
    print("\nSESSION SUMMARY  -  median (SD over repetitions). The n column and the "
          "basis differ PER METRIC:")
    print("  [all] every take   [win] takes with usable window medians   "
          "[pert] takes with usable perturbation")
    _basis_tag = {"all": "all", "windows_usable": "win", "perturbation_usable": "pert",
                  "usable": "pert"}
    hdr = f"{'date':>9s} {'condition':10s} {'n':>7s}  " + "  ".join(
        f"{m.replace('sv_','')[:16]:>22s}" for m in headline)
    print(hdr); print("-" * len(hdr))
    for (date, cond) in sessions:
        sub = {r["metric"]: r for r in rows if r["date"] == date and r["condition"] == cond}
        any_row = next(iter(sub.values()))
        n = f"{any_row['n_usable']}/{any_row['n_reps']}"
        cells = []
        for m in headline:
            r = sub.get(m)
            if r is None:
                cells.append(f"{'-':>22s}"); continue
            sd = f"{r['sd']:.2f}" if np.isfinite(r["sd"]) else "n/a"
            tag = _basis_tag.get(r.get("rep_basis", ""), "?")
            # v17: the caption used to claim the whole table was "only sv_usable
            # reps", but the robust-block columns are aggregated over ALL takes -
            # which is why a session with 0/3 usable still printed a pitch value.
            # Every cell now carries the basis and the n actually used.
            cells.append(f"{r['median']:9.2f}({sd:>5s})[{tag}{r['n_used']}]")
        flag = "" if any_row["enough_reps"] else "   <- too few usable reps"
        print(f"{date:>9s} {cond:10s} {n:>7s}  " + "  ".join(cells) + flag)
    print("SD over repetitions = your measurement noise. A condition difference is "
          "only interpretable if it exceeds roughly 2.8x that SD (sdc_95 column).")

    robust = ("voiced_fraction", "sv_analyzed_fraction",
              "sv_steady_frame_fraction", "sv_cpps_token_db",
              "f0_subharmonic_lock")
    if any(r["metric"] in robust for r in rows):
        print("\nSAME SESSIONS, ROBUST BLOCK  -  median over ALL repetitions "
              "(no take excluded)")
        hdr = f"{'date':>9s} {'condition':10s} {'usable':>7s}  " + "  ".join(
            f"{m.replace('sv_',''):>18s}" for m in robust)
        print(hdr); print("-" * len(hdr))
        for (date, cond) in sessions:
            sub = {r["metric"]: r for r in rows
                   if r["date"] == date and r["condition"] == cond}
            any_row = next(iter(sub.values()))
            cells = []
            for m in robust:
                r = sub.get(m)
                cells.append(f"{r['median']:18.2f}" if r else f"{'-':>18s}")
            print(f"{date:>9s} {cond:10s} "
                  f"{str(any_row['n_usable'])+'/'+str(any_row['n_reps']):>7s}  "
                  + "  ".join(cells))
        print("These are defined for every take, so they are the only lines you can "
              "compare between sessions without selection bias. A session with a low "
              "usable count is not missing data - the low count IS the observation.")
        print("sv_analyzed_fraction = seconds measured / seconds phonated; "
              "sv_steady_frame_fraction = fraction of the vowel on which a valid window "
              "could be centred at all. A session where these two fall is a session where "
              "the voice stopped holding still, whether or not jitter came out.")


def make_per_stim_outputs(results, output_folder, output_prefix,
                          make_individual_plots=True, make_overview=True,
                          plot_scope="core", on_aggregation="all"):
    plots_dir = output_folder / "plots_per_stim"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels, tables = build_per_stim_columns(results, on_aggregation=on_aggregation)
    if not tables:
        print("plots_per_stim: nothing to aggregate."); return plots_dir
    if make_individual_plots:
        _make_bar_plots_from_tables(
            labels, tables, plots_dir,
            metrics=(core_metrics_for(results) if plot_scope == "core" else None),
            task=_task_of(results))
    if make_overview:
        _make_overview_grid_from_tables(labels, tables,
                                        plots_dir / f"{output_prefix}_per_stim_overview.png",
                                        metrics=overview_metrics_for(results),
                                        task=_task_of(results))
    _write_per_stim_csv(labels, tables,
                        plots_dir / f"{output_prefix}_per_stim_all_metrics.csv",
                        task=_task_of(results))
    print(f"\nplots_per_stim written to: {plots_dir}")
    return plots_dir


_DATE_PATTERNS = (
    # (regex, group order as (year, month, day))
    (re.compile(r'(?<!\d)(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)'), (1, 2, 3)),
    (re.compile(r'(?<!\d)(\d{1,2})[-_.](\d{1,2})[-_.](\d{4})(?!\d)'), (3, 2, 1)),
    (re.compile(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)'), (1, 2, 3)),
)


def extract_date_from_filename(file_path: str) -> str:
    """
    Session date as a sortable YYYYMMDD string, or "99999999" when the filename
    carries no date.

    v16 BUG FIX. This used to be a bare search for eight consecutive digits, so
    it matched 20240314 but NOT 2024-03-14, 2024_03_14 or 14.03.2024 - which are
    how people actually name recordings. Every dashed-date file therefore
    returned "99999999", and since this value is the SESSION KEY in
    build_per_stim_columns() and write_session_csv(), all of them collapsed into
    one pseudo-session: different days were silently averaged together, and the
    "SD across repetitions" that the whole session table exists to provide became
    an SD across unrelated sessions. Separated forms are now accepted, and the
    day-first form is disambiguated by the position of the 4-digit year.

    Ambiguity note: 03-04-2024 is read as day-first (4 April), because a 4-digit
    year in last position implies the European order. If your files are
    US-ordered (month-first) with a separated date, rename them to ISO
    YYYY-MM-DD - it is the only form no parser can misread.
    """
    stem = Path(file_path).stem
    for rx, (gy, gm, gd) in _DATE_PATTERNS:
        m = rx.search(stem)
        if not m:
            continue
        try:
            y, mo, d = int(m.group(gy)), int(m.group(gm)), int(m.group(gd))
        except (TypeError, ValueError):
            continue
        if 1900 <= y <= 2999 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}{mo:02d}{d:02d}"
    return "99999999"


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_files(wav_files, output_folder, output_prefix="speech_analysis",
                  make_individual_plots=True, make_overview=True,
                  task="auto", min_vowel_token_dur=0.8,
                  vowel_edge_trim_s=0.25, vowel_window_max_s=2.0,
                  token_bridge_gap_s=0.25, token_min_voiced_fraction=0.5,
                  split_at_splices=False, splice_f0_step_st=1.5,
                  vowel_window_hop_s=3.0, vowel_window_grid_hop_s=None,
                  window_min_voiced_fraction=0.90,
                  window_max_f0_deviation_st=3.0, window_max_internal_step_st=1.0,
                  plot_scope="core",
                  # ---- v9 ----
                  pitch_range=None, octave_check=True, octave_autocorrect=True,
                  pitch_calibration_file=None, recalibrate=False,
                  calibration_on_ambiguous="as_measured",
                  min_valid_windows=2, min_measured_s=4.0,
                  boundaries_dir=None, boundaries_suffix="_tokens.csv",
                  on_aggregation="all",
                  # ---- v10 ----
                  single_token_per_file=True, formant_ceiling_hz=None,
                  octave_strong_db=-10.0, min_usable_reps=2,
                  legacy_compat=False, voicing_threshold=0.45,
                  enable_fluency_index=False):
    """
    task: "auto" (route per file from folder/filename, then acoustics),
          "sustained_vowel", or "reading". Set it explicitly when a folder
          contains one task only - that is the safest option.

    pitch_range: "auto" derives ONE range for the whole batch from the audio
        (see calibrate_pitch_range), caches it in pitch_calibration_file and
        re-uses it verbatim on later runs, so it is still fixed across sessions
        without having to be typed in. A tuple still overrides everything.
        pitch_range: (floor_hz, ceiling_hz) FIXED for every file in the run. Use it
        for a longitudinal / within-subject design: with the range re-estimated
        per file, the jitter/shimmer period bounds and the PowerCepstrogram
        window change between sessions, so part of any PRE/POST difference is a
        settings difference. Set it once per subject from a session you have
        verified by ear (e.g. (120, 320) for a ~190 Hz voice).
    octave_check / octave_autocorrect: verify the tracked F0 against the
        harmonic comb of the signal, and (if autocorrect) fix a clear
        subharmonic/doubling error. Marginal cases are flagged, never corrected.
    min_valid_windows / min_measured_s: usability gate for sustained files.
        NOTE the geometry these two imply: with non-overlapping windows a take
        needs min_valid_windows * vowel_window_max_s + 2 * vowel_edge_trim_s
        seconds of PERFECTLY steady phonation to pass at all (3 x 2.0 s + 0.5 s
        = 6.5 s), and every rejected window slot costs another
        vowel_window_max_s. Check sv_analyzed_fraction and
        sv_steady_frame_fraction before lowering them: if the frames are steady
        and the gate still fails, the gate is the problem, not the voice.
    vowel_window_grid_hop_s: step of the candidate-window grid (default None =
        vowel_window_max_s / 2, i.e. unchanged). A finer step, e.g. 0.5 s, lets a
        window slide around a short wobble instead of losing the whole slot, and
        raises the yield without touching any validity criterion.
    boundaries_dir / boundaries_suffix: read token boundaries from
        <stem><suffix> (CSV "onset,offset" per line) or <stem>.TextGrid instead
        of inferring them. Strongly preferred when you cut the tokens yourself.
    on_aggregation: "all" | "median" | "best" (see build_per_stim_columns).
    """
    wav_files = expand_to_wav_files(wav_files)
    if not wav_files:
        print("No .wav files to analyze."); return []
    output_path = Path(output_folder); output_path.mkdir(parents=True, exist_ok=True)
    # ---- v18: pitch_range="auto" -> derive it from the audio, once ----------
    # The hand-entered constant is the single most dangerous setting in this
    # script: when it is an octave off, or simply left over from another
    # subject, every file comes out "aperiodic" and nothing in the per-file
    # output points at the cause. "auto" derives one range for the whole batch
    # and caches it, so it is still FIXED across sessions (which the
    # longitudinal design requires) without having to be remembered.
    calibration = None
    if isinstance(pitch_range, str):
        if pitch_range.lower() not in ("auto", "calibrate"):
            raise ValueError("pitch_range must be (floor, ceiling), None, or 'auto'")
        cache = pitch_calibration_file or str(
            output_path / f"{output_prefix}_pitch_range.json")
        calibration = calibrate_pitch_range(
            wav_files, cache_path=cache, recalibrate=recalibrate,
            on_ambiguous=calibration_on_ambiguous)
        if calibration.get("pitch_range"):
            pitch_range = tuple(calibration["pitch_range"])
            if calibration.get("confidence") != "high":
                print("\n  !! CALIBRATION FLAGGED (confidence "
                      f"{calibration.get('confidence')}). The range above WILL be used "
                      "for every file in this run. Read the warnings, verify one file "
                      "by ear, and if the octave is wrong delete the calibration JSON "
                      "and set pitch_range by hand - a wrong floor does not look like "
                      "an error, it looks like a severely dysphonic patient.\n")
        else:
            pitch_range = None
            print("  !! calibration produced no range; falling back to per-file "
                  "estimation. Sessions are then NOT strictly comparable.")
    analyzer = CompleteSpeechAnalyzer(
        pitch_floor=50.0, pitch_ceiling=400.0,
        voice_report_pitch_floor=75.0, voice_report_pitch_ceiling=500.0,
        silence_threshold=0.03, voicing_threshold=voicing_threshold,
        max_period_factor=1.3, max_amplitude_factor=1.6,
        min_pause_duration=0.2, noise_margin_db=6.0,
        min_filler_duration=0.15, max_filler_duration=1.5,
        max_formant_hz=5500.0, num_formants=5,
        task=task, min_vowel_token_dur=min_vowel_token_dur,
        vowel_edge_trim_s=vowel_edge_trim_s, vowel_window_max_s=vowel_window_max_s,
        token_bridge_gap_s=token_bridge_gap_s,
        token_min_voiced_fraction=token_min_voiced_fraction,
        split_at_splices=split_at_splices, splice_f0_step_st=splice_f0_step_st,
        vowel_window_hop_s=vowel_window_hop_s,
        vowel_window_grid_hop_s=vowel_window_grid_hop_s,
        window_min_voiced_fraction=window_min_voiced_fraction,
        window_max_f0_deviation_st=window_max_f0_deviation_st,
        window_max_internal_step_st=window_max_internal_step_st,
        pitch_range=pitch_range, octave_check=octave_check,
        octave_autocorrect=octave_autocorrect,
        min_valid_windows=min_valid_windows, min_measured_s=min_measured_s,
        boundaries_dir=boundaries_dir, boundaries_suffix=boundaries_suffix,
        single_token_per_file=single_token_per_file,
        enable_fluency_index=enable_fluency_index,
        formant_ceiling_hz=formant_ceiling_hz, octave_strong_db=octave_strong_db,
        legacy_compat=legacy_compat)
    if legacy_compat:
        print("\n*** legacy_compat=True: agg() uses the pre-v9 one-sided trim and writes "
              "0.0 where nothing was measured, and the formant check is relative-only. "
              "This exists ONLY to reproduce an older run. Do NOT use these numbers for "
              "new analyses. ***\n")
    results = []
    for wav_file in wav_files:
        if not Path(wav_file).exists():
            print(f"WARNING: File not found: {wav_file}"); continue
        try:
            results.append(analyzer.analyze(wav_file))
        except Exception as e:
            print(f"ERROR analyzing {wav_file}: {e}")
            import traceback; traceback.print_exc(); continue
    if not results:
        print("No files were successfully analyzed."); return []
    results.sort(key=lambda r: extract_date_from_filename(r.file_path))
    tasks = sorted({getattr(r, "task", "") for r in results})
    if len(tasks) > 1:
        print("\nWARNING: this batch mixes tasks (" + ", ".join(tasks) + "). Metrics that "
              "do not apply to a file are blank in the CSV and skipped in the plots, but "
              "columns are then not comparable across files. Prefer one run per task.")
    else:
        print(f"\nTask for all files: {tasks[0]}")

    # ---- BATCH-LEVEL F0 CONSISTENCY (v9) --------------------------------------
    # These files are usually repeated sessions of ONE subject. Habitual F0 moves
    # by a couple of semitones across sessions, not by an octave, so a bimodal
    # distribution with a ~2:1 ratio is a tracking artefact, not a finding. This
    # is the check that would have caught the 95 Hz / 190 Hz split immediately.
    f0s = np.array([r.pitch_metrics.median_hz for r in results
                    if np.isfinite(getattr(r.pitch_metrics, "median_hz", np.nan))
                    and r.pitch_metrics.median_hz > 0], dtype=float)
    if f0s.size >= 2:
        spread_st = float(12.0 * np.log2(np.max(f0s) / np.min(f0s)))
        print(f"\nF0 across the batch: {np.min(f0s):.0f}-{np.max(f0s):.0f} Hz "
              f"(spread {spread_st:.1f} st, median {np.median(f0s):.0f} Hz)")
        if spread_st >= 8.0:
            print("  !! F0 SPREAD >= 8 semitones across files. If these are sessions of "
                  "the SAME subject this is almost certainly an octave/subharmonic "
                  "tracking error, not a physiological change - a 2:1 ratio especially. "
                  "Check octave_verdict in the CSV, verify one file of each cluster by "
                  "ear or on a narrowband spectrum, then re-run with "
                  "pitch_range=(floor, ceiling) fixed for this subject.")
        elif spread_st >= 4.0:
            print("  ! F0 spread >= 4 semitones across files: plausible but worth a look, "
                  "especially if it correlates with condition.")
    # ---- CHANNEL HETEROGENEITY ACROSS THE BATCH (v17) ---------------------
    # A recording-chain difference between sessions is indistinguishable from a
    # treatment effect in every spectral and cepstral measure, and it is the one
    # confound no amount of within-file care can fix. CPPS, CPP, alpha ratio,
    # Hammarberg and spectral tilt are all computed over a frequency range, so if
    # one session was captured through a wider channel it will score differently
    # for the same voice. Observed in a real DBS batch: the session with the best
    # CPPS (19.1 dB) was also the only one with ~3000 Hz effective bandwidth while
    # every other session sat at ~1400-1600 Hz.
    #
    # This does NOT prove a confound - a clearer voice genuinely does put more
    # energy into the upper harmonics, so bandwidth and CPPS are expected to
    # correlate somewhat. It tells you to go and check, and which column to check.
    def _fin(attr, src="recording_quality"):
        out = []
        for r in results:
            v = getattr(getattr(r, src), attr, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                out.append(v)
        return np.array(out, dtype=float)

    srs = _fin("sample_rate_hz")
    if srs.size and np.unique(srs).size > 1:
        print(f"\n  !! MIXED SAMPLE RATES in this batch: "
              f"{', '.join(f'{v:.0f}' for v in np.unique(srs))} Hz. This is a "
              f"recording-chain difference, not a voice difference. Resample every "
              f"file to the LOWEST rate present before comparing any spectral or "
              f"cepstral metric (CPPS, CPP, alpha ratio, Hammarberg, tilt).")
    bws = _fin("effective_bandwidth_hz")
    edges = _fin("spectral_edge_hz")
    if bws.size >= 2:
        ratio = float(np.max(bws) / np.min(bws))
        print(f"\nEffective bandwidth across the batch: {np.min(bws):.0f}-"
              f"{np.max(bws):.0f} Hz (ratio {ratio:.2f})")
        if ratio >= 1.5:
            print("  !! BANDWIDTH VARIES BY >=1.5x ACROSS FILES. Before reading any "
                  "CPPS / CPP / alpha-ratio / Hammarberg / tilt difference as a voice "
                  "change, rule out the recording chain:")
            print("     1. rec_sample_rate_hz and rec_spectral_edge_hz in the AUDIT "
                  "block - a HARD edge at a fixed frequency (8000/11025/16000 Hz) that "
                  "differs between groups is a codec or resampling artifact, full stop.")
            print("     2. check whether alpha_ratio_db, hammarberg_index and "
                  "spectral_tilt_db_per_khz all shift TOGETHER with bandwidth. Moving "
                  "as a block is a channel signature; moving independently is voice.")
            print("     3. compare file metadata directly (ffprobe / soxi): codec, "
                  "bitrate, channels, and whether anything was re-encoded.")
            print("     4. listen to one file from the widest and one from the "
                  "narrowest group back to back.")
            print("     If it IS the chain: low-pass every file to the narrowest "
                  "common bandwidth and re-run, or restrict comparisons to sessions "
                  "sharing a chain. Do not compare cepstral measures across it.")
        elif ratio >= 1.25:
            print("  ! Bandwidth varies by >=1.25x: check rec_spectral_edge_hz in the "
                  "AUDIT block if any cepstral/spectral difference looks interesting.")
    if edges.size >= 2 and np.unique(np.round(edges, -2)).size > 1:
        print(f"  Spectral edge across the batch: {np.min(edges):.0f}-"
              f"{np.max(edges):.0f} Hz (a fixed differing edge = codec/resample).")

    n_unusable = sum(1 for r in results
                     if getattr(r, "task", "") == TASK_SUSTAINED
                     and not getattr(r.sustained_metrics, "usable", False))
    n_win_bad = sum(1 for r in results
                    if getattr(r, "task", "") == TASK_SUSTAINED
                    and not getattr(r.sustained_metrics, "windows_usable", False))
    n_pert_only = sum(1 for r in results
                      if getattr(r, "task", "") == TASK_SUSTAINED
                      and getattr(r.sustained_metrics, "windows_usable", False)
                      and not getattr(r.sustained_metrics, "perturbation_usable", False))
    if n_unusable:
        print(f"\n  ! {n_unusable} of {len(results)} sustained file(s) failed the "
              f"combined gate (sv_usable=0), of which:")
        print(f"      {n_win_bad} have unusable WINDOW medians (sv_windows_usable=0): "
              f"exclude from the cepstral block too.")
        print(f"      {n_pert_only} are usable for CPPS/CPP and spectral measures but "
              f"NOT for jitter/shimmer (sv_perturbation_usable=0). Do NOT drop these "
              f"files wholesale - dropping them from the cepstral block selects on "
              f"voice quality and biases the session median.")
        print(f"    See sv_usable_reasons / sv_windows_usable / "
              f"sv_perturbation_usable in the CSV.")

    if make_individual_plots:
        make_bar_plots(results, output_path, output_prefix, plot_scope=plot_scope)
    if make_overview:
        make_overview_grid(results, output_path, output_prefix)
    write_consolidated_csv(results, output_path, output_prefix)
    write_long_csv(results, output_path, output_prefix)
    write_session_csv(results, output_path, output_prefix,
                      min_usable_reps=min_usable_reps)
    make_per_stim_outputs(results, output_path, output_prefix,
                          make_individual_plots=make_individual_plots,
                          make_overview=make_overview, plot_scope=plot_scope,
                          on_aggregation=on_aggregation)
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

# =============================================================================
# SETTINGS-DRIVEN ENTRY POINT
# =============================================================================
# Everything below is new. It replaces the block of hardcoded paths that used
# to live under `if __name__ == "__main__"`. The interface passes a plain dict
# of the upper-case settings; this turns it into an analyze_files() call.

def set_passage_syllables(mapping) -> None:
    """
    Replace the syllable counts used for exact speech-rate computation.

    The counts shipped in KNOWN_PASSAGE_SYLLABLES are for one specific wording
    of each passage. A different edition of the Caterpillar passage changes the
    count, and a wrong count biases the rate for every file of that passage, so
    this is editable from the interface. Keys are matched case-insensitively
    against the filename stem. `None` as a value means "known passage, count
    not established" and falls back to the estimator.
    """
    if not isinstance(mapping, dict):
        raise TypeError("passage syllables must be a dict of name -> count")
    cleaned = {}
    for key, value in mapping.items():
        name = str(key).strip().lower()
        if not name:
            continue
        cleaned[name] = None if value in (None, "", "null") else int(value)
    KNOWN_PASSAGE_SYLLABLES.clear()
    KNOWN_PASSAGE_SYLLABLES.update(cleaned)


def _pitch_range_from_settings(cfg):
    """
    Turn PITCH_RANGE_MODE / PITCH_FLOOR_HZ / PITCH_CEILING_HZ into the single
    `pitch_range` argument analyze_files() expects.

    Three modes, because the original comment block describes three behaviours
    that were selected by editing the constant to a different type:
      "auto"    -> derive once for the batch, cache it, re-use it verbatim
      "manual"  -> a fixed (floor, ceiling) tuple, which overrides everything
      "per_file" -> None, i.e. the pre-v18 behaviour of re-estimating per file
    """
    mode = str(cfg.get("PITCH_RANGE_MODE", "auto")).lower()
    if mode == "manual":
        floor = float(cfg["PITCH_FLOOR_HZ"])
        ceiling = float(cfg["PITCH_CEILING_HZ"])
        if not 0 < floor < ceiling:
            raise ValueError(
                f"pitch floor must be above 0 and below the ceiling "
                f"(got floor={floor}, ceiling={ceiling})")
        return (floor, ceiling)
    if mode == "per_file":
        return None
    return "auto"


def run_acoustics(cfg, log=print):
    """
    Run the acoustic batch described by `cfg` (a dict of upper-case settings).

    Returns the list of SpeechAnalysisResult objects, exactly as
    analyze_files() does. `log` is called with progress strings so the
    interface can stream them; it defaults to print for command-line use.
    """
    inputs = [p for p in cfg.get("WAV_INPUTS", []) if str(p).strip()]
    if not inputs:
        raise ValueError("no input files or folders selected")

    output_folder = str(cfg.get("OUTPUT_FOLDER", "")).strip()
    if not output_folder:
        raise ValueError("no output folder selected")

    task = str(cfg.get("TASK", "")).strip()
    if not task:
        raise ValueError(
            "choose what is in these recordings first: a reading passage, a "
            "sustained vowel, or a mixed folder. The two tasks are measured "
            "differently, so there is no safe default.")
    if task not in {"reading", "sustained_vowel", "auto"}:
        raise ValueError(f"unknown task {task!r}")

    if cfg.get("KNOWN_PASSAGE_SYLLABLES"):
        set_passage_syllables(cfg["KNOWN_PASSAGE_SYLLABLES"])
        log(f"[passages] {len(KNOWN_PASSAGE_SYLLABLES)} syllable counts in use")

    wav_files = expand_to_wav_files(inputs, recursive=bool(cfg.get("RECURSIVE", True)))
    if not wav_files:
        raise ValueError(f"no .wav files found under: {', '.join(map(str, inputs))}")
    log(f"[input] {len(wav_files)} wav file(s)")

    calib = str(cfg.get("PITCH_CALIBRATION_FILE", "")).strip() or None
    boundaries = str(cfg.get("BOUNDARIES_DIR", "")).strip() or None
    grid_hop = cfg.get("VOWEL_WINDOW_GRID_HOP_S", None)
    formant_ceiling = cfg.get("FORMANT_CEILING_HZ", None)

    results = analyze_files(
        wav_files=wav_files,
        output_folder=output_folder,
        output_prefix=cfg.get("OUTPUT_PREFIX", "praat"),
        make_individual_plots=bool(cfg.get("MAKE_INDIVIDUAL_PLOTS", True)),
        make_overview=bool(cfg.get("MAKE_OVERVIEW", True)),
        task=task,
        plot_scope=cfg.get("PLOT_SCOPE", "core"),
        # ---- F0 integrity ----
        pitch_range=_pitch_range_from_settings(cfg),
        pitch_calibration_file=calib,
        recalibrate=bool(cfg.get("RECALIBRATE", False)),
        calibration_on_ambiguous=cfg.get("CALIBRATION_ON_AMBIGUOUS", "as_measured"),
        formant_ceiling_hz=formant_ceiling,
        octave_check=bool(cfg.get("OCTAVE_CHECK", True)),
        octave_autocorrect=bool(cfg.get("OCTAVE_AUTOCORRECT", True)),
        octave_strong_db=float(cfg.get("OCTAVE_STRONG_DB", -10.0)),
        voicing_threshold=float(cfg.get("VOICING_THRESHOLD", 0.45)),
        # ---- tokenisation ----
        single_token_per_file=bool(cfg.get("SINGLE_TOKEN_PER_FILE", True)),
        split_at_splices=bool(cfg.get("SPLIT_AT_SPLICES", False)),
        splice_f0_step_st=float(cfg.get("SPLICE_F0_STEP_ST", 1.5)),
        min_vowel_token_dur=float(cfg.get("MIN_VOWEL_TOKEN_DUR", 0.8)),
        token_bridge_gap_s=float(cfg.get("TOKEN_BRIDGE_GAP_S", 0.25)),
        token_min_voiced_fraction=float(cfg.get("TOKEN_MIN_VOICED_FRACTION", 0.5)),
        boundaries_dir=boundaries,
        boundaries_suffix=cfg.get("BOUNDARIES_SUFFIX", "_tokens.csv"),
        # ---- analysis windows ----
        vowel_window_max_s=float(cfg.get("VOWEL_WINDOW_MAX_S", 2.0)),
        vowel_window_hop_s=float(cfg.get("VOWEL_WINDOW_HOP_S", 2.0)),
        vowel_window_grid_hop_s=grid_hop,
        vowel_edge_trim_s=float(cfg.get("VOWEL_EDGE_TRIM_S", 0.25)),
        window_min_voiced_fraction=float(cfg.get("WINDOW_MIN_VOICED_FRACTION", 0.90)),
        window_max_f0_deviation_st=float(cfg.get("WINDOW_MAX_F0_DEVIATION_ST", 3.0)),
        window_max_internal_step_st=float(cfg.get("WINDOW_MAX_INTERNAL_STEP_ST", 1.0)),
        # ---- usability gates ----
        min_valid_windows=int(cfg.get("MIN_VALID_WINDOWS", 3)),
        min_measured_s=float(cfg.get("MIN_MEASURED_S", 6.0)),
        min_usable_reps=int(cfg.get("MIN_USABLE_REPS", 2)),
        # ---- outputs ----
        on_aggregation=cfg.get("ON_AGGREGATION", "all"),
        legacy_compat=bool(cfg.get("LEGACY_COMPAT", False)),
        enable_fluency_index=bool(cfg.get("ENABLE_FLUENCY_INDEX", False)),
    )
    log(f"[done] {len(results)} file(s) analysed into {output_folder}")
    return results
