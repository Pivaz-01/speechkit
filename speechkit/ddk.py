"""
DDK batch analysis: diadochokinetic rate and regularity of /pa/ /ta/ /ka/ /pataka/
repetitions, with best-window and Pierce, Cotton & Perry (2013) first-nine-syllable
analyses. Python port of DDK_processing_06_03_26.m.

    from speechkit.ddk import run_ddk
    run_ddk({"DDK_INPUT": "path/to/subject", "DDK_OUTPUT": "path/to/results"})

In the original script every value below was an upper-case constant edited at the
top of the file. In speechkit each one is a setting called DDK_<NAME> (see
speechkit/settings.py), set from the interface or the command line, and
configure() copies those settings onto the constants before a run. The analysis
code itself reads the constants exactly as before, so its results are unchanged.
Details: see "HOW IT WORKS" below.
"""


# ════════════════════════════════════════════════════════════════════════════
#  DEFAULTS  (overwritten by configure() from the DDK_* settings before a run)
# ════════════════════════════════════════════════════════════════════════════
INPUT  = ""      # DDK_INPUT: day folders (0309/, 0311/, ...) + the metadata file
OUTPUT = ""      # DDK_OUTPUT: "" = INPUT/DDK_results. Receives two files:
#   DDK_results_<subject>.xlsx     all results (one sheet per table)
#   DDK_file_check_<subject>.xlsx  missing / unlisted / misplaced audio, metadata problems

# ════════════════════════════════════════════════════════════════════════════
#  METADATA (one row per audio trial: which file, which day, stimulation, syllable)
# ════════════════════════════════════════════════════════════════════════════
METADATA_FILE  = ""      # "" = the single .xlsx/.xls/.csv placed directly in INPUT
METADATA_SHEET = 0       # sheet index or name (Excel only)
META_COLUMNS   = dict(date="SessionDate", audio="Audio", stim="Stim",
                      filename="Filename", cue="CueLabel")   # column names in the sheet
SUBJECT_ID     = ""      # "" = from the metadata file name (ddk_S04.xlsx -> S04)
# Only trials listed in the sheet are analysed; files are located by name anywhere under
# INPUT. Day folders are expected to be named MMDD (e.g. 0309) — a mismatch with
# SessionDate is reported but not fatal.

# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS SETTINGS (defaults = DDK_processing_06_03_26.m)
# ════════════════════════════════════════════════════════════════════════════
# 'SubjectID': (minPeakDistance s, minPeakHeight, minPeakProminence, smoothWindow s).
# Per-subject values are study data, so none ship with the code: enter them in the
# interface (DDK_PEAK_PARAMS) and keep them in a saved preset.
PEAK_PARAMS = {}
DEFAULT_PARAMS = (0.12, 0.10, 0.10, 0.10)
# A SUBJECT_ID also matches a longer key that ends with it ("S04" finds "PD_S04").

TRIPLET_WIN      = 3       # syllables per local-rate window
ENERGY_THRESH    = 0.10    # RMS threshold (normalised envelope) for active speech
BANDPASS_LOW_HZ  = 100
BANDPASS_HIGH_HZ = 4000    # capped at fs/2 - 100, as in MATLAB
FILTER_ORDER     = 4       # total band-pass order (MATLAB designfilt 'FilterOrder')

# ── Best-window analysis ────────────────────────────────────────────────────
BEST_WIN_SEC       = 3.0
BEST_WIN_CRITERION = "max_rate"
#   "max_rate"   : most syllables in the window; ties -> lowest ISI CV -> earliest
#   "min_isi_cv" : most regular window (lowest ISI CV, >= 3 syllables);
#                  ties -> most syllables -> earliest
#   "first"      : window starting at speech onset (set BEST_WIN_SEC = 4.0 to
#                  reproduce the 4-s window of DDK_processing_updatedCode_08_17_26.m)
BEST_WIN_STEP_SEC  = 0.01  # sliding step for candidate windows
# Candidate windows ("max_rate"/"min_isi_cv") must lie inside the active-speech span.

# ── Pierce, Cotton & Perry (2013) — third analysis ─────────────────────────
# Measure (paper, "Data analyses"): selection from the onset of glottal pulsing of
# syllable 1 to that of syllable 10; rate = 9 / duration (duration to 4 dp, rate to 1 dp).
PIERCE_N_SYLL         = 9        # paper: nine syllables (= three complete /pataka/)
PIERCE_MIN_SPEECH_SEC = 4.0      # paper: trials with < 4 s of repetition were excluded
VOICING_BAND_HZ       = (60, 500)   # band carrying glottal pulsing (F0 + low harmonics)
VOICING_SMOOTH_SEC    = 0.005       # smoothing of the voiced-energy envelope
VOICING_ONSET_FRAC    = 0.20        # onset = last rise through 20 % of the syllable's voiced peak
# Variability metrics of the Pierce selection (ISI mean, ISI CV, InstRate CV) use the
# same syllables 1-10; PIERCE_VAR_TIMES sets which time point marks each syllable:
#   "onsets": voicing onsets (the points that define the duration -> ISI mean = duration / 9)
#   "peaks" : envelope peaks of those same syllables (same definition as Whole_ / Best3s_)
PIERCE_VAR_TIMES      = "onsets"

# Statistics (paper, "Data analyses" + "Results"). Unit of analysis = one value per
# session (date x audio) x task; each cue is its own task (/pa/ /ta/ /ka/ /pataka/).
# Factors are session-level columns: Stim, Day, Date, Audio, or any column you add
# to the metadata sheet (it must be constant within a session).
RUN_STATS          = True
GROUP_FACTORS      = ("Stim",)        # paper: ("AgeGroup", "Gender") -> two-way ANOVA
LEVENE_FACTOR      = "Stim"           # paper: "AgeGroup"
STATS_STIM_LEVELS  = ("OFF", "ON")    # Stim levels kept for the stats; () = all
                                      # (PRE / POST / FU then become extra groups)
MEDIAN_SPLIT       = ()               # numeric metadata columns split at the median
                                      # (paper: ("Age",) -> factor 'Age_split')
ANOVA_INTERACTION    = True      # paper tested the age x gender interaction
EXCLUDE_IQR_OUTLIERS = True      # paper: values outside 1.5 x IQR removed before analysis
TRIAL_AGG     = "first"          # paper recorded a single trial per task -> first valid trial;
                                 # "mean" averages all valid trials of the cue in the session
STATS_METRICS = ("Pierce9_DDK_Rate",)   # the paper's measure; any Trials column can be added,
                                        # e.g. "Pierce9_ISI_CV", "Pierce9_InstRate_CV"

# Reliability (paper: 10 % of participants re-measured; ICC(2,1) inter-, ICC(3,1) intra-rater)
RELIABILITY_EXPORT   = False     # True = write OUTPUT/reliability/ (sample sheet + WAVs to measure)
RELIABILITY_FRACTION = 0.10      # share of sessions drawn for manual re-measurement
RELIABILITY_SEED     = 2013
RELIABILITY_FILE     = ""        # filled-in copy of the sample sheet -> ICC sheets in results

# ── Session summary ─────────────────────────────────────────────────────────
CUES     = ("pataka", "tatata", "papapa", "kakaka")
AMR_CUES = ("tatata", "papapa", "kakaka")   # AMR = mean of these cue means

# ── Input files ─────────────────────────────────────────────────────────────
AUDIO_EXTENSIONS = (".mat", ".wav")
MAT_AUDIO_VAR    = "audioDataSegment"   # variable names inside the .mat trials
MAT_FS_VAR       = "fs"

# ── Optional extra outputs (all off = only the two files above) ─────────────
SAVE_PLOTS        = False  # True = ONE pdf (DDK_plots_<subject>.pdf), a diagnostic page per trial
SAVE_MATLAB_FILES = False  # True = per-session CSVs + DDK_mat/ files for the old MATLAB scripts
N_WORKERS         = 0      # 0 = auto (all CPU cores but one); 1 = run serially

# ── Audio segments ──────────────────────────────────────────────────────────
SAVE_SEGMENTS   = True   # True = WAV of the selected portions in OUTPUT/"best3s segments"
                         #        and OUTPUT/"pierce9 segments" (one file per trial)
SEGMENT_PAD_SEC = 0.0    # margin added before/after each segment (0 = exactly the selection)


# ════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════
# DDK batch analysis — Python port of DDK_processing_06_03_26.m
# ================================================================
#
# In speechkit: INPUT and OUTPUT are the DDK input and output folders chosen in the
# interface (or `python -m speechkit ddk INPUT -o OUTPUT`); every constant above is
# the setting of the same name with a DDK_ prefix.
#
# INPUT holds one folder per recording day (MMDD, e.g. 0309/) and one metadata sheet
# (e.g. ddk_S04.xlsx) with a row per trial: SessionDate, Audio, Stim, Filename, CueLabel.
# The sheet decides what is analysed: each listed file is located by name anywhere under
# INPUT, a session is one SessionDate x Audio recording, and Stim / CueLabel come from the
# sheet. Missing files, files present but not listed, and files sitting in a day folder
# that does not match their SessionDate are reported in
# OUTPUT/DDK_file_check_<subject>.xlsx.
#
# Every listed trial runs the same pipeline as the MATLAB script:
#
#     band-pass 100–4000 Hz (Butterworth, order 4, zero-phase)
#     -> Hilbert envelope -> moving mean (smoothWindow) -> normalise to max
#     -> peak picking (MinPeakHeight / MinPeakProminence / MinPeakDistance,
#        applied in MATLAB findpeaks order)
#     -> active-speech span from 20 ms / 10 ms RMS frames > ENERGY_THRESH
#     -> DDK_Rate, ISI mean, ISI CV, InstRate CV (3-syllable windows)
#
# In parallel, each trial gets two more analyses:
#
#   * Best window: the BEST_WIN_SEC (default 3 s) window, re-using the whole-trial
#     peaks (as the 08_17_26 script does for its 4-s window). Rate = syllables in
#     window / window length (needs >= 2 syllables); ISI CV needs >= 3 syllables.
#
#   * Pierce, Cotton & Perry (2013), Int J Lang Commun Disord 48:257-264, replicated:
#     - Measure: the FIRST NINE syllables, from the onset of glottal pulsing of syllable 1
#       to that of syllable 10; rate = 9 / duration (duration 4 dp, rate 1 dp). No syllable
#       is discarded and no "fastest" segment is searched for. The paper marked onsets by
#       hand in Praat (waveform + 0-5000 Hz spectrogram); here they are detected as the rise
#       of low-frequency (glottal) energy in each syllable, which voiceless bursts and
#       aspiration lack. Each plot shows the same waveform + spectrogram view for checking.
#     - Variability on the same nine syllables: ISI mean, ISI CV and InstRate CV from the
#       ten time points of syllables 1-10 (voicing onsets or envelope peaks, PIERCE_VAR_TIMES).
#     - Inclusion: >= 10 syllables and >= 4 s of repetition (shorter trials are measured
#       but excluded from the statistics, as in the paper).
#     - One trial per task (the paper recorded a single trial): first valid trial of each
#       cue per session (TRIAL_AGG).
#     - Normality screening per task (skewness, kurtosis with SPSS standard errors; histogram
#       + frequency polygon + box plot figure); values outside 1.5 x IQR removed.
#     - Normative table in the layout of the paper's Table 3 (mean 1 dp, SD 2 dp, range).
#     - Levene's F test for equality of variances (paper's Table 4).
#     - Independent-groups ANOVA per task (two-way with two factors, Type III SS) with
#       partial eta^2, reported as "F(df1,df2) = ..., p = ...".
#     - Reliability: a random RELIABILITY_FRACTION of sessions is exported as WAV for blind
#       manual measurement; the filled sheet gives ICC(2,1) and ICC(3,1) vs the automatic
#       measure (paper: ICC(2,1) inter-rater = 0.996, ICC(3,1) intra-rater = 0.985).
#     - Factors come from the metadata (default Stim, ON vs OFF); any session-level column
#       added to the sheet can be used; MEDIAN_SPLIT reproduces the paper's age split.
#
# Outputs — OUTPUT receives two files:
#     DDK_results_<subject>.xlsx      ALL results, one sheet per table:
#         Info          run summary, sheet guide, column-prefix legend, every setting used
#         Trials        one row per trial: Whole_ / Best3s_ / Pierce9_ analyses side by side
#         Sessions      per session x cue means (+ AMR), like the 08_17_26 script
#         Table3, ANOVA, Levene, Normality (+ distribution figure), Descriptives
#         Reliability_ICC, Reliability_trials     (only when RELIABILITY_FILE is set)
#     DDK_file_check_<subject>.xlsx   Summary (count per issue type) + Issues (one row each):
#         missing audio, unreadable audio, unlisted audio, wrong day folder, duplicates,
#         metadata problems (with the Excel row number)
# Optional (off by default): SAVE_PLOTS -> one PDF with a page per trial;
# SAVE_MATLAB_FILES -> per-session CSVs + DDK_mat/ for the MATLAB scripts;
# RELIABILITY_EXPORT -> OUTPUT/reliability/ (sample sheet + WAVs to measure in Praat).
# SAVE_SEGMENTS (on by default) -> OUTPUT/"best3s segments"/ and OUTPUT/"pierce9 segments"/:
# one 16-bit WAV per trial with the portion selected by each analysis.
#
# Requirements: numpy, scipy, pandas, matplotlib, openpyxl (to read the .xlsx sheet);
# h5py only for MATLAB -v7.3 .mat files


# ════════════════════════════════════════════════════════════════════════════
#  IMPLEMENTATION
# ════════════════════════════════════════════════════════════════════════════
import io
import multiprocessing
import os
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.stats
import scipy.signal as sps
from scipy.io import wavfile



class DDKError(ValueError):
    """A problem with the input or the settings, worded for the user. Raised where
    the original script called sys.exit(), so the interface can report it instead of
    the worker thread dying silently."""


def _log(*parts):
    """Progress output. print() on the command line, the job log in the interface."""
    _LOG(" ".join(str(p) for p in parts))


_LOG = print


def _derive_labels():
    """Names that depend on the settings; recomputed by configure()."""
    global WIN_LABEL, PIERCE_LABEL, BEST_SEG_DIR, PIERCE_SEG_DIR, SKIP_DIR_NAMES
    WIN_LABEL = f"{'First' if BEST_WIN_CRITERION == 'first' else 'Best'}{BEST_WIN_SEC:g}s"
    PIERCE_LABEL = f"Pierce{PIERCE_N_SYLL}"
    BEST_SEG_DIR   = f"{WIN_LABEL.lower()} segments"      # "best3s segments"
    PIERCE_SEG_DIR = f"{PIERCE_LABEL.lower()} segments"   # "pierce9 segments"
    SKIP_DIR_NAMES = {"DDK_plots", "DDK_mat", BEST_SEG_DIR, PIERCE_SEG_DIR}


_derive_labels()


# ── MATLAB-compatible numerics ──────────────────────────────────────────────
def mround(x, n=0):
    """MATLAB round(x, n): half away from zero; NaN passes through."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(x):
        return x
    f = 10.0 ** n
    return float(np.sign(x) * np.floor(abs(x) * f + 0.5) / f)


def mstd(x):
    """MATLAB std(): N-1 normalisation, but 0 for a single element."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    return float(np.std(x, ddof=1))


def movmean_shrink(x, k):
    """MATLAB movmean(x, k) with default 'shrink' endpoints.
    Even k: window covers k/2 samples before and k/2-1 after the current one."""
    x = np.asarray(x, dtype=float)
    n, k = x.size, int(k)
    if k <= 1 or n == 0:
        return x.copy()
    before, after = ((k - 1) // 2, (k - 1) // 2) if k % 2 else (k // 2, k // 2 - 1)
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(idx - before, 0)
    hi = np.minimum(idx + after, n - 1) + 1
    return (c[hi] - c[lo]) / (hi - lo)


def _local_maxima_matlab(y):
    """Port of findpeaks' findLocalMaxima: first index of a plateau, no endpoints."""
    yb = np.concatenate(([np.nan], y, [np.nan]))
    fin = ~np.isnan(yb)
    neq = (yb[:-1] != yb[1:]) & (fin[:-1] | fin[1:])
    itemp = np.concatenate(([0], np.flatnonzero(neq) + 1))
    with np.errstate(invalid="ignore"):
        s = np.sign(np.diff(yb[itemp]))
        imax = 1 + np.flatnonzero(np.diff(s) < 0)
    return itemp[imax] - 1


def findpeaks_matlab(y, min_height, min_prom, min_dist_samples):
    """MATLAB findpeaks(y,'MinPeakHeight',h,'MinPeakProminence',p,'MinPeakDistance',d)
    with x = sample index. Order matters and differs from scipy.find_peaks:
    height -> prominence -> distance (largest peaks kept first, inclusive)."""
    y = np.asarray(y, dtype=float)
    locs = _local_maxima_matlab(y)
    locs = locs[y[locs] > min_height]
    if locs.size and min_prom > 0:
        prom = sps.peak_prominences(y, locs)[0]
        locs = locs[prom >= min_prom]
    if locs.size > 1 and min_dist_samples > 0:
        order = np.argsort(-y[locs], kind="stable")
        lt = locs[order]
        delete = np.zeros(lt.size, dtype=bool)
        for i in range(lt.size):
            if not delete[i]:
                delete |= (lt >= lt[i] - min_dist_samples) & (lt <= lt[i] + min_dist_samples)
                delete[i] = False
        locs = np.sort(lt[~delete])
    return locs, y[locs]


@lru_cache(maxsize=None)
def bandpass_sos(fs):
    hi = min(BANDPASS_HIGH_HZ, fs / 2 - 100)
    return sps.butter(FILTER_ORDER // 2, [BANDPASS_LOW_HZ, hi], btype="bandpass",
                      fs=fs, output="sos")


# ── Signal processing ───────────────────────────────────────────────────────
def compute_envelope(audio, fs, smooth_win):
    filtered = sps.sosfiltfilt(bandpass_sos(float(fs)), audio)
    env = np.abs(sps.hilbert(filtered))
    m = env.max() if env.size else 0.0
    if not np.isfinite(m) or m <= 0:
        raise ValueError("silent or empty audio")
    env = movmean_shrink(env / m, mround(smooth_win * fs))   # pre-scaling is exact (linear)
    return env / env.max()


def active_speech(env, fs, dur_sec):
    flen, hop = int(mround(0.02 * fs)), int(mround(0.01 * fs))
    nfr = (env.size - flen) // hop + 1
    if nfr >= 1:
        c2 = np.concatenate(([0.0], np.cumsum(env ** 2)))
        st = np.arange(nfr) * hop
        rms = np.sqrt(np.maximum(c2[st + flen] - c2[st], 0) / flen)
        act = np.flatnonzero(rms > ENERGY_THRESH)
        if act.size:
            start = act[0] * hop / fs
            end = act[-1] * hop / fs + flen / fs
            return start, end, end - start
    return 0.0, dur_sec, dur_sec


def interval_metrics(locs_sec, min_n_for_isi_cv=2):
    """ISIs (ms), ISI mean, ISI CV, local triplet rates and their CV."""
    n = locs_sec.size
    out = dict(ISIs=np.array([np.nan]), ISI_mean=np.nan, ISI_cv=np.nan,
               instRates=np.array([np.nan]), instTimes=np.array([np.nan]), instRate_cv=np.nan)
    if n >= 2:
        isis = np.diff(locs_sec) * 1000.0
        out["ISIs"], out["ISI_mean"] = isis, float(isis.mean())
        if n >= min_n_for_isi_cv:
            out["ISI_cv"] = mstd(isis) / out["ISI_mean"]
    if n >= TRIPLET_WIN:
        w = np.lib.stride_tricks.sliding_window_view(locs_sec, TRIPLET_WIN)
        rates = (TRIPLET_WIN - 1) / (w[:, -1] - w[:, 0])
        out["instRates"], out["instTimes"] = rates, w.mean(axis=1)
        out["instRate_cv"] = mstd(rates) / float(rates.mean())
    return out


# v1.2: change WITHIN the trial. Mean rate and ISI CV describe a trial as a
# whole and cannot show a speaker who accelerates (oral festination, a
# hypokinetic sign) or whose syllables fade (articulatory force decay). Both
# are measured on the whole trial only: a 3-s window or nine syllables are too
# short for a trend. The envelope is normalised to each trial's maximum, so
# the amplitude measures are RELATIVE (CV, % of mean per second); absolute
# peak heights are not comparable between trials.
MIN_TREND_POINTS = 5


def _slope(x, y):
    """Least-squares slope of y on x; NaN with too few points or no spread."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < MIN_TREND_POINTS or np.ptp(x) <= 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def trend_metrics(locs_sec, pks, inst_rates, inst_times):
    """InstRate_Slope (syll/s per s; > 0 = accelerating), PeakAmp_CV, and
    PeakAmp_Trend_pct_s (peak amplitude change per second, % of the mean;
    < 0 = syllables getting weaker)."""
    pks = np.asarray(pks, float)
    out = dict(InstRate_Slope=_slope(inst_times, inst_rates),
               PeakAmp_CV=np.nan, PeakAmp_Trend_pct_s=np.nan)
    if pks.size >= MIN_TREND_POINTS and pks.mean() > 0:
        out["PeakAmp_CV"] = mstd(pks) / float(pks.mean())
        sl = _slope(locs_sec, pks)
        out["PeakAmp_Trend_pct_s"] = 100.0 * sl / float(pks.mean()) if np.isfinite(sl) else np.nan
    return out


def _window_cv(locs_sec, i0, i1):
    seg = locs_sec[i0:i1]
    if seg.size < 3:
        return np.inf
    isis = np.diff(seg)
    return mstd(isis) / float(isis.mean())


def select_window(locs_sec, speech_start, speech_end):
    """Return (t0, t1, note) for the analysis window."""
    W = BEST_WIN_SEC
    if BEST_WIN_CRITERION == "first":
        return speech_start, speech_start + W, ""
    if speech_end - speech_start < W:
        return speech_start, speech_start + W, \
            f"active speech {speech_end - speech_start:.2f}s < {W:g}s window"
    last = speech_end - W
    starts = np.arange(speech_start, last, BEST_WIN_STEP_SEC)
    starts = np.append(starts[starts < last], last)
    i0 = np.searchsorted(locs_sec, starts, side="left")
    i1 = np.searchsorted(locs_sec, starts + W, side="right")
    pairs, first = np.unique(np.stack([i0, i1], axis=1), axis=0, return_index=True)
    cands = [(int(b - a), _window_cv(locs_sec, a, b), float(starts[f]))
             for (a, b), f in zip(pairs, first)]
    note = ""
    if BEST_WIN_CRITERION == "min_isi_cv":
        ok = [c for c in cands if np.isfinite(c[1])]
        if ok:
            best = min(ok, key=lambda c: (c[1], -c[0], c[2]))
        else:
            best, note = min(cands, key=lambda c: (-c[0], c[1], c[2])), \
                "no window with >=3 syllables; used max_rate"
    else:
        best = min(cands, key=lambda c: (-c[0], c[1], c[2]))
    return best[2], best[2] + W, note


def window_metrics(locs_sec, t0, t1):
    sel = locs_sec[(locs_sec >= t0) & (locs_sec <= t1)]
    m = interval_metrics(sel, min_n_for_isi_cv=3)
    n = int(sel.size)
    return dict(SyllableCount=n,
                DDK_Rate=n / (t1 - t0) if n >= 2 else np.nan,
                ISI_mean_ms=m["ISI_mean"], ISI_CV=m["ISI_cv"], InstRate_CV=m["instRate_cv"])


# ── Pierce et al. (2013) ────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def voicing_sos(fs):
    lo, hi = VOICING_BAND_HZ
    return sps.butter(2, [lo, min(hi, fs / 2 - 100)], btype="bandpass", fs=fs, output="sos")


def voiced_envelope(audio, fs):
    v = np.abs(sps.hilbert(sps.sosfiltfilt(voicing_sos(float(fs)), audio)))
    m = v.max()
    return movmean_shrink(v / m if m > 0 else v, max(1, int(mround(VOICING_SMOOTH_SEC * fs))))


def voicing_onsets(env, venv, locs, fs, n_needed):
    """Onset of glottal pulsing for the first n_needed syllables (seconds).

    Each syllable spans the envelope minima around its peak. Within it, the onset
    is the point where the voiced (low-band) energy last rises through
    VOICING_ONSET_FRAC of its maximum before that maximum, i.e. where periodic
    pulsing starts after the voiceless burst/aspiration of /p t k/."""
    n = min(n_needed, locs.size)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    dflt = int(mround(0.25 * fs))
    valleys = np.array([locs[i] + int(np.argmin(env[locs[i]:locs[i + 1] + 1]))
                        for i in range(locs.size - 1)], dtype=int)
    for i in range(n):
        if i > 0:
            left = valleys[i - 1]
        else:
            left = max(0, int(locs[0] - (locs[1] - locs[0] if locs.size > 1 else dflt)))
        if i < valleys.size:
            right = valleys[i]
        else:
            right = min(env.size - 1, int(locs[i] + (locs[i] - locs[i - 1] if i > 0 else dflt)))
        seg = venv[left:right + 1]
        j = int(np.argmax(seg))
        below = np.flatnonzero(seg[:j + 1] < VOICING_ONSET_FRAC * seg[j])
        idx = left + (below[-1] + 1 if below.size else 0)
        out[i] = (idx + 1) / fs           # same 1-based convention as locsSec
    return out


def pierce_metrics(onsets, locs_sec):
    """Rate of the first N syllables (paper) + variability of the same syllables.
    Returns (metrics dict, note, ISIs in ms)."""
    N = PIERCE_N_SYLL
    if onsets.size >= N + 1 and np.all(np.isfinite(onsets[:N + 1])):
        dur = mround(onsets[N] - onsets[0], 4)          # paper: 4 decimal places
        times = onsets[:N + 1] if PIERCE_VAR_TIMES == "onsets" else locs_sec[:N + 1]
        m = interval_metrics(times)                     # N ISIs, N-1 triplet rates
        return dict(OnsetFirstSec=onsets[0], OnsetLastSec=onsets[N], DurationSec=dur,
                    DDK_Rate=mround(N / dur, 1),        # paper: 1 decimal place
                    ISI_mean_ms=m["ISI_mean"], ISI_CV=m["ISI_cv"],
                    InstRate_CV=m["instRate_cv"]), "", m["ISIs"]
    return dict(OnsetFirstSec=np.nan, OnsetLastSec=np.nan, DurationSec=np.nan,
                DDK_Rate=np.nan, ISI_mean_ms=np.nan, ISI_CV=np.nan, InstRate_CV=np.nan), \
        f"fewer than {N + 1} syllables detected", np.array([np.nan])


# ── Audio segments ──────────────────────────────────────────────────────────
def write_segment(folder, name, audio, fs, t0, t1, peak):
    """Write audio[t0 - pad : t1 + pad] as 16-bit WAV; returns the file name ('' if nothing written)."""
    if not (np.isfinite(t0) and np.isfinite(t1)):
        return ""
    i0 = max(0, int(round((t0 - SEGMENT_PAD_SEC) * fs)))
    i1 = min(audio.size, int(round((t1 + SEGMENT_PAD_SEC) * fs)))
    if i1 - i0 < 2:
        return ""
    os.makedirs(folder, exist_ok=True)
    seg = audio[i0:i1] / peak * 0.9                # peak = whole-trial peak -> no clipping
    wavfile.write(os.path.join(folder, name), int(round(fs)), np.int16(np.round(seg * 32767)))
    return name


def save_segments(job, audio, fs, res):
    """Best-window and Pierce selections of one trial -> OUTPUT/<label> segments/<session>__<file>.wav"""
    peak = float(np.abs(audio).max()) or 1.0      # same gain for both segments of a trial
    name = f"{job['session_id']}__{job['stem']}.wav"
    b, pc = res["best"], res["pierce"]
    seg = {}
    for key, folder, t0, t1, note_key in (
            ("best", BEST_SEG_DIR, b["WindowStartSec"], b["WindowEndSec"], "best_note"),
            ("pierce", PIERCE_SEG_DIR, pc["OnsetFirstSec"], pc["OnsetLastSec"], "pierce_note")):
        try:
            seg[key] = write_segment(os.path.join(job["out_root"], folder), name,
                                     audio, fs, t0, t1, peak)
        except Exception as e:
            seg[key] = ""
            res[note_key] += ("; " if res[note_key] else "") + f"segment not saved: {e}"
    return seg


# ── File I/O ────────────────────────────────────────────────────────────────
def load_audio(path):
    note = ""
    if path.lower().endswith(".mat"):
        try:
            d = scipy.io.loadmat(path, variable_names=[MAT_AUDIO_VAR, MAT_FS_VAR])
            audio, fs = d[MAT_AUDIO_VAR], d[MAT_FS_VAR]
        except NotImplementedError:
            try:
                import h5py
            except ImportError as e:
                raise RuntimeError("MATLAB v7.3 file: pip install h5py") from e
            with h5py.File(path, "r") as f:
                audio, fs = np.array(f[MAT_AUDIO_VAR]).T, np.array(f[MAT_FS_VAR])
        fs = float(np.asarray(fs).ravel()[0])
    else:
        fs, audio = wavfile.read(path)
        fs = float(fs)
    audio = np.squeeze(np.asarray(audio, dtype=float))
    if audio.ndim == 2:
        if audio.shape[0] < audio.shape[1]:    # channels x samples -> samples x channels
            audio = audio.T
        if audio.shape[1] > 1:
            note = "stereo->mono; "
        audio = audio.mean(axis=1)
    return audio.ravel(), fs, note


def _parse_name(stem):
    m = re.match(r"^(.*?)[_\-](\d+)$", stem)
    trial = int(m.group(2)) if m else None
    a = re.search(r"audio[_\-]?(\d+)", stem, flags=re.I)
    d = re.search(r"(?<!\d)(20\d{6})(?!\d)", stem)
    return trial, (a.group(1) if a else None), (d.group(1) if d else None)


# ── Metadata-driven session building ────────────────────────────────────────
STD_META = ("date", "audio", "stim", "filename", "cue")


def _to_yyyymmdd(v):
    """SessionDate cell -> 'YYYYMMDD' (Excel date, Excel serial number, 20260309 or text)."""
    if isinstance(v, datetime):
        return v.strftime("%Y%m%d")
    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
        if 19000101 <= v <= 21001231:
            return f"{int(v)}"
        if 20000 < v < 80000:
            return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))).strftime("%Y%m%d")
    try:
        return pd.to_datetime(str(v)).strftime("%Y%m%d")
    except Exception:
        return None


def find_metadata_file(root):
    if METADATA_FILE:
        path = METADATA_FILE if os.path.isabs(METADATA_FILE) else os.path.join(root, METADATA_FILE)
        if not os.path.isfile(path):
            raise DDKError(f"METADATA_FILE not found: {path}")
        return path
    cands = [f for f in sorted(os.listdir(root))
             if f.lower().endswith((".xlsx", ".xls", ".csv"))
             and not f.startswith(("~$", ".", "DDK_"))]
    if len(cands) != 1:
        raise DDKError(f"Expected exactly one metadata file (.xlsx/.xls/.csv) directly in INPUT, found "
                 f"{len(cands)}: {cands}. Choose the metadata sheet (DDK_METADATA_FILE).")
    return os.path.join(root, cands[0])


def subject_from(path):
    if SUBJECT_ID:
        return SUBJECT_ID
    m = re.search(r"(DBSS\d+|S\d+)", os.path.basename(path), flags=re.I)
    return m.group(1).upper() if m else "UNKNOWN"


def resolve_params(subject):
    """Exact PEAK_PARAMS key, else a unique key that ends with the ID (S04 -> DBSS04)."""
    if subject in PEAK_PARAMS:
        return subject, PEAK_PARAMS[subject]
    hits = [k for k in PEAK_PARAMS if k.upper().endswith(subject.upper())]
    if len(hits) == 1:
        return hits[0], PEAK_PARAMS[hits[0]]
    return None, DEFAULT_PARAMS


def load_metadata(path, issues):
    meta = (pd.read_excel(path, sheet_name=METADATA_SHEET)
            if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path))
    lower = {str(c).strip().lower(): c for c in meta.columns}
    missing = [n for n in META_COLUMNS.values() if n.strip().lower() not in lower]
    if missing:
        raise DDKError(f"Metadata lacks column(s) {missing}; found {list(meta.columns)}. "
                 f"Set the metadata column names (DDK_COL_*).")
    meta = meta.rename(columns={lower[n.strip().lower()]: k for k, n in META_COLUMNS.items()})
    extra = [c for c in meta.columns if c not in STD_META]
    meta = meta[meta["filename"].notna() & meta["filename"].astype(str).str.strip().ne("")].copy()
    meta["meta_row"] = meta.index + 2                      # row number as seen in Excel
    meta["filename"] = meta["filename"].astype(str).str.strip()
    meta["stem"] = meta["filename"].map(lambda f: os.path.splitext(os.path.basename(f))[0])
    meta["date"] = meta["date"].map(_to_yyyymmdd)
    meta["audio"] = pd.to_numeric(meta["audio"], errors="coerce").astype("Int64")
    meta["stim"] = meta["stim"].fillna("").astype(str).str.strip().str.upper()
    meta["cue"] = meta["cue"].fillna("N/A").astype(str).str.strip().str.lower()
    meta["trial"] = meta["stem"].map(lambda s: _parse_name(s)[0])
    bad = meta["date"].isna() | meta["audio"].isna()
    for _, r in meta[bad].iterrows():
        issues.append(dict(Type="metadata", File=r["filename"], Row=r["meta_row"],
                           Detail="unreadable date or audio number; row skipped"))
    meta = meta[~bad]
    dup = meta["stem"].str.lower().duplicated()
    for _, r in meta[dup].iterrows():
        issues.append(dict(Type="metadata", File=r["filename"], Row=r["meta_row"],
                           Detail="duplicate filename; later row skipped"))
    meta = meta[~dup]
    for _, r in meta.iterrows():                           # filename vs date/audio columns
        _, a, d = _parse_name(r["stem"])
        if (a and int(a) != int(r["audio"])) or (d and d != r["date"]):
            issues.append(dict(Type="metadata", File=r["filename"], Row=r["meta_row"],
                               Detail=f"filename says audio {a}, date {d}; sheet says audio "
                                      f"{r['audio']}, date {r['date']} (sheet used)"))
    for col in ["stim"] + extra:                           # session-level columns must be constant
        n = meta.groupby(["date", "audio"])[col].nunique(dropna=False)
        for (d, a), k in n[n > 1].items():
            issues.append(dict(Type="metadata", File=f"{d} audio {a}", Row="",
                               Detail=f"'{col}' varies within the session; first value used"))
    return meta, extra


def index_audio(root, out_root):
    """stem (lower case) -> paths of every audio file under INPUT."""
    index = {}
    out_inside = os.path.abspath(out_root) != os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")
            and not (out_inside and os.path.abspath(os.path.join(dirpath, d)) == os.path.abspath(out_root)))
        for fn in sorted(filenames):
            stem, ext = os.path.splitext(fn)
            if fn.startswith(".") or ext.lower() not in AUDIO_EXTENSIONS or stem.endswith("_DDK_data"):
                continue
            index.setdefault(stem.lower(), []).append(os.path.join(dirpath, fn))
    return index


def diagnose_missing(issues, root, audio_index, listed_stems):
    """For every missing file: does its day folder exist, is it elsewhere under another
    extension or in a skipped folder, or is there a near-identical name (typo)?"""
    missing = [i for i in issues if i["Type"] == "missing"]
    if not missing:
        return
    import difflib
    every, top_dirs = {}, []
    for dirpath, dirnames, filenames in os.walk(root):          # every file, nothing skipped
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            top_dirs = sorted(dirnames)
        for fn in filenames:
            every.setdefault(os.path.splitext(fn)[0].lower(), []).append(
                os.path.normpath(os.path.join(rel, fn)))
    per_folder = {}
    for paths in audio_index.values():
        for pth in paths:
            t = os.path.relpath(pth, root).split(os.sep)[0]
            per_folder[t] = per_folder.get(t, 0) + 1
    stems = [k for k in audio_index if k not in listed_stems]   # typo candidates: unlisted files only
    for it in missing:
        stem, mmdd = os.path.splitext(it["File"])[0].lower(), it["ExpectedFolder"]
        folders = [d for d in top_dirs if mmdd in d]
        it["FolderFound"] = ", ".join(folders) if folders else "NO"
        it["AudioFilesInFolder"] = sum(per_folder.get(f, 0) for f in folders)
        close = difflib.get_close_matches(stem, stems, n=1, cutoff=0.85)
        it["SimilarFileFound"] = os.path.relpath(audio_index[close[0]][0], root) if close else ""
        elsewhere = every.get(stem, [])
        if elsewhere:
            p0 = elsewhere[0]
            ext = os.path.splitext(p0)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                cause = f"exists as {p0}, but '{ext}' is not in AUDIO_EXTENSIONS"
            else:
                cause = f"exists only inside a skipped folder: {p0}"
        elif not folders:
            cause = f"there is no day folder '{mmdd}' under INPUT"
        elif close:
            cause = f"not found; a near-identical file exists ({os.path.basename(it['SimilarFileFound'])}) " \
                    f"— typo in the sheet or in the file name?"
        else:
            cause = f"folder {it['FolderFound']} exists ({it['AudioFilesInFolder']} audio files) but " \
                    f"this file is not in it, nor anywhere else under INPUT"
        it["Detail"] = cause


def build_sessions(meta, extra, index, root, issues):
    """Sessions = (date, audio) groups of metadata rows, each row matched to its file."""
    sessions = {}
    for _, r in meta.iterrows():
        paths = index.get(r["stem"].lower(), [])
        if not paths:
            d = r["date"]
            issues.append(dict(Type="missing", File=r["filename"], Row=r["meta_row"],
                               Detail="not found under INPUT", SessionDate=f"{d[:4]}-{d[4:6]}-{d[6:]}",
                               Audio=int(r["audio"]), Stim=r["stim"], CueLabel=r["cue"],
                               ExpectedFolder=d[4:]))
            continue
        mmdd = r["date"][4:]
        top = lambda p: os.path.relpath(p, root).split(os.sep)[0]
        if len(paths) > 1:
            pref = [p for p in paths if mmdd in top(p)] or paths
            issues.append(dict(Type="duplicate", File=r["filename"], Row=r["meta_row"],
                               Detail=f"{len(paths)} copies found; using "
                                      f"{os.path.relpath(pref[0], root)}"))
            paths = pref
        path = paths[0]
        rel = os.path.relpath(path, root)
        day = top(path) if os.sep in rel else mmdd
        if os.sep in rel and mmdd not in day:
            issues.append(dict(Type="folder_mismatch", File=rel, Row=r["meta_row"],
                               Detail=f"in folder '{day}' but SessionDate is {r['date']}"))
        sessions.setdefault((r["date"], int(r["audio"])), []).append(dict(
            filename=os.path.basename(path), path=path, rel_dir=os.path.dirname(rel),
            trial=r["trial"], cue=r["cue"], stim=r["stim"], day=day, meta_row=r["meta_row"],
            extra={c: r[c] for c in extra}))
    for files in sessions.values():
        files.sort(key=lambda f: (f["trial"] if f["trial"] is not None else 10**9, f["filename"]))
    listed = set(meta["stem"].str.lower())
    for stem, paths in index.items():
        if stem not in listed:
            for pth in paths:
                issues.append(dict(Type="unlisted", File=os.path.relpath(pth, root), Row="",
                                   Detail="audio file not in the metadata; not analysed"))
    return sessions


# ── Per-trial worker ────────────────────────────────────────────────────────
def analyze_trial(job):
    res = dict(job)
    res.update(note="", best_note="", pierce_note="", pierce_included=False,
               fs=np.nan, file_dur=np.nan)
    try:
        audio, fs, note = load_audio(job["path"])
        res["note"], res["fs"] = note, fs
        dist, height, prom, smooth = job["params"]
        dur = audio.size / fs
        res["file_dur"] = dur
        env = compute_envelope(audio, fs, smooth)
        locs, pks = findpeaks_matlab(env, height, prom, int(mround(dist * fs)))
        locs_sec = (locs + 1) / fs     # MATLAB's 1-based locs/fs (keeps .mat identical)
        start, end, active = active_speech(env, fs, dur)
        n = int(locs.size)
        whole = interval_metrics(locs_sec)
        res.update(SyllableCount=n, DDK_Rate=n / active, ActiveDurationSec=active,
                   SpeechStartSec=start, SpeechEndSec=end, ISI_mean_ms=whole["ISI_mean"],
                   ISI_CV=whole["ISI_cv"], InstRate_CV=whole["instRate_cv"],
                   **trend_metrics(locs_sec, pks, whole["instRates"], whole["instTimes"]))
        t0, t1, wnote = select_window(locs_sec, start, end)
        res["best_note"] = note + wnote
        res["best"] = dict(WindowStartSec=t0, WindowEndSec=t1, **window_metrics(locs_sec, t0, t1))

        onsets = voicing_onsets(env, voiced_envelope(audio, fs), locs, fs, PIERCE_N_SYLL + 1)
        res["pierce"], pnote, pierce_isis = pierce_metrics(onsets, locs_sec)
        if active < PIERCE_MIN_SPEECH_SEC:          # paper's exclusion criterion
            pnote += (("; " if pnote else "") +
                      f"repetition {active:.2f}s < {PIERCE_MIN_SPEECH_SEC:g}s: excluded from stats")
        res["pierce_note"] = note + pnote
        res["pierce_included"] = bool(np.isfinite(res["pierce"]["DDK_Rate"])
                                      and active >= PIERCE_MIN_SPEECH_SEC)
        res["onsets"] = onsets
        if SAVE_SEGMENTS:
            res["segments"] = save_segments(job, audio, fs, res)

        if SAVE_MATLAB_FILES:
            os.makedirs(os.path.join(job["out_dir"], "DDK_mat"), exist_ok=True)
            td = dict(filename=job["filename"], cueLabel=job["cue"], subjectID=job["subject"],
                      condLabel=job["stim"], day=job["day"],
                      dateStr=job["date"] or "", audioNum=job["audio_num"] or "", fs=fs,
                      envelope=env.astype(np.float32), locsSec=locs_sec, ISIs=whole["ISIs"],
                      instRates=whole["instRates"], instTimes=whole["instTimes"],
                      syllCount=float(n), ddkRate=n / active, activeDur=active,
                      speechStart=start, speechEnd=end, ISI_mean_ms=whole["ISI_mean"],
                      ISI_cv=whole["ISI_cv"], instRate_cv=whole["instRate_cv"],
                      **{k[0].lower() + k[1:]: float(res[k]) for k in TREND_COLS},
                      params=dict(minPeakDistance=dist, minPeakHeight=height, minPeakProm=prom,
                                  smoothWindow=smooth, tripletWin=float(TRIPLET_WIN)),
                      bestWin=dict(criterion=BEST_WIN_CRITERION, winSec=BEST_WIN_SEC,
                                   **{k: float(v) for k, v in res["best"].items()}),
                      voicingOnsetsSec=onsets, pierceISIs=pierce_isis,
                      pierce=dict(nSyll=float(PIERCE_N_SYLL), included=float(res["pierce_included"]),
                                  **{k: float(v) for k, v in res["pierce"].items()}))
            scipy.io.savemat(os.path.join(job["out_dir"], "DDK_mat", job["stem"] + "_DDK_data.mat"),
                             td, oned_as="column", do_compression=True)
        if SAVE_PLOTS:                       # rendered in memory -> one PDF in main()
            buf = io.BytesIO()
            save_plot(buf, job, audio, fs, env, locs_sec, pks, res, whole, height)
            res["plot_png"] = buf.getvalue()
    except Exception as e:
        res["note"] += f"ERROR: {e}"
        res["best_note"] = res["pierce_note"] = res["note"]
        res["traceback"] = traceback.format_exc()
    return res


def save_plot(path, job, audio, fs, env, locs_sec, pks, res, whole, height):
    import matplotlib.pyplot as plt
    dur = audio.size / fs
    b = res["best"]
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 0.45, 1.1], hspace=0.55)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    nb = 4000                                   # min/max decimation for the waveform
    if audio.size > 2 * nb:
        edges = np.linspace(0, audio.size, nb + 1).astype(int)
        mn = np.minimum.reduceat(audio, edges[:-1])
        mx = np.maximum.reduceat(audio, edges[:-1])
        ax1.fill_between(edges[:-1] / fs, mn, mx, color=(0.3, 0.5, 0.8), lw=0)
    else:
        ax1.plot(np.arange(audio.size) / fs, audio, color=(0.3, 0.5, 0.8), lw=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{job['filename']} | {job['cue']} | {job['stim']} | day {job['day']} | "
                  f"{job['session_id']}", fontsize=10)

    step = max(1, env.size // 20000)
    ax2.plot(np.arange(0, env.size, step) / fs, env[::step], color=(0.2, 0.6, 0.2), lw=1.2)
    if locs_sec.size:
        ax2.plot(locs_sec, pks, "rv", mfc="r", ms=6)
    ax2.axhline(height, ls="--", color="k", lw=0.8)
    pc, ons = res["pierce"], res["onsets"]
    green = (0.0, 0.5, 0.25)
    ax2.vlines(ons[np.isfinite(ons)], 0, 0.12, color=green, lw=1.4)
    if np.isfinite(pc["DurationSec"]):
        ax2.plot([pc["OnsetFirstSec"], pc["OnsetLastSec"]], [1.1, 1.1], color=green, lw=3,
                 solid_capstyle="butt")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Norm. envelope")
    ax2.set_title(f"Whole: n={res['SyllableCount']}  rate={res['DDK_Rate']:.2f} syl/s  "
                  f"dur={res['ActiveDurationSec']:.2f}s   |   {WIN_LABEL} "
                  f"[{b['WindowStartSec']:.2f}-{b['WindowEndSec']:.2f}s]: n={b['SyllableCount']}  "
                  f"rate={b['DDK_Rate']:.2f}  ISI CV={b['ISI_CV']:.3f}\n"
                  f"{PIERCE_LABEL} (green bar, onset 1 -> {PIERCE_N_SYLL + 1}; ticks = voicing onsets): "
                  f"dur={pc['DurationSec']:.4f}s  rate={pc['DDK_Rate']:.1f} syl/s  "
                  f"ISI CV={pc['ISI_CV']:.3f}  InstRate CV={pc['InstRate_CV']:.3f}", fontsize=9)

    if res["SyllableCount"] >= TRIPLET_WIN:
        ax3.plot(whole["instTimes"], whole["instRates"], "-o", color=(0.6, 0.4, 0.8), ms=4)
        ax3.axhline(res["DDK_Rate"], ls="--", color="k", lw=0.8)
        ax3.set_ylim(0, np.nanmax(whole["instRates"]) * 1.4 + 0.1)
    ax3.set_ylabel("Local rate (syl/s)")
    if res["SyllableCount"] >= 2:
        axr = ax3.twinx()
        axr.vlines(locs_sec[:-1], 0, whole["ISIs"], color=(0.85, 0.45, 0.1), lw=1)
        axr.plot(locs_sec[:-1], whole["ISIs"], "o", color=(0.85, 0.45, 0.1), ms=3)
        axr.set_ylabel("ISI (ms)")
    ax3.set_title(f"InstRate CV={res['InstRate_CV']:.3f} | ISI CV={res['ISI_CV']:.3f} | "
                  f"Mean ISI={res['ISI_mean_ms']:.0f} ms", fontsize=9)
    ax3.set_xlabel("Time (s)")

    for ax in (ax1, ax2, ax3):
        ax.axvspan(res["SpeechStartSec"], res["SpeechEndSec"], color=(1, 0.9, 0.6), alpha=0.3, lw=0)
        ax.axvspan(b["WindowStartSec"], b["WindowEndSec"], color=(0.55, 0.75, 1.0), alpha=0.45, lw=0)
        for xv in (b["WindowStartSec"], b["WindowEndSec"]):
            ax.axvline(xv, color=(0.1, 0.3, 0.8), lw=1.2, ls="--")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, dur)

    # Paper's inspection view: waveform + spectrogram (0-5000 Hz) of the 9-syllable selection
    axw, axs = fig.add_subplot(gs[3]), fig.add_subplot(gs[4])
    if np.isfinite(pc["DurationSec"]):
        t0, t1 = pc["OnsetFirstSec"] - 0.08, pc["OnsetLastSec"] + 0.08
    else:
        t0, t1 = res["SpeechStartSec"], min(dur, res["SpeechStartSec"] + 2.5)
    i0, i1 = max(0, int(t0 * fs)), min(audio.size, int(t1 * fs))
    seg = audio[i0:i1]
    if seg.size > int(0.02 * fs):
        axw.plot((i0 + np.arange(seg.size)) / fs, seg, color=(0.3, 0.3, 0.3), lw=0.4)
        nper = max(16, int(0.005 * fs))              # 5 ms analysis window (Praat-like)
        f, tt, S = sps.spectrogram(seg, fs=fs, window="hann", nperseg=nper,
                                   noverlap=int(nper * 0.8), nfft=max(nper, 1024))
        keep = f <= 5000
        SdB = 10 * np.log10(S[keep] + 1e-20)
        axs.pcolormesh(i0 / fs + tt, f[keep], SdB, cmap="Greys", shading="auto",
                       vmin=SdB.max() - 50, vmax=SdB.max())
        for ax in (axw, axs):
            for k, o in enumerate(ons[np.isfinite(ons)]):
                ax.axvline(o, color=green, lw=1.3 if k in (0, PIERCE_N_SYLL) else 0.7,
                           ls="-" if k in (0, PIERCE_N_SYLL) else ":")
            ax.set_xlim(i0 / fs, i1 / fs)
    axw.set_yticks([])
    axw.set_title(f"{PIERCE_LABEL} selection: onset of glottal pulsing, syllable 1 -> "
                  f"{PIERCE_N_SYLL + 1} (solid green)" +
                  ("" if res["pierce_included"] else "   [NOT INCLUDED: " + res["pierce_note"] + "]"),
                  fontsize=9)
    axs.set_ylabel("Hz")
    axs.set_ylim(0, 5000)
    axs.set_xlabel("Time (s)")
    fig.savefig(path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── Output tables ───────────────────────────────────────────────────────────
DIGITS = dict(DDK_Rate=2, ActiveDurationSec=2, ISI_mean_ms=1, ISI_CV=3, InstRate_CV=3,
              WindowStartSec=2, WindowEndSec=2, SpeechStartSec=2, SpeechEndSec=2,
              InstRate_Slope=3, PeakAmp_CV=3, PeakAmp_Trend_pct_s=2)
TREND_COLS = ("InstRate_Slope", "PeakAmp_CV", "PeakAmp_Trend_pct_s")


def _r(col, v):
    return mround(v, DIGITS[col]) if col in DIGITS else v


def _pierce_names():
    return dict(OnsetFirstSec="Onset1Sec", OnsetLastSec=f"Onset{PIERCE_N_SYLL + 1}Sec",
                DurationSec="DurationSec", DDK_Rate="DDK_Rate",
                ISI_mean_ms="ISI_mean_ms", ISI_CV="ISI_CV", InstRate_CV="InstRate_CV")


def _pierce_val(c, v):
    """Onsets 4 dp; duration and rate already rounded as in the paper; the rest as DIGITS."""
    return mround(v, 4) if c.startswith("Onset") else _r(c, v)


def session_tables(results):
    whole_rows, best_rows, pierce_rows = [], [], []
    pn = _pierce_names()
    for r in results:
        b, pc = r.get("best", {}), r.get("pierce", {})
        whole_rows.append(dict(
            Filename=r["filename"], CueLabel=r["cue"], SyllableCount=r.get("SyllableCount"),
            **{c: _r(c, r.get(c, np.nan)) for c in
               ("DDK_Rate", "ActiveDurationSec", "ISI_mean_ms", "ISI_CV", "InstRate_CV")
               + TREND_COLS},
            Notes=r["note"]))
        best_rows.append(dict(
            Filename=r["filename"], CueLabel=r["cue"],
            **{c: _r(c, b.get(c, np.nan)) for c in ("WindowStartSec", "WindowEndSec")},
            SyllableCount=b.get("SyllableCount"),
            **{c: _r(c, b.get(c, np.nan)) for c in ("DDK_Rate", "ISI_mean_ms", "ISI_CV", "InstRate_CV")},
            Notes=r["best_note"]))
        pierce_rows.append(dict(
            Filename=r["filename"], CueLabel=r["cue"],
            **{name: _pierce_val(c, pc.get(c, np.nan)) if pc else np.nan
               for c, name in pn.items()},
            Included=r["pierce_included"], Notes=r["pierce_note"]))
    out = []
    for rows in (whole_rows, best_rows):
        df = pd.DataFrame(rows)
        df["SyllableCount"] = df["SyllableCount"].astype("Int64")
        out.append(df)
    out.append(pd.DataFrame(pierce_rows))
    return out


def master_rows(results):
    rows = []
    for r in results:
        b, pc = r.get("best", {}), r.get("pierce", {})
        row = dict(RelativePath=r["rel_path"], Session=r["session_id"], Filename=r["filename"],
                   Subject=r["subject"], Date=r["date"], Day=r["day"], AudioNum=r["audio_num"],
                   Stim=r["stim"], **r["extra"], Trial=r["trial"], CueLabel=r["cue"],
                   SampleRate=r["fs"], FileDurationSec=mround(r["file_dur"], 2),
                   Whole_SyllableCount=r.get("SyllableCount"))
        for c in ("DDK_Rate", "ActiveDurationSec", "SpeechStartSec", "SpeechEndSec",
                  "ISI_mean_ms", "ISI_CV", "InstRate_CV") + TREND_COLS:
            row[f"Whole_{c}"] = _r(c, r.get(c, np.nan))
        for c in ("WindowStartSec", "WindowEndSec"):
            row[f"{WIN_LABEL}_{c}"] = _r(c, b.get(c, np.nan))
        row[f"{WIN_LABEL}_SyllableCount"] = b.get("SyllableCount")
        for c in ("DDK_Rate", "ISI_mean_ms", "ISI_CV", "InstRate_CV"):
            row[f"{WIN_LABEL}_{c}"] = _r(c, b.get(c, np.nan))
        row[f"{WIN_LABEL}_SegmentFile"] = r.get("segments", {}).get("best", "")
        for c, name in _pierce_names().items():
            v = pc.get(c, np.nan) if pc else np.nan
            row[f"{PIERCE_LABEL}_{name}"] = _pierce_val(c, v)
        row[f"{PIERCE_LABEL}_Included"] = r["pierce_included"]
        row[f"{PIERCE_LABEL}_SegmentFile"] = r.get("segments", {}).get("pierce", "")
        row["Notes"] = r["note"]
        row[f"{WIN_LABEL}_Notes"], row[f"{PIERCE_LABEL}_Notes"] = r["best_note"], r["pierce_note"]
        rows.append(row)
    return rows


def summary_rows(key, results):
    """Per session x cue means, AMR = mean of AMR cue means (as in 08_17_26 script)."""
    r0 = results[0]
    base = dict(RelativeFolder=r0["rel_dir"], Session=r0["session_id"], Subject=r0["subject"],
                Date=r0["date"], Day=r0["day"], AudioNum=r0["audio_num"], Stim=r0["stim"])
    vals = {}
    for cue in CUES:
        sel = [r for r in results if r["cue"].strip().lower() == cue and "best" in r]
        v = dict(n_trials=len(sel))
        for lbl, get in (("Whole_DDK_Rate", lambda r: r["DDK_Rate"]),
                         ("Whole_ISI_CV", lambda r: r["ISI_CV"]),
                         (f"{WIN_LABEL}_DDK_Rate", lambda r: r["best"]["DDK_Rate"]),
                         (f"{WIN_LABEL}_ISI_CV", lambda r: r["best"]["ISI_CV"]),
                         (f"{PIERCE_LABEL}_DDK_Rate", lambda r: r["pierce"]["DDK_Rate"]),
                         (f"{PIERCE_LABEL}_ISI_CV", lambda r: r["pierce"]["ISI_CV"])):
            x = np.array([get(r) for r in sel], dtype=float)
            v[lbl] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        vals[cue] = v
    amr = dict(n_trials=sum(vals[c]["n_trials"] for c in AMR_CUES))
    for lbl in [k for k in vals[CUES[0]] if k != "n_trials"]:
        x = np.array([vals[c][lbl] for c in AMR_CUES], dtype=float)
        amr[lbl] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    vals["AMR"] = amr
    rows = []
    for cue, v in vals.items():
        rows.append(dict(base, Cue=cue, n_trials=v["n_trials"],
                         **{k: mround(x, 3) for k, x in v.items() if k != "n_trials"}))
    return rows


# ── Statistics as in Pierce et al. (2013) ───────────────────────────────────
def _effect_code(col):
    """Sum-to-zero coding (levels-1 columns) -> Type III tests, as in SPSS."""
    levels = sorted(col.unique(), key=str)
    M = np.zeros((col.size, len(levels) - 1))
    for k, lev in enumerate(levels[:-1]):
        M[:, k] = (col == lev).astype(float) - (col == levels[-1]).astype(float)
    return M, levels


def _design(df, factors, interaction):
    blocks, notes = {}, []
    for f in factors:
        M, levels = _effect_code(df[f])
        if len(levels) < 2:
            notes.append(f"'{f}' has one level here; dropped")
            continue
        blocks[f] = M
    used = list(blocks)
    if interaction and len(used) >= 2:
        a, b = used[:2]
        blocks[f"{a}*{b}"] = np.einsum("ij,ik->ijk", blocks[a], blocks[b]).reshape(len(df), -1)
    cols, terms, j = [np.ones((len(df), 1))], {}, 1
    for name, M in blocks.items():
        cols.append(M)
        terms[name] = list(range(j, j + M.shape[1]))
        j += M.shape[1]
    X = np.hstack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        if interaction and len(used) >= 2:
            X, terms, more = _design(df, factors, False)
            return X, terms, notes + ["empty cells: interaction dropped"] + more
        raise ValueError(f"factors {used} are confounded (some level of one factor occurs with "
                         f"only one level of another), so their effects cannot be separated")
    return X, terms, notes


def anova_type3(y, X, terms):
    """Independent-groups ANOVA, Type III sums of squares (SPSS default).
    Returns per term: SS, df, F, p, partial eta^2 = SS_effect / (SS_effect + SS_error)."""
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sse = float(np.sum((y - X @ beta) ** 2))
    dfe = n - p
    xtx_inv = np.linalg.inv(X.T @ X)
    out = {}
    for term, idx in terms.items():
        b = beta[idx]
        ss = float(b @ np.linalg.solve(xtx_inv[np.ix_(idx, idx)], b))
        q = len(idx)
        F = (ss / q) / (sse / dfe)
        out[term] = dict(SS=ss, df1=q, df2=dfe, F=F, p=float(scipy.stats.f.sf(F, q, dfe)),
                         partial_eta_sq=ss / (ss + sse))
    return out


warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")


def spss_skew_kurt(x):
    """Skewness/kurtosis with standard errors as reported by SPSS (G1, G2)."""
    n = x.size
    r = dict(Skewness=np.nan, SE_Skew=np.nan, Kurtosis=np.nan, SE_Kurt=np.nan)
    if n >= 3 and np.ptp(x) == 0:                  # all values identical: undefined
        return r
    if n >= 3:
        ses = np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        r.update(Skewness=float(scipy.stats.skew(x, bias=False)), SE_Skew=float(ses))
        if n >= 4:
            sek = 2 * ses * np.sqrt((n * n - 1.0) / ((n - 3) * (n + 5)))
            r.update(Kurtosis=float(scipy.stats.kurtosis(x, fisher=True, bias=False)),
                     SE_Kurt=float(sek))
    return r


def iqr_outliers(x):
    """Boolean mask of values outside 1.5 x IQR of Tukey's hinges (SPSS boxplot rule)."""
    s = np.sort(x)
    h = (s.size + 1) // 2
    q1, q3 = np.median(s[:h]), np.median(s[s.size - h:])
    k = 1.5 * (q3 - q1)
    return (x < q1 - k) | (x > q3 + k)


def icc_2_1_3_1(Y):
    """Shrout & Fleiss (1979) ICC(2,1) and ICC(3,1) for an n-targets x k-raters matrix."""
    n, k = Y.shape
    gm = Y.mean()
    ssr = k * np.sum((Y.mean(axis=1) - gm) ** 2)
    ssc = n * np.sum((Y.mean(axis=0) - gm) ** 2)
    sse = np.sum((Y - gm) ** 2) - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return ((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
            (msr - mse) / (msr + (k - 1) * mse))


def _task_order():
    first = [c for c in ("papapa", "tatata", "kakaka") if c in CUES]   # paper: /pa/ /ta/ /ka/ /pataka/
    return first + [c for c in CUES if c not in first]


def _unit_values(master, metric):
    """One value per session x task: trials failing the paper's inclusion criteria are
    dropped, then TRIAL_AGG picks the first valid trial (single trial, as in the paper)."""
    df = master.copy()
    df["Task"] = df["CueLabel"].astype(str).str.strip().str.lower()
    df = df[df["Task"].isin(CUES) & df[metric].notna()]
    if metric.startswith(PIERCE_LABEL):
        df = df[df[f"{PIERCE_LABEL}_Included"].astype(bool)]
    df = df.sort_values(["Session", "Trial", "Filename"])
    g = df.groupby(["Session", "Task"], sort=False)
    if TRIAL_AGG == "first":
        out = g.first().reset_index()[["Session", "Task", metric, "RelativePath"]]
    else:
        out = g[metric].agg(TRIAL_AGG).reset_index()
    return out


def _session_factors(session_table):
    """Session-level factors for the stats, taken from the metadata."""
    groups = session_table.copy()
    for col in MEDIAN_SPLIT:                         # paper: age groups by median split
        if col not in groups.columns:
            _log(f"  MEDIAN_SPLIT column '{col}' not in the metadata — ignored.")
            continue
        x = pd.to_numeric(groups[col], errors="coerce")
        med = float(x.median())
        split = pd.Series(np.where(x <= med, f"<= {med:g}", f"> {med:g}"), index=groups.index,
                          dtype=object)
        groups[f"{col}_split"] = split.where(x.notna(), None)
        _log(f"  Median split of '{col}': median = {med:g} -> factor '{col}_split'")
    need = list(dict.fromkeys(list(GROUP_FACTORS) + [LEVENE_FACTOR]))
    missing = [c for c in need if c not in groups.columns]
    if missing:
        _log(f"  Factor column(s) {missing} not in the metadata (available: "
              f"{list(groups.columns)}) — statistics skipped.")
        return None
    if STATS_STIM_LEVELS:
        keep = {str(v).strip().upper() for v in STATS_STIM_LEVELS}
        groups = groups[groups["Stim"].isin(keep)]
    groups = groups[["Session"] + need].copy()
    for c in need:
        groups = groups[groups[c].notna() & (groups[c].astype(str).str.strip() != "")]
        groups[c] = groups[c].astype(str).str.strip()
    return groups


def _distribution_figure(metric, screened):
    """Frequency polygon + histogram and box plot per task (paper's normality screening)."""
    import matplotlib.pyplot as plt
    tasks = [t for t in _task_order() if t in screened]
    if not tasks:
        return None
    fig = plt.figure(figsize=(3.2 * len(tasks), 4.2))
    gs = fig.add_gridspec(2, len(tasks), height_ratios=[3, 1], hspace=0.08)
    for j, t in enumerate(tasks):
        x, out = screened[t]
        axh = fig.add_subplot(gs[0, j]); axb = fig.add_subplot(gs[1, j], sharex=axh)
        cnt, edges, _ = axh.hist(x, bins="auto", color=(0.75, 0.82, 0.92), edgecolor="w")
        mids = (edges[:-1] + edges[1:]) / 2
        axh.plot(np.r_[edges[0], mids, edges[-1]], np.r_[0, cnt, 0], "-o", color=(0.1, 0.3, 0.6), ms=3)
        axh.set_title(t, fontsize=10, style="italic")
        axh.tick_params(labelbottom=False)
        axb.boxplot(x, vert=False, whis=1.5, widths=0.6, showfliers=False)
        axb.plot(x[out], np.ones(out.sum()), "o", color="red", ms=5)
        axb.set_yticks([])
        axb.set_xlabel(metric, fontsize=8)
    fig.suptitle(f"Normality screening ({metric}); red = outside 1.5 x IQR", fontsize=10)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def run_group_stats(master, session_table):
    """Returns ({sheet name: DataFrame}, {metric: PNG bytes})."""
    groups = _session_factors(session_table)
    if groups is None or groups.empty:
        if groups is not None:
            _log(f"  No sessions with Stim in {STATS_STIM_LEVELS} — statistics skipped.")
        return {}, {}
    factors = list(GROUP_FACTORS)
    metrics = [m for m in STATS_METRICS if m in master.columns]
    if len(metrics) < len(STATS_METRICS):
        _log(f"  STATS_METRICS not found (skipped): {[m for m in STATS_METRICS if m not in metrics]}")
    norm, desc, lev, aov, figs = [], [], [], [], {}
    for metric in metrics:
        units = _unit_values(master, metric).merge(groups, on="Session", how="inner")
        _log(f"  {metric}: {units['Session'].nunique()} sessions "
              f"({', '.join(f'{k}={v}' for k, v in groups[factors[0]].value_counts().items())}), "
              f"{len(units)} session x task values (TRIAL_AGG={TRIAL_AGG}).")
        screened = {}
        for task in _task_order():
            d = units[units["Task"] == task].reset_index(drop=True)
            if d.empty:
                continue
            x = d[metric].to_numpy(float)
            # 1) normality screening per task over all units, 1.5 x IQR outliers
            out = iqr_outliers(x) if x.size >= 4 else np.zeros(x.size, bool)
            screened[task] = (x, out)
            stages = [("all units", x, out)]
            if EXCLUDE_IQR_OUTLIERS and out.any():
                d = d[~out].reset_index(drop=True)
                stages.append(("after outlier removal", d[metric].to_numpy(float), None))
            for stage, xs, o in stages:
                sk = spss_skew_kurt(xs)
                row = dict(Metric=metric, Task=task, Stage=stage, n=xs.size)
                for k in ("Skew", "Kurt"):
                    val, se = sk["Skewness" if k == "Skew" else "Kurtosis"], sk[f"SE_{k}"]
                    z = val / se if np.isfinite(se) else np.nan
                    row.update({("Skewness" if k == "Skew" else "Kurtosis"): mround(val, 3),
                                f"SE_{k}": mround(se, 3), f"z_{k}": mround(z, 2),
                                f"{k}_significant": bool(abs(z) > 1.96) if np.isfinite(z) else np.nan})
                if o is not None:
                    row["n_outliers"] = int(o.sum())
                    row["Outliers"] = "; ".join(f"{s}={v:g}" for s, v in
                                                zip(units[units['Task'] == task]["Session"][o], x[o]))
                norm.append(row)
            xs = d[metric].to_numpy(float)
            # 2) normative descriptives (Table 3): mean 1 dp, SD 2 dp, range 1 dp
            cells = list(d.groupby(factors)) + [((("All",) * len(factors)), d)]
            for keys, g in cells:
                keys = keys if isinstance(keys, tuple) else (keys,)
                v = g[metric].to_numpy(float)
                desc.append(dict(Metric=metric, Task=task, **dict(zip(factors, keys)), n=v.size,
                                 Mean=mround(v.mean(), 1),
                                 SD=mround(np.std(v, ddof=1), 2) if v.size > 1 else np.nan,
                                 Range=f"{mround(v.min(), 1):.1f}–{mround(v.max(), 1):.1f}"))
            # 3) Levene's F test for equality of variances (mean-centred, as SPSS)
            gl = [g[metric].to_numpy(float) for _, g in d.groupby(LEVENE_FACTOR)]
            row = dict(Metric=metric, Task=task, Factor=LEVENE_FACTOR, k_groups=len(gl), N=len(d))
            if len(gl) >= 2 and all(len(v) >= 3 for v in gl) and \
                    any(np.ptp(v) > 0 for v in gl):     # n = 2 per group makes Levene undefined
                F, pv = scipy.stats.levene(*gl, center="mean")
                row.update(F=mround(F, 3), df1=len(gl) - 1, df2=len(d) - len(gl), Significance=mround(pv, 3))
            else:
                row.update(Note="needs >= 2 groups with >= 3 sessions each and some spread")
            lev.append(row)
            # 4) independent-groups ANOVA (two-way when two factors), partial eta^2
            try:
                X, terms, notes = _design(d, factors, ANOVA_INTERACTION)
                if not terms:
                    raise ValueError("no factor with >= 2 levels")
                if len(d) <= X.shape[1]:
                    raise ValueError(f"n={len(d)} too small for {X.shape[1]} parameters")
                for term, r in anova_type3(xs, X, terms).items():
                    aov.append(dict(Metric=metric, Task=task, Term=term, F=mround(r["F"], 3),
                                    df1=r["df1"], df2=r["df2"], p=mround(r["p"], 3),
                                    partial_eta_sq=mround(r["partial_eta_sq"], 3),
                                    Report=f"F({r['df1']},{r['df2']}) = {r['F']:.3f}, p = {r['p']:.3f}, "
                                           f"partial eta2 = {100 * r['partial_eta_sq']:.1f}%",
                                    N=len(d), Note="; ".join(notes)))
            except Exception as e:
                aov.append(dict(Metric=metric, Task=task, Term="", N=len(d), Note=str(e)))
        figs[metric] = _distribution_figure(metric, screened)

    sheets = {}
    # Table 3 layout: rows = first factor x task, columns = second factor x (n, Mean, SD, Range)
    dd = pd.DataFrame(desc)
    if not dd.empty:
        dd = dd[~(dd[factors] == "All").all(axis=1)]
        rows, cols = (["Metric", factors[0], "Task"], factors[1]) if len(factors) >= 2 \
            else (["Metric", "Task"], factors[0])
        order = {t: i for i, t in enumerate(_task_order())}
        t3 = dd.pivot_table(index=rows, columns=cols, values=["n", "Mean", "SD", "Range"],
                            aggfunc="first")
        t3 = t3.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        t3 = t3.reindex(columns=[(c, s) for c in t3.columns.levels[0]
                                 for s in ("n", "Mean", "SD", "Range") if (c, s) in t3.columns])
        t3 = t3.reset_index()
        t3["_o"] = t3["Task"].map(order)
        t3 = t3.sort_values(rows[:-1] + ["_o"]).drop(columns="_o", level=0)
        t3.columns = [" ".join(str(x) for x in c if str(x)).strip() for c in t3.columns]
        for c in [c for c in t3.columns if c.endswith(" n")]:
            t3[c] = t3[c].astype("Int64")
        sheets["Table3"] = t3
    sheets.update(ANOVA=pd.DataFrame(aov), Levene=pd.DataFrame(lev),
                  Normality=pd.DataFrame(norm), Descriptives=pd.DataFrame(desc))
    return sheets, {k: v for k, v in figs.items() if v}


def _filled(path, cols):
    """True if a template at `path` already contains user entries in `cols` (never overwrite)."""
    if not os.path.isfile(path):
        return False
    try:
        t = pd.read_csv(path, dtype=str)
    except Exception:
        return True
    return any(c in t.columns and t[c].fillna("").str.strip().ne("").any() for c in cols)


def write_reliability_sample(master, out_root):
    """Paper: 10 % of participants drawn at random, their files re-measured by hand.
    Here: RELIABILITY_FRACTION of sessions; every included trial of those sessions is listed
    and exported as 16-bit WAV so it can be measured in Praat (blind to automatic values)."""
    rel_dir = os.path.join(out_root, "reliability")
    os.makedirs(rel_dir, exist_ok=True)
    path = os.path.join(rel_dir, "DDK_reliability_sample.csv")
    if _filled(path, ["ManualDurationSec"]):
        _log("  DDK_reliability_sample.csv already filled in — kept as is.")
        return
    inc = master[master[f"{PIERCE_LABEL}_Included"].astype(bool)]
    sessions = sorted(inc["Session"].unique())
    if not sessions:
        return
    n = max(1, int(round(RELIABILITY_FRACTION * len(sessions))))
    pick = set(np.random.default_rng(RELIABILITY_SEED).choice(sessions, n, replace=False))
    sample = inc[inc["Session"].isin(pick)].sort_values(["Session", "Trial"])
    wav_dir = os.path.join(rel_dir, "audio")
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    for _, r in sample.iterrows():
        wav = f"{r['Session']}__{os.path.splitext(r['Filename'])[0]}.wav"
        try:
            audio, fs, _ = load_audio(os.path.join(os.path.abspath(INPUT), r["RelativePath"]))
            pk = np.abs(audio).max() or 1.0
            wavfile.write(os.path.join(wav_dir, wav), int(round(fs)),
                          np.int16(np.round(audio / pk * 0.9 * 32767)))
        except Exception as e:
            wav = f"export failed: {e}"
        rows.append(dict(Session=r["Session"], RelativePath=r["RelativePath"], Filename=r["Filename"],
                         CueLabel=r["CueLabel"], WavFile=wav, ManualDurationSec=""))
    pd.DataFrame(rows).to_csv(path, index=False)
    _log(f"  Reliability sample: {len(rows)} trials from {n}/{len(sessions)} sessions "
          f"-> reliability/DDK_reliability_sample.csv + reliability/audio/")


def run_reliability(master):
    """Returns {sheet name: DataFrame} with the ICCs and the per-trial comparison."""
    path = os.path.abspath(RELIABILITY_FILE)
    if not os.path.isfile(path):
        _log(f"  RELIABILITY_FILE not found: {path}")
        return {}
    man = pd.read_csv(path)
    man["ManualDurationSec"] = pd.to_numeric(man["ManualDurationSec"], errors="coerce")
    man = man[man["ManualDurationSec"] > 0]
    man["ManualDurationSec"] = man["ManualDurationSec"].map(lambda v: mround(v, 4))
    man["Manual_DDK_Rate"] = man["ManualDurationSec"].map(lambda v: mround(PIERCE_N_SYLL / v, 1))
    a = master[["RelativePath", f"{PIERCE_LABEL}_DurationSec", f"{PIERCE_LABEL}_DDK_Rate"]]
    cmp_ = man.merge(a, on="RelativePath", how="inner").dropna(
        subset=[f"{PIERCE_LABEL}_DurationSec", f"{PIERCE_LABEL}_DDK_Rate"])
    if len(cmp_) < 3:
        _log(f"  Reliability: only {len(cmp_)} measured trials matched — need >= 3.")
        return {}
    cmp_["Diff_ms"] = (cmp_[f"{PIERCE_LABEL}_DurationSec"] - cmp_["ManualDurationSec"]) * 1000
    rows = []
    for label, auto, manual in (("DDK_Rate (1 dp)", f"{PIERCE_LABEL}_DDK_Rate", "Manual_DDK_Rate"),
                                ("DurationSec (4 dp)", f"{PIERCE_LABEL}_DurationSec", "ManualDurationSec")):
        i21, i31 = icc_2_1_3_1(cmp_[[auto, manual]].to_numpy(float))
        rows.append(dict(Measure=label, n=len(cmp_), ICC_2_1=mround(i21, 3), ICC_3_1=mround(i31, 3)))
    rows[1].update(MeanDiff_ms=mround(cmp_["Diff_ms"].mean(), 2),
                   MeanAbsDiff_ms=mround(cmp_["Diff_ms"].abs().mean(), 2),
                   MaxAbsDiff_ms=mround(cmp_["Diff_ms"].abs().max(), 2))
    _log(f"  Reliability (automatic vs manual, n={len(cmp_)}): rate ICC(2,1) = {rows[0]['ICC_2_1']}, "
          f"ICC(3,1) = {rows[0]['ICC_3_1']}")
    return {"Reliability_ICC": pd.DataFrame(rows), "Reliability_trials": cmp_}


# ── Output files ────────────────────────────────────────────────────────────
CHECK_TYPES = {
    "missing":         "listed in the metadata, but no audio file with that name under INPUT",
    "load_error":      "audio file found but could not be read or analysed",
    "unlisted":        "audio file under INPUT that is not in the metadata (not analysed)",
    "folder_mismatch": "file sits in a day folder that does not match its SessionDate",
    "duplicate":       "same file name found more than once (copy in the matching day folder used)",
    "metadata":        "problem inside the sheet (unreadable date/audio, duplicate row, value "
                       "varying within a session, filename disagreeing with its columns)",
}


def _print_setup(ws):
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def _style(ws, freeze="A2"):
    """Arial throughout, bold shaded header, frozen header, filter, fitted column widths."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    base, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    fill = PatternFill("solid", fgColor="DDE6F0")
    widths = {}
    for row in ws.iter_rows():
        for c in row:
            c.font = bold if c.row == 1 else base
            if c.value is not None:
                widths[c.column] = max(widths.get(c.column, 0), len(str(c.value)))
    for c in ws[1]:
        c.fill = fill
        c.alignment = Alignment(vertical="center")
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = min(max(8, w + 2), 60)
    if ws.max_row > 1:
        ws.freeze_panes = freeze
        ws.auto_filter.ref = ws.dimensions
    _print_setup(ws)


def _write_info(ws, info):
    """info: list of (kind, label, value) with kind in title / section / row."""
    from openpyxl.styles import Alignment, Font
    r = 1
    for kind, label, value in info:
        if kind == "section":
            r += 1
        c = ws.cell(row=r, column=1, value=label)
        c.font = Font(name="Arial", size=13 if kind == "title" else 10,
                      bold=kind in ("title", "section"))
        if kind == "row":
            v = ws.cell(row=r, column=2, value=value if isinstance(value, (int, float, str))
                        else str(value))
            v.font = Font(name="Arial", size=10)
            v.alignment = Alignment(horizontal="left", wrap_text=True, vertical="top")
        r += 1
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 110
    _print_setup(ws)


def write_results_workbook(path, info, sheets, images):
    """One workbook: Info first, then one sheet per table; PNGs placed beside their table."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame().to_excel(xw, sheet_name="Info")
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        wb = xw.book
        _write_info(wb["Info"], info)
        for name in sheets:
            _style(wb[name[:31]], freeze="D2" if name == "Trials" else "A2")
        try:
            from openpyxl.drawing.image import Image as XLImage
            from openpyxl.utils import get_column_letter
            for name, pngs in images.items():
                ws, row = wb[name], 2
                col = get_column_letter(ws.max_column + 2)
                for png in pngs:
                    img = XLImage(io.BytesIO(png))
                    img.width, img.height = 900, int(img.height * 900 / img.width)
                    ws.add_image(img, f"{col}{row}")
                    row += img.height // 20 + 2
        except ImportError:
            _log("  (Pillow not installed: figures not embedded in the workbook)")


CHECK_LABELS = {"missing": "Missing audio", "load_error": "Unreadable audio",
                "unlisted": "Unlisted audio", "folder_mismatch": "Wrong day folder",
                "duplicate": "Duplicate files", "metadata": "Metadata problems"}
MISSING_COLS = ["MetadataRow", "SessionDate", "Audio", "Stim", "CueLabel", "File",
                "ExpectedFolder", "FolderFound", "AudioFilesInFolder", "SimilarFileFound", "Detail"]


def write_file_check(path, issues, n_listed, subject, meta_path, root, n_analysed):
    """Sheet 'File check': counts, then the full list of every problem, grouped by type
    (missing audio first). Sheet 'All issues': the same entries as one filterable table."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    df = pd.DataFrame(issues).rename(columns={"Row": "MetadataRow"})
    for c in ["Type", "File", "MetadataRow", "Detail"]:
        if c not in df.columns:
            df[c] = pd.Series(dtype=object)
    df["MetadataRow"] = pd.to_numeric(df["MetadataRow"], errors="coerce").astype("Int64")
    order = {t: i for i, t in enumerate(CHECK_TYPES)}
    df = df.assign(_o=df["Type"].map(order).fillna(99)).sort_values(["_o", "MetadataRow", "File"]) \
           .drop(columns="_o").reset_index(drop=True)
    counts = df["Type"].value_counts()

    F = lambda **k: Font(name="Arial", size=k.pop("size", 10), **k)
    fill = PatternFill("solid", fgColor="DDE6F0")
    warn = PatternFill("solid", fgColor="FCE4D6")
    wb = Workbook()
    ws = wb.active
    ws.title = "File check"
    widths = {}

    def put(r, c, v, fit=True, **font):
        """fit=False for long free text in a row's last column: it overflows to the right
        instead of widening a column that holds short values in the other tables."""
        cell = ws.cell(row=r, column=c, value=None if (v is None or (isinstance(v, float) and np.isnan(v))
                                                       or v is pd.NA) else v)
        cell.font = F(**font)
        cell.alignment = Alignment(horizontal="left", vertical="top")
        if cell.value is not None and r > 3 and fit:
            widths[c] = max(widths.get(c, 0), len(str(cell.value)))
        return cell

    put(1, 1, f"File check — subject {subject}", bold=True, size=13)
    put(2, 1, f"Metadata: {os.path.basename(meta_path)}   |   INPUT: {root}   |   "
              f"{n_listed} trials listed, {n_analysed} analysed")
    r = 4
    for j, h in enumerate(["Problem", "Count", "Meaning"], 1):
        put(r, j, h, bold=True).fill = fill
    for t, meaning in CHECK_TYPES.items():
        r += 1
        n = int(counts.get(t, 0))
        put(r, 1, CHECK_LABELS[t], bold=n > 0)
        put(r, 2, n, bold=n > 0)
        put(r, 3, meaning, fit=False)
        if n:
            for j in (1, 2, 3):
                ws.cell(row=r, column=j).fill = warn
    r += 1
    if df.empty:
        put(r + 1, 1, f"No problems: all {n_listed} listed files were found and analysed, and no "
                      f"audio file under INPUT is missing from the metadata.", bold=True)
    for t in CHECK_TYPES:
        block = df[df["Type"] == t]
        if block.empty:
            continue
        r += 2
        put(r, 1, f"{CHECK_LABELS[t]} — {len(block)}: {CHECK_TYPES[t]}", fit=False, bold=True, size=11)
        cols = [c for c in MISSING_COLS if c in block.columns] if t == "missing" \
            else ["MetadataRow", "File", "Detail"]
        r += 1
        for j, c in enumerate(cols, 1):
            put(r, j, c, bold=True).fill = fill
        for _, row in block.iterrows():
            r += 1
            for j, c in enumerate(cols, 1):
                put(r, j, row[c], fit=c != "Detail")
    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = min(max(10, w + 2), 70)
    _print_setup(ws)

    ws2 = wb.create_sheet("All issues")
    flat = df[["Type", "MetadataRow", "File", "Detail"] +
              [c for c in MISSING_COLS if c in df.columns and c not in ("MetadataRow", "File", "Detail")]]
    for j, c in enumerate(flat.columns, 1):
        ws2.cell(row=1, column=j, value=c)
    for i, row in enumerate(flat.itertuples(index=False), 2):
        for j, v in enumerate(row, 1):
            ws2.cell(row=i, column=j, value=None if (v is pd.NA or (isinstance(v, float) and np.isnan(v))) else v)
    _style(ws2)
    wb.save(path)


def write_plots_pdf(path, results):
    """All diagnostic plots in ONE multi-page PDF (page order = session, trial)."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for r in results:
            png = r.pop("plot_png", None)
            if not png:
                continue
            img = mpimg.imread(io.BytesIO(png), format="png")
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 110, h / 110), dpi=110)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=110)
            plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def _run(cfg, should_stop):
    """The original main(), reading the constants set by configure()."""
    global OUTPUT
    if not INPUT:
        raise DDKError("no input folder selected: choose the folder that holds the day "
                       "folders and the metadata sheet")
    root = os.path.abspath(INPUT)
    if not os.path.isdir(root):
        raise DDKError(f"INPUT folder not found: {root}")
    OUTPUT = OUTPUT or os.path.join(root, "DDK_results")
    out_root = os.path.abspath(OUTPUT)
    if BEST_WIN_CRITERION not in ("max_rate", "min_isi_cv", "first"):
        raise DDKError(f"Unknown BEST_WIN_CRITERION: {BEST_WIN_CRITERION}")
    if PIERCE_VAR_TIMES not in ("onsets", "peaks"):
        raise DDKError(f"Unknown PIERCE_VAR_TIMES: {PIERCE_VAR_TIMES} (use 'onsets' or 'peaks')")
    os.makedirs(out_root, exist_ok=True)
    t_start = time.time()

    _log("=" * 70)
    _log(f"INPUT : {root}\nOUTPUT: {out_root}\nWindow: {WIN_LABEL} ({BEST_WIN_CRITERION})")
    _log("=" * 70)
    issues = []
    meta_path = find_metadata_file(root)
    subject = subject_from(meta_path)
    pkey, params = resolve_params(subject)
    meta, extra = load_metadata(meta_path, issues)
    _log(f"Metadata: {os.path.basename(meta_path)} — {len(meta)} trials listed | subject {subject} | "
          f"peak params {params} ({'from ' + pkey if pkey else 'DEFAULT_PARAMS'})")
    audio_index = index_audio(root, out_root)
    sessions = build_sessions(meta, extra, audio_index, root, issues)
    diagnose_missing(issues, root, audio_index, set(meta["stem"].str.lower()))
    missing = [i for i in issues if i["Type"] == "missing"]
    if missing:
        _log(f"\nMISSING AUDIO: {len(missing)} of {len(meta)} listed trials not found under INPUT:")
        for it in missing[:15]:
            _log(f"  row {it['Row']:>4}  {it['File']:<38} {it['Detail']}")
        if len(missing) > 15:
            _log(f"  ... {len(missing) - 15} more — full list in DDK_file_check_{subject}.xlsx")
    if not sessions:
        path = os.path.join(out_root, f"DDK_file_check_{subject}.xlsx")
        write_file_check(path, issues, len(meta), subject, meta_path, root, 0)
        raise DDKError(f"None of the files listed in the metadata were found under INPUT ({root}). "
                 f"Top-level folders there: {sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))}. "
                 f"Details: {path}")

    jobs, sess_info, sess_rows = [], {}, []
    for key in sorted(sessions):
        date, audio = key
        files = sessions[key]
        session_id = f"{subject}_{date}_audio{audio}"
        rel_dir = max(set(f["rel_dir"] for f in files), key=[f["rel_dir"] for f in files].count)
        out_dir = os.path.join(out_root, rel_dir) if rel_dir else out_root
        stim, day = files[0]["stim"], files[0]["day"]
        sess_info[key] = dict(session_id=session_id, out_dir=out_dir)
        sess_rows.append(dict(Session=session_id, Subject=subject, Date=date, Day=day, Audio=audio,
                              Stim=stim, RelativeFolder=rel_dir, **files[0]["extra"]))
        cues = {}
        for f in files:
            cues[f["cue"]] = cues.get(f["cue"], 0) + 1
        _log(f"[{session_id}] {stim:<4} day {day:<5} {len(files):2d} trials  "
              + " ".join(f"{c}x{n}" for c, n in cues.items()))
        for i, f in enumerate(files):
            jobs.append(dict(key=key, order=i, path=f["path"], filename=f["filename"],
                             stem=os.path.splitext(f["filename"])[0],
                             rel_path=os.path.relpath(f["path"], root), rel_dir=rel_dir,
                             trial=f["trial"], date=date, audio_num=str(audio), subject=subject,
                             params=params, cue=f["cue"], stim=stim, day=day, extra=f["extra"],
                             meta_row=f["meta_row"],
                             session_id=session_id, out_dir=out_dir, out_root=out_root))
    session_table = pd.DataFrame(sess_rows)
    if issues:
        counts = pd.Series([i["Type"] for i in issues]).value_counts().to_dict()
        _log(f"\nFile check: {counts}")

    workers = N_WORKERS if N_WORKERS > 0 else max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(jobs))
    _log(f"\nAnalysing {len(jobs)} trials in {len(sessions)} sessions with {workers} worker(s)...\n")

    results = []

    def report(r, k):
        b = r.get("best")
        if b:
            _log(f"  [{k:4d}/{len(jobs)}] {r['rel_path']:<50.50s} cue={r['cue']:<8s} "
                  f"n={r['SyllableCount']:3d} rate={r['DDK_Rate']:.2f} ISI_CV={r['ISI_CV']:.3f} | "
                  f"{WIN_LABEL}: n={b['SyllableCount']:3d} rate={b['DDK_Rate']:.2f} ISI_CV={b['ISI_CV']:.3f} | "
                  f"{PIERCE_LABEL}: rate={r['pierce']['DDK_Rate']:.1f} ISI_CV={r['pierce']['ISI_CV']:.3f}")
        else:
            _log(f"  [{k:4d}/{len(jobs)}] {r['rel_path']}  {r['note']}")

    # Stop requests are honoured between trials: the trials already running finish,
    # nothing is written, so no half-filled workbook is left behind.
    if workers == 1:
        for k, job in enumerate(jobs, 1):
            if should_stop():
                break
            results.append(analyze_trial(job))
            report(results[-1], k)
    else:
        # Spawned, not forked, on every platform: the interface runs this on a worker
        # thread, and forking a multi-threaded process can deadlock. Spawned workers
        # start from the defaults, so each one is configured with the same settings.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=configure, initargs=(cfg,)) as ex:
            futs = [ex.submit(analyze_trial, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                report(results[-1], k)
                if should_stop():
                    for f in futs:
                        f.cancel()
                    break
    if should_stop():
        _log(f"[stopped] after {len(results)} of {len(jobs)} trials; no workbook written")
        return None

    results.sort(key=lambda r: (r["key"], r["order"]))
    by_sess = {}
    for r in results:
        by_sess.setdefault(r["key"], []).append(r)
    all_rows, summ_rows = [], []
    for key, rs in by_sess.items():
        if SAVE_MATLAB_FILES:                      # optional: per-session files for MATLAB
            info = sess_info[key]
            os.makedirs(info["out_dir"], exist_ok=True)
            base = os.path.join(info["out_dir"], f"DDK_syllable_counts_{info['session_id']}")
            for df, suffix in zip(session_tables(rs), ("", f"_{WIN_LABEL.lower()}",
                                                       f"_{PIERCE_LABEL.lower()}")):
                df.to_csv(base + suffix + ".csv", index=False, na_rep="NaN")
        all_rows += master_rows(rs)
        summ_rows += summary_rows(key, rs)
    master = pd.DataFrame(all_rows)
    for c in ("Whole_SyllableCount", f"{WIN_LABEL}_SyllableCount", "Trial"):
        master[c] = master[c].astype("Int64")

    errors = [r for r in results if "traceback" in r]
    for r in errors:
        issues.append(dict(Type="load_error", File=r["rel_path"], Row=r["meta_row"],
                           Detail=r["note"].replace("ERROR: ", "")))

    _log(f"\n{PIERCE_LABEL} analysis (Pierce, Cotton & Perry 2013):")
    sheets = {"Trials": master, "Sessions": pd.DataFrame(summ_rows)}
    images = {}
    if RUN_STATS:
        stat_sheets, figs = run_group_stats(master, session_table)
        sheets.update(stat_sheets)
        if figs and "Normality" in sheets:
            images["Normality"] = list(figs.values())
    if RELIABILITY_EXPORT:
        write_reliability_sample(master, out_root)
    if RELIABILITY_FILE:
        sheets.update(run_reliability(master))

    # ── Info sheet: run summary, sheet guide, column legend, every setting ──
    n_issues = sum(1 for i in issues)
    check_name = f"DDK_file_check_{subject}.xlsx"
    guide = {
        "Trials": "one row per analysed audio trial: identifiers, Stim, cue, and the three "
                  "analyses side by side (see column prefixes)",
        "Sessions": "per session x cue means of the trial values; AMR = mean of the "
                    f"{'/'.join(AMR_CUES)} means",
        "Table3": "Pierce et al. (2013) Table 3 layout: n, mean (1 dp), SD (2 dp), range per "
                  "task and group",
        "ANOVA": "independent-groups ANOVA per task (Type III SS): F, p, partial eta^2",
        "Levene": f"Levene's F test for equal variances between {LEVENE_FACTOR} groups "
                  "(paper's Table 4)",
        "Normality": "skewness / kurtosis with SPSS standard errors and 1.5 x IQR outliers "
                     "(removed before the tests); distribution figure on the right",
        "Descriptives": "long-format descriptives per task and group, incl. totals ('All')",
        "Reliability_ICC": "automatic vs manual measurement: ICC(2,1) and ICC(3,1)",
        "Reliability_trials": "per-trial automatic vs manual durations and rates",
    }
    info = [("title", f"DDK analysis — subject {subject}", None),
            ("section", "Run", None),
            ("row", "Finished", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("row", "INPUT", root), ("row", "Metadata file", os.path.basename(meta_path)),
            ("row", "Trials listed in metadata", len(meta)),
            ("row", "Trials analysed", len(results) - len(errors)),
            ("row", "Sessions", len(by_sess)),
            ("row", "File-check entries", f"{n_issues} — see {check_name}"),
            ("row", "Peak parameters", f"{params} ({'PEAK_PARAMS ' + pkey if pkey else 'DEFAULT_PARAMS'})"),
            ("row", "Statistics", f"unit = session x task; Stim levels {STATS_STIM_LEVELS or 'all'}; "
                                  f"factors {GROUP_FACTORS}; trials combined: {TRIAL_AGG}"),
            ("section", "Sheets", None)]
    info += [("row", k, v) for k, v in guide.items() if k in sheets]
    info += [("section", "Column prefixes (Trials sheet)", None),
             ("row", "Whole_", "whole trial, method of DDK_processing_06_03_26.m (envelope peaks; "
                               "rate = syllables / active-speech duration)"),
             ("row", "Whole_InstRate_Slope", "change of the local (3-syllable) rate over the trial, "
                                             "syll/s per s; > 0 = accelerating (festination)"),
             ("row", "Whole_PeakAmp_CV / _Trend_pct_s", "syllable peak amplitude: CV, and change "
                                                       "per second as % of the mean (< 0 = fading); "
                                                       "relative only, the envelope is normalised per trial"),
             ("row", f"{WIN_LABEL}_", f"{BEST_WIN_SEC:g}-s window chosen by '{BEST_WIN_CRITERION}' "
                                      "(rate = syllables in window / window length)"),
             ("row", f"{PIERCE_LABEL}_", f"Pierce, Cotton & Perry (2013): onset of glottal pulsing of "
                                         f"syllable 1 to syllable {PIERCE_N_SYLL + 1}; rate = "
                                         f"{PIERCE_N_SYLL} / duration; ISI mean / ISI CV / "
                                         f"InstRate CV from the same syllables 1-{PIERCE_N_SYLL + 1} "
                                         f"({'voicing onsets' if PIERCE_VAR_TIMES == 'onsets' else 'envelope peaks'})"),
             ("row", f"{PIERCE_LABEL}_Included", f"meets the paper's criteria (>= {PIERCE_N_SYLL + 1} "
                                                 f"syllables, >= {PIERCE_MIN_SPEECH_SEC:g} s of repetition)"),
             ("row", "..._SegmentFile", f"WAV of the selected portion in OUTPUT/'{BEST_SEG_DIR}' "
                                      f"and OUTPUT/'{PIERCE_SEG_DIR}' (empty = not saved)"),
             ("row", "Notes columns", "processing notes; empty = no remarks"),
             ("section", "Settings", None)]
    info += [("row", k, v) for k, v in globals().items()
             if k.isupper() and not k.startswith("_") and k not in ("CHECK_TYPES", "STD_META",
                                                                    "SKIP_DIR_NAMES", "DIGITS")
             and isinstance(v, (str, int, float, tuple, list, dict, bool))]
    info += [("section", "Software", None),
             ("row", "Versions", f"Python {sys.version.split()[0]}, numpy {np.__version__}, scipy "
                                 f"{scipy.__version__}, pandas {pd.__version__}")]

    res_path = os.path.join(out_root, f"DDK_results_{subject}.xlsx")
    write_results_workbook(res_path, info, sheets, images)
    write_file_check(os.path.join(out_root, check_name), issues, len(meta), subject, meta_path,
                     root, len(results) - len(errors))
    written = [os.path.basename(res_path), check_name]
    if SAVE_PLOTS:
        pdf_path = os.path.join(out_root, f"DDK_plots_{subject}.pdf")
        write_plots_pdf(pdf_path, results)
        written.append(os.path.basename(pdf_path))
    if SAVE_SEGMENTS:
        written += [BEST_SEG_DIR + "/", PIERCE_SEG_DIR + "/"]

    _log(f"\n[done] {len(results)} trials, {len(by_sess)} sessions, {len(errors)} error(s), "
          f"{time.time() - t_start:.1f} s.\nWritten to {out_root}: " + ", ".join(written))
    for r in errors:
        _log(f"  ERROR {r['rel_path']}: {r['note']}")
    outputs = [res_path, os.path.join(out_root, check_name)]
    if SAVE_PLOTS:
        outputs.append(pdf_path)
    return dict(subject=subject, folder=out_root, outputs=outputs,
                trials=len(results), sessions=len(by_sess), errors=len(errors),
                issues=n_issues,
                folders=[os.path.join(out_root, d) for d in written if d.endswith("/")])


# ════════════════════════════════════════════════════════════════════════════
#  SPEECHKIT SETTINGS -> CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
def _items(text, lower=False, upper=False):
    """'pataka, tatata' -> ('pataka', 'tatata'). Lists pass through."""
    parts = text if isinstance(text, (list, tuple)) else re.split(r"[,;\n]", str(text or ""))
    out = [str(p).strip() for p in parts if str(p).strip()]
    if lower:
        out = [p.lower() for p in out]
    if upper:
        out = [p.upper() for p in out]
    return tuple(out)


def _num(x):
    """100.0 -> 100, so the Info sheet shows the values as the script wrote them."""
    x = float(x)
    return int(x) if x.is_integer() else x


def _peak_params(raw):
    """{"S04": [0.12, 0.10, 0.15, 0.10]} -> {"S04": (0.12, 0.1, 0.15, 0.1)}."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise DDKError('Per-subject peak parameters must be a JSON object such as '
                       '{"S04": [0.12, 0.10, 0.15, 0.10]}')
    out = {}
    for key, value in raw.items():
        try:
            vals = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            vals = ()
        if len(vals) != 4:
            raise DDKError(f"Per-subject peak parameters for {key!r} need four numbers: "
                           f"[minPeakDistance s, minPeakHeight, minPeakProminence, "
                           f"smoothWindow s], got {value!r}")
        out[str(key).strip()] = vals
    return out


def configure(cfg=None):
    """
    Copy the DDK_* settings onto the module constants the analysis reads.

    Called before every run in the parent process, and as the initializer of every
    worker process, so spawned workers measure with the same values. Settings that
    are missing from `cfg` take their speechkit default.
    """
    from . import settings as _settings

    values = _settings.defaults()
    for name, raw in (cfg or {}).items():
        if name.startswith("DDK_") and name in _settings.BY_NAME:
            values[name] = _settings.coerce(name, raw)
    v = {k[4:]: values[k] for k in values if k.startswith("DDK_")}
    g = globals()

    g["INPUT"] = str(v["INPUT"]).strip()
    g["OUTPUT"] = str(v["OUTPUT"]).strip()
    g["METADATA_FILE"] = str(v["METADATA_FILE"]).strip()
    sheet = str(v["METADATA_SHEET"]).strip()
    g["METADATA_SHEET"] = int(sheet) if sheet.isdigit() else (sheet or 0)
    g["META_COLUMNS"] = dict(date=v["COL_DATE"], audio=v["COL_AUDIO"], stim=v["COL_STIM"],
                             filename=v["COL_FILENAME"], cue=v["COL_CUE"])
    g["SUBJECT_ID"] = str(v["SUBJECT_ID"]).strip()

    g["PEAK_PARAMS"] = _peak_params(v["PEAK_PARAMS"])
    g["DEFAULT_PARAMS"] = (float(v["MIN_PEAK_DISTANCE_S"]), float(v["MIN_PEAK_HEIGHT"]),
                           float(v["MIN_PEAK_PROMINENCE"]), float(v["SMOOTH_WINDOW_S"]))
    g["TRIPLET_WIN"] = int(v["TRIPLET_WIN"])
    g["ENERGY_THRESH"] = float(v["ENERGY_THRESH"])
    g["BANDPASS_LOW_HZ"] = _num(v["BANDPASS_LOW_HZ"])
    g["BANDPASS_HIGH_HZ"] = _num(v["BANDPASS_HIGH_HZ"])
    g["FILTER_ORDER"] = int(v["FILTER_ORDER"])

    g["BEST_WIN_SEC"] = float(v["BEST_WIN_SEC"])
    g["BEST_WIN_CRITERION"] = v["BEST_WIN_CRITERION"]
    g["BEST_WIN_STEP_SEC"] = float(v["BEST_WIN_STEP_SEC"])

    g["PIERCE_N_SYLL"] = int(v["PIERCE_N_SYLL"])
    g["PIERCE_MIN_SPEECH_SEC"] = float(v["PIERCE_MIN_SPEECH_SEC"])
    g["VOICING_BAND_HZ"] = (_num(v["VOICING_LOW_HZ"]), _num(v["VOICING_HIGH_HZ"]))
    g["VOICING_SMOOTH_SEC"] = float(v["VOICING_SMOOTH_SEC"])
    g["VOICING_ONSET_FRAC"] = float(v["VOICING_ONSET_FRAC"])
    g["PIERCE_VAR_TIMES"] = v["PIERCE_VAR_TIMES"]

    g["RUN_STATS"] = bool(v["RUN_STATS"])
    g["GROUP_FACTORS"] = _items(v["GROUP_FACTORS"])
    g["LEVENE_FACTOR"] = str(v["LEVENE_FACTOR"]).strip()
    g["STATS_STIM_LEVELS"] = _items(v["STATS_STIM_LEVELS"], upper=True)
    g["MEDIAN_SPLIT"] = _items(v["MEDIAN_SPLIT"])
    g["ANOVA_INTERACTION"] = bool(v["ANOVA_INTERACTION"])
    g["EXCLUDE_IQR_OUTLIERS"] = bool(v["EXCLUDE_IQR_OUTLIERS"])
    g["TRIAL_AGG"] = v["TRIAL_AGG"]
    g["STATS_METRICS"] = _items(v["STATS_METRICS"])

    g["RELIABILITY_EXPORT"] = bool(v["RELIABILITY_EXPORT"])
    g["RELIABILITY_FRACTION"] = float(v["RELIABILITY_FRACTION"])
    g["RELIABILITY_SEED"] = int(v["RELIABILITY_SEED"])
    g["RELIABILITY_FILE"] = str(v["RELIABILITY_FILE"]).strip()

    g["CUES"] = _items(v["CUES"], lower=True)
    g["AMR_CUES"] = _items(v["AMR_CUES"], lower=True)

    g["AUDIO_EXTENSIONS"] = tuple(e if e.startswith(".") else "." + e
                                  for e in _items(v["AUDIO_EXTENSIONS"], lower=True))
    g["MAT_AUDIO_VAR"] = str(v["MAT_AUDIO_VAR"]).strip()
    g["MAT_FS_VAR"] = str(v["MAT_FS_VAR"]).strip()

    g["SAVE_PLOTS"] = bool(v["SAVE_PLOTS"])
    g["SAVE_MATLAB_FILES"] = bool(v["SAVE_MATLAB_FILES"])
    g["N_WORKERS"] = int(v["N_WORKERS"])
    g["SAVE_SEGMENTS"] = bool(v["SAVE_SEGMENTS"])
    g["SEGMENT_PAD_SEC"] = float(v["SEGMENT_PAD_SEC"])

    _check()
    _derive_labels()
    bandpass_sos.cache_clear()     # both filters depend on the settings above
    voicing_sos.cache_clear()


def _check():
    """Combinations a range check on one field cannot catch."""
    problems = []
    if BANDPASS_LOW_HZ >= BANDPASS_HIGH_HZ:
        problems.append("band-pass low edge must be below the high edge")
    if FILTER_ORDER % 2:
        problems.append("band-pass filter order must be even (it is split into two "
                        "Butterworth halves, as MATLAB designfilt does)")
    if VOICING_BAND_HZ[0] >= VOICING_BAND_HZ[1]:
        problems.append("voicing band low edge must be below the high edge")
    if not CUES:
        problems.append("the list of cues is empty")
    stray = [c for c in AMR_CUES if c not in CUES]
    if stray:
        problems.append(f"AMR cues {list(stray)} are not in the list of cues {list(CUES)}")
    if not AUDIO_EXTENSIONS:
        problems.append("no audio extensions given")
    if RUN_STATS and not GROUP_FACTORS:
        problems.append("statistics are on but no group factor is given")
    if RUN_STATS and not LEVENE_FACTOR:
        problems.append("statistics are on but no Levene factor is given")
    if problems:
        raise DDKError("DDK settings: " + "; ".join(problems))


def run_ddk(cfg, log=print, should_stop=lambda: False):
    """
    Run the DDK batch described by `cfg` (a dict of upper-case speechkit settings;
    only the DDK_* keys are read). `log` receives the progress lines; it defaults to
    print for command-line use. Returns a summary dict, or None if stopped.
    """
    global _LOG
    configure(cfg)
    previous, _LOG = _LOG, log
    try:
        return _run(dict(cfg or {}), should_stop)
    finally:
        _LOG = previous


if __name__ == "__main__":
    from speechkit.cli import main as _cli

    raise SystemExit(_cli(["ddk", *sys.argv[1:]]))
