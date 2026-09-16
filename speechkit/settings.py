"""
Every setting, in one place.

In the original scripts the tunable values were upper-case constants edited in
the source: a `CONFIG` block at the top of `phonemes_alignment.py`, a handful of
constants at the top of `phonemes_from_audio.py`, and a long `__main__` block at
the foot of `acoutstics.py`. They are all listed here instead, each with a type,
a default and the explanation that used to be a comment beside it.

The interface is generated from this list, so exposing a new setting means
adding one `Setting(...)` line here and reading it in the relevant module. There
is no second copy of the field list in the HTML.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Where saved settings live. Not in the package directory: the package may be
# installed read-only, and settings are per-user, not per-checkout.
# --------------------------------------------------------------------------
CONFIG_DIR = Path(os.environ.get("SPEECHKIT_CONFIG_DIR", Path.home() / ".speechkit"))
SETTINGS_FILE = CONFIG_DIR / "settings.json"
PRESET_DIR = CONFIG_DIR / "presets"


@dataclass(frozen=True)
class Setting:
    """One tunable value, and everything the interface needs to render it."""

    name: str                       # the upper-case name used in the config dict
    label: str                      # what the interface calls it
    kind: str                       # text|path|dir|paths|int|float|float_or_none|bool|choice|json|passages
    default: Any
    section: str                    # acoustics | phonemes | alignment | server
    group: str                      # fieldset heading inside the section
    help: str = ""
    choices: tuple[tuple[str, str], ...] = ()   # (value, label) for kind="choice"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    advanced: bool = False          # hidden behind "show advanced" by default
    depends_on: str = ""            # only meaningful when another setting has a value
    depends_value: Any = None
    # Which speech task this setting affects. Empty means both. Determined from
    # where the parameter is actually read in acoustics.py, not by guesswork:
    # the window and token settings are read only inside
    # analyze_sustained_vowel / _sustained_from_tokens, _load_boundaries is
    # called only on the sustained branch, min_usable_reps gates on sv_usable,
    # and compute_fluency is documented as not computed for sustained vowels.
    tasks: tuple[str, ...] = ()


# =============================================================================
# ACOUSTIC ANALYSIS  (was the __main__ block of acoustics.py)
# =============================================================================

_ACOUSTICS: tuple[Setting, ...] = (
    Setting("WAV_INPUTS", "Audio files or folders", "paths", [],
            "acoustics", "Input and output",
            "Folders are searched for .wav files; individual files can be added too. "
            "This replaces the WAV_FILES list that used to be pasted into the script."),
    Setting("RECURSIVE", "Include subfolders", "bool", True,
            "acoustics", "Input and output",
            "Search inside subfolders of any folder listed above."),
    Setting("OUTPUT_FOLDER", "Output folder", "dir", "",
            "acoustics", "Input and output",
            "Where the CSV files and plots are written. Created if missing."),
    Setting("OUTPUT_PREFIX", "Output file prefix", "text", "speechkit",
            "acoustics", "Input and output",
            "Every output file starts with this, e.g. speechkit_consolidated.csv."),
    Setting("TASK", "What is in these recordings?", "task", "",
            "acoustics", "Recording type",
            "The two tasks are measured differently and have different settings, so "
            "this comes first. Choose 'mixed' only if one folder really does hold both, "
            "in which case each file is routed by its folder and filename and then by "
            "its acoustics.",
            choices=(("reading", "Reading passage"),
                     ("sustained_vowel", "Sustained vowel"),
                     ("auto", "Mixed folder, detect each file"))),

    # ---- F0 integrity -----------------------------------------------------
    Setting("PITCH_RANGE_MODE", "Pitch range", "choice", "auto",
            "acoustics", "Pitch tracking",
            "A wrong pitch range is the single most damaging setting here: an octave "
            "error makes every file look aperiodic with nothing in the output pointing "
            "at the cause. 'Derive once' estimates one range for the batch and caches "
            "it, so later sessions of the same subject reuse it exactly.",
            choices=(("auto", "Derive once for the batch, then reuse the cache"),
                     ("manual", "Fixed floor and ceiling (entered below)"),
                     ("per_file", "Re-estimate for every file (not for longitudinal work)"))),
    Setting("PITCH_FLOOR_HZ", "Pitch floor (Hz)", "float", 110.0,
            "acoustics", "Pitch tracking",
            "Rule of thumb: floor is roughly the speaker's F0 divided by 1.7.",
            minimum=20.0, maximum=600.0, step=1.0,
            depends_on="PITCH_RANGE_MODE", depends_value="manual"),
    Setting("PITCH_CEILING_HZ", "Pitch ceiling (Hz)", "float", 330.0,
            "acoustics", "Pitch tracking",
            "Roughly F0 multiplied by 1.7.",
            minimum=50.0, maximum=1200.0, step=1.0,
            depends_on="PITCH_RANGE_MODE", depends_value="manual"),
    Setting("PITCH_CALIBRATION_FILE", "Pitch calibration file", "path", "",
            "acoustics", "Pitch tracking",
            "JSON file holding the derived range. Put it in the subject folder, above "
            "the per-session folders, so every session of that subject shares one range. "
            "Leave empty to derive without caching."),
    Setting("RECALIBRATE", "Re-derive and overwrite the calibration", "bool", False,
            "acoustics", "Pitch tracking",
            "Off means an existing calibration file is read verbatim, which is what "
            "keeps settings identical across sessions."),
    Setting("CALIBRATION_ON_AMBIGUOUS", "When the octave is ambiguous", "choice", "as_measured",
            "acoustics", "Pitch tracking",
            "Period doubling and diplophonia can leave the harmonic comb unable to "
            "settle the octave. Leave this alone until you have listened to one file.",
            choices=(("as_measured", "Keep as measured"),
                     ("upper", "Take the upper octave"),
                     ("lower", "Take the lower octave")),
            advanced=True),
    Setting("OCTAVE_CHECK", "Check F0 against the harmonic comb", "bool", True,
            "acoustics", "Pitch tracking",
            "Keep on as a monitor even when the floor is set correctly."),
    Setting("OCTAVE_AUTOCORRECT", "Correct clear octave errors", "bool", True,
            "acoustics", "Pitch tracking",
            "Only unambiguous subharmonic or doubling errors are corrected. Marginal "
            "cases are flagged, never changed."),
    Setting("OCTAVE_STRONG_DB", "Octave evidence threshold (dB)", "float", -10.0,
            "acoustics", "Pitch tracking",
            "How much stronger a competing harmonic must be before the correction fires.",
            minimum=-40.0, maximum=0.0, step=0.5, advanced=True),
    Setting("VOICING_THRESHOLD", "Voicing threshold", "float", 0.45,
            "acoustics", "Pitch tracking",
            "Praat's voicing decision threshold.",
            minimum=0.0, maximum=1.0, step=0.05, advanced=True),
    Setting("FORMANT_CEILING_HZ", "Formant ceiling (Hz)", "float_or_none", 5500.0,
            "acoustics", "Pitch tracking",
            "Fixed, so F1 and F2 stop depending on the tracked F0. Empty means derive "
            "per file, which was the pre-v10 behaviour.",
            minimum=2000.0, maximum=8000.0, step=50.0),

    # ---- tokenisation -----------------------------------------------------
    Setting("SINGLE_TOKEN_PER_FILE", "One production per file", "bool", True,
            "acoustics", "Token detection",
            "True when the protocol is exactly one reading passage or one sustained "
            "phonation per file: the whole speech span is the production, so there is "
            "nothing to segment. An internal dropout is then reported by the coverage "
            "ledger instead of becoming a second token.",
            tasks=("sustained_vowel",)),
    Setting("SPLIT_AT_SPLICES", "Split at detected edit points", "bool", False,
            "acoustics", "Token detection",
            "Leave off with one production per file: an edit point found inside a "
            "single take is a false positive that only shortens maximum phonation time.",
            tasks=("sustained_vowel",)),
    Setting("SPLICE_F0_STEP_ST", "Edit-point F0 step (semitones)", "float", 1.5,
            "acoustics", "Token detection",
            "How large an instantaneous F0 jump has to be to look like a splice.",
            minimum=0.1, maximum=12.0, step=0.1, advanced=True,
            depends_on="SPLIT_AT_SPLICES", depends_value=True,
            tasks=("sustained_vowel",)),
    Setting("MIN_VOWEL_TOKEN_DUR", "Minimum token duration (s)", "float", 0.8,
            "acoustics", "Token detection",
            "Shorter voiced stretches are not treated as a production.",
            minimum=0.05, maximum=10.0, step=0.05,
            tasks=("sustained_vowel",)),
    Setting("TOKEN_BRIDGE_GAP_S", "Bridge gaps shorter than (s)", "float", 0.25,
            "acoustics", "Token detection",
            "Two voiced stretches separated by less than this are joined into one token.",
            minimum=0.0, maximum=2.0, step=0.05,
            tasks=("sustained_vowel",)),
    Setting("TOKEN_MIN_VOICED_FRACTION", "Minimum voiced fraction per token", "float", 0.5,
            "acoustics", "Token detection",
            minimum=0.0, maximum=1.0, step=0.05, advanced=True,
            tasks=("sustained_vowel",)),
    Setting("BOUNDARIES_DIR", "Hand-cut boundaries folder", "dir", "",
            "acoustics", "Token detection",
            "Strongly preferred when you have cut the tokens yourself. Reads "
            "<stem><suffix> as a CSV of onset,offset pairs, or <stem>.TextGrid.",
            advanced=True,
            tasks=("sustained_vowel",)),
    Setting("BOUNDARIES_SUFFIX", "Boundaries file suffix", "text", "_tokens.csv",
            "acoustics", "Token detection", advanced=True,
            depends_on="BOUNDARIES_DIR", depends_value="__truthy__",
            tasks=("sustained_vowel",)),

    # ---- windows ----------------------------------------------------------
    Setting("VOWEL_WINDOW_MAX_S", "Analysis window length (s)", "float", 2.0,
            "acoustics", "Measurement windows",
            "Perturbation measures are taken inside steady windows of this length. "
            "2 s suits per-repetition files of 10 to 30 s; 3 s wastes the remainder.",
            minimum=0.5, maximum=10.0, step=0.1,
            tasks=("sustained_vowel",)),
    Setting("VOWEL_WINDOW_GRID_HOP_S", "Candidate grid step (s)", "float_or_none", None,
            "acoustics", "Measurement windows",
            "Step of the candidate-window grid. Empty means half the window length. "
            "A finer step such as 0.5 s lets a window slide around a short wobble "
            "instead of losing the whole slot, raising yield without loosening any "
            "validity criterion. Re-run the whole batch after changing it.",
            minimum=0.05, maximum=5.0, step=0.05,
            tasks=("sustained_vowel",)),
    Setting("VOWEL_EDGE_TRIM_S", "Trim from each token edge (s)", "float", 0.25,
            "acoustics", "Measurement windows",
            "Onset and offset transients are excluded from measurement.",
            minimum=0.0, maximum=2.0, step=0.05,
            tasks=("sustained_vowel",)),
    Setting("WINDOW_MIN_VOICED_FRACTION", "Window must be this voiced", "float", 0.90,
            "acoustics", "Measurement windows",
            minimum=0.0, maximum=1.0, step=0.01,
            tasks=("sustained_vowel",)),
    Setting("WINDOW_MAX_F0_DEVIATION_ST", "Max F0 deviation in window (st)", "float", 3.0,
            "acoustics", "Measurement windows",
            minimum=0.1, maximum=24.0, step=0.1,
            tasks=("sustained_vowel",)),
    Setting("WINDOW_MAX_INTERNAL_STEP_ST", "Max F0 step inside window (st)", "float", 1.0,
            "acoustics", "Measurement windows",
            minimum=0.05, maximum=12.0, step=0.05,
            tasks=("sustained_vowel",)),
    Setting("VOWEL_WINDOW_HOP_S", "Window hop (s, unused)", "float", 2.0,
            "acoustics", "Measurement windows",
            "Ignored since v8, kept so old scripts and saved settings still load.",
            minimum=0.1, maximum=10.0, step=0.1, advanced=True,
            tasks=("sustained_vowel",)),

    # ---- gates ------------------------------------------------------------
    Setting("MIN_VALID_WINDOWS", "Valid windows needed per repetition", "int", 3,
            "acoustics", "Usability gates",
            "Mind the geometry this implies: with non-overlapping windows a take needs "
            "windows x window length + 2 x edge trim seconds of perfectly steady "
            "phonation to pass at all, and every rejected slot costs another window "
            "length. Check sv_analyzed_fraction and sv_steady_frame_fraction before "
            "lowering it; if the frames are steady and the gate still fails, the gate "
            "is the problem rather than the voice.",
            minimum=1, maximum=20, step=1,
            tasks=("sustained_vowel",)),
    Setting("MIN_MEASURED_S", "Measured seconds needed per repetition", "float", 6.0,
            "acoustics", "Usability gates",
            minimum=0.5, maximum=60.0, step=0.5,
            tasks=("sustained_vowel",)),
    Setting("MIN_USABLE_REPS", "Usable repetitions needed per session", "int", 2,
            "acoustics", "Usability gates",
            "Drives the enough_reps column of the by-session CSV.",
            minimum=1, maximum=20, step=1,
            tasks=("sustained_vowel",)),

    # ---- outputs ----------------------------------------------------------
    Setting("MAKE_INDIVIDUAL_PLOTS", "Bar plot per metric", "bool", True,
            "acoustics", "Outputs"),
    Setting("MAKE_OVERVIEW", "Overview grid", "bool", True,
            "acoustics", "Outputs"),
    Setting("PLOT_SCOPE", "Metrics to plot", "choice", "core",
            "acoustics", "Outputs",
            choices=(("core", "Curated set for the detected task"),
                     ("all", "Every computed metric"))),
    Setting("ON_AGGREGATION", "Combine repetitions by", "choice", "all",
            "acoustics", "Outputs",
            choices=(("all", "Keep every repetition as its own column"),
                     ("median", "Median across repetitions"),
                     ("best", "Best repetition per metric"))),
    Setting("ENABLE_FLUENCY_INDEX", "Compute the composite fluency index", "bool", False,
            "acoustics", "Outputs",
            "A composite of several metrics. Off by default because it is derived "
            "rather than measured.", advanced=True,
            tasks=("reading",)),
    Setting("LEGACY_COMPAT", "Legacy-compatible output columns", "bool", False,
            "acoustics", "Outputs",
            "For re-running an older analysis and diffing the CSVs.", advanced=True),
    Setting("KNOWN_PASSAGE_SYLLABLES", "Passage syllable counts", "passages",
            {"caterpillar": 260, "grandfather": 174, "rainbow": 459, "outside": 195,
             "visittomarket": 204, "daily": 209, "house": 181, "community": 350},
            "acoustics", "Passage syllable counts",
            "Matched case-insensitively against the filename stem, and used to make the "
            "speech rate exact rather than estimated. Check each count against your own "
            "wording of the passage: a wrong count biases the rate for every file of "
            "that passage. The speaking_time_fraction metric does not depend on these.",
            tasks=("reading",)),
)

# =============================================================================
# PHONEME RECOGNITION  (was the constants and __main__ of phonemes_from_audio.py)
# =============================================================================

_PHONEMES: tuple[Setting, ...] = (
    Setting("PHONEME_INPUTS", "Audio files or folders", "paths", [],
            "phonemes", "Input and output",
            "Replaces the folders and extra_files lists at the foot of the script."),
    Setting("PHONEME_RECURSIVE", "Include subfolders", "bool", False,
            "phonemes", "Input and output"),
    Setting("PHONEME_OUTPUT_DIR", "Output folder for TSV files", "dir", "",
            "phonemes", "Input and output",
            "Leave empty to write each TSV next to its .wav, which is what the original "
            "script did. Set it to keep results out of the audio folders."),
    Setting("PHONEME_TSV_SUFFIX", "Batch TSV suffix", "text", "_auto.tsv",
            "phonemes", "Input and output",
            "Suffix for TSV files written without review. Files saved from the editor "
            "keep the original _edited.tsv suffix, so a reviewed file is always "
            "distinguishable from an unreviewed one."),

    Setting("MODEL_NAME", "Phoneme model", "text", "facebook/wav2vec2-lv-60-espeak-cv-ft",
            "phonemes", "Recognition",
            "Any wav2vec2 CTC model whose vocabulary is phonemes. Downloaded once and "
            "cached by huggingface_hub; check the model's own licence before "
            "redistributing results."),
    Setting("DEVICE", "Compute device", "choice", "auto",
            "phonemes", "Recognition",
            choices=(("auto", "CUDA if available, otherwise CPU"),
                     ("cuda", "CUDA"),
                     ("cpu", "CPU"))),
    Setting("SAMPLE_RATE", "Model sample rate (Hz)", "int", 16000,
            "phonemes", "Recognition",
            "Audio is resampled to this. It must match what the model was trained on; "
            "16000 for every wav2vec2 checkpoint.",
            minimum=8000, maximum=48000, step=1000, advanced=True),

    Setting("FRAME_DURATION_MS", "Energy frame length (ms)", "int", 20,
            "phonemes", "Pause detection",
            "Resolution of the RMS envelope used to find pauses.",
            minimum=5, maximum=100, step=5),
    Setting("MIN_PAUSE_DURATION_S", "Minimum pause (s)", "float", 0.15,
            "phonemes", "Pause detection",
            "Silences shorter than this are not counted as pauses.",
            minimum=0.02, maximum=2.0, step=0.01),
    Setting("NOISE_K", "Noise margin (robust sigma)", "float", 3.0,
            "phonemes", "Pause detection",
            "How far above the estimated noise floor a frame must sit to count as "
            "speech. The threshold actually used is the higher of this and Otsu's "
            "threshold on the RMS histogram, so neither a noisy room nor a very quiet "
            "one needs hand tuning.",
            minimum=0.5, maximum=10.0, step=0.1),
)

# =============================================================================
# PHONEME-TO-TEXT ALIGNMENT  (was the CONFIG block of phonemes_alignment.py)
# =============================================================================

_ALIGNMENT: tuple[Setting, ...] = (
    Setting("INPUT_DIRS", "Passage folders", "paths", [],
            "alignment", "Input and output",
            "Each folder is processed independently. A folder needs, for every passage, "
            "a <passage>.txt with the reference text and a <passage>_phonemes.txt with "
            "the canonical phoneme sequence, plus the session TSV files."),
    Setting("PASSAGES", "Passages", "text", "",
            "alignment", "Input and output",
            "Comma-separated passage names, e.g. rainbow, grandfather, caterpillar. "
            "Leave empty to discover every passage in each folder."),
    Setting("OUTPUT_SUFFIX", "Output filename suffix", "text", "_all_aligned.txt",
            "alignment", "Input and output",
            "Written into the passage folder as <passage><suffix>."),
    Setting("PREFER_EDITED_TSV", "Prefer reviewed TSV files", "bool", True,
            "alignment", "Input and output",
            "When a recording has both a reviewed _edited.tsv and an unreviewed batch "
            "TSV, use only the reviewed one instead of aligning the same recording twice."),

    Setting("PAUSE_THRESHOLD", "Show a pause marker above (s)", "float", 0.7,
            "alignment", "Alignment",
            "Pauses at least this long become a ... column in the text line. Shorter "
            "ones only widen the spacing. Lower means more markers."),
    Setting("G2P_BACKEND", "Word-boundary pronunciations from", "choice", "builtin",
            "alignment", "Alignment",
            "Used only to decide which word each canonical phoneme belongs to; the "
            "canonical sequence in <passage>_phonemes.txt is always what gets printed. "
            "The built-in guesser is a rough English letter-to-sound table. espeak-ng, "
            "through the phonemizer package, is better and works for other languages.",
            choices=(("builtin", "Built-in English guesser"),
                     ("phonemizer", "espeak-ng via phonemizer (if installed)"))),
    Setting("PHONEMIZER_LANGUAGE", "espeak-ng language", "text", "en-us",
            "alignment", "Alignment",
            depends_on="G2P_BACKEND", depends_value="phonemizer"),
    Setting("VERIFY", "Check that word splitting is lossless", "bool", True,
            "alignment", "Alignment",
            "Prints whether the per-word split reconstructs the canonical phoneme "
            "sequence exactly. Useful when adding a passage."),
    Setting("MANUAL_PASSAGES", "Manual passage list", "json", [],
            "alignment", "Pinned pronunciations",
            'Bypasses the naming convention entirely and overrides discovery. A list '
            'of objects, one per passage: {"name": "rainbow", "text": "...rainbow.txt", '
            '"phonemes": "...rainbow_phonemes.txt", "tsvs": ["...OFF.tsv", "...ON.tsv"]}. '
            'Paths are used as given. Leave empty unless discovery cannot see your '
            'layout.', advanced=True),
    Setting("USER_DICTS", "Pinned pronunciations", "json", {},
            "alignment", "Pinned pronunciations",
            'Pin exact word boundaries per passage, for words the guesser gets wrong. '
            'Space-separate multi-character IPA tokens exactly as they appear in the '
            '_phonemes.txt file. Anything that cannot be reconciled with the canonical '
            'sequence is ignored, so this can never break a run. Example: '
            '{"rainbow": {"sun\'s": "s ʌ n z", "rainbow": "ɹ eɪ n b oʊ"}}'),
)

# =============================================================================
# SERVER
# =============================================================================

_SERVER: tuple[Setting, ...] = (
    Setting("HOST", "Bind address", "text", "127.0.0.1",
            "server", "Local server",
            "Leave as 127.0.0.1 so the interface is reachable from this machine only. "
            "There is no authentication and the file browser can see the whole disk, so "
            "do not bind it to a public interface."),
    Setting("PORT", "Port", "int", 7331,
            "server", "Local server", minimum=1024, maximum=65535, step=1),
    Setting("OPEN_BROWSER", "Open a browser on start", "bool", True,
            "server", "Local server"),
    Setting("ALLOW_DUPLICATE_OPENMP", "Tolerate duplicate OpenMP runtimes", "bool", True,
            "server", "Local server",
            "Sets KMP_DUPLICATE_LIB_OK, which torch and MKL together often need on "
            "Windows. Turn it off only if you want the process to abort instead.",
            advanced=True),
)

SETTINGS: tuple[Setting, ...] = _ACOUSTICS + _PHONEMES + _ALIGNMENT + _SERVER
BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}

SECTION_TITLES = {
    "acoustics": "Acoustic analysis",
    "phonemes": "Phoneme recognition",
    "alignment": "Alignment to text",
    "server": "Server",
}


def defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in SETTINGS:
        out[s.name] = json.loads(json.dumps(s.default))  # deep copy, JSON-safe
    return out


# --------------------------------------------------------------------------
# Coercion. Values arrive from an HTML form as strings, so every kind needs an
# explicit conversion with an error message a user can act on.
# --------------------------------------------------------------------------

class SettingError(ValueError):
    """A setting could not be interpreted; the message names the field."""


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def coerce(name: str, raw: Any) -> Any:
    s = BY_NAME.get(name)
    if s is None:
        raise SettingError(f"unknown setting: {name}")
    try:
        return _coerce_kind(s, raw)
    except SettingError:
        raise
    except (TypeError, ValueError) as exc:
        raise SettingError(f"{s.label}: {exc}") from exc


def _coerce_kind(s: Setting, raw: Any) -> Any:
    kind = s.kind
    if kind == "bool":
        return _as_bool(raw)

    if kind in {"int", "float"}:
        text = str(raw).strip().replace(",", ".")
        if text == "":
            return s.default
        value = int(round(float(text))) if kind == "int" else float(text)
        return _range_check(s, value)

    if kind == "float_or_none":
        text = str(raw).strip().replace(",", ".")
        if text == "" or text.lower() in {"none", "null", "auto"}:
            return None
        return _range_check(s, float(text))

    if kind in {"choice", "task"}:
        value = str(raw).strip()
        allowed = [c[0] for c in s.choices]
        # A task starts unchosen on purpose: the interface asks for it before
        # showing anything else, and run_acoustics refuses to start without it.
        if kind == "task" and value == "":
            return ""
        if value not in allowed:
            raise SettingError(f"{s.label}: expected one of {', '.join(allowed)}")
        return value

    if kind == "paths":
        if isinstance(raw, str):
            items = [line.strip() for line in raw.splitlines()]
        else:
            items = [str(x).strip() for x in (raw or [])]
        return [i for i in items if i]

    if kind in {"text", "path", "dir"}:
        return str(raw).strip()

    if kind == "json":
        if isinstance(raw, (dict, list)):
            return raw
        text = str(raw).strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SettingError(f"{s.label}: not valid JSON ({exc.msg} at position {exc.pos})")

    if kind == "passages":
        return _coerce_passages(s, raw)

    raise SettingError(f"{s.label}: unsupported kind {kind!r}")


def _range_check(s: Setting, value: float) -> float:
    if s.minimum is not None and value < s.minimum:
        raise SettingError(f"{s.label}: must be at least {s.minimum}")
    if s.maximum is not None and value > s.maximum:
        raise SettingError(f"{s.label}: must be at most {s.maximum}")
    return value


def _coerce_passages(s: Setting, raw: Any) -> dict[str, int | None]:
    """Accept either a JSON object or `name = count` lines, one per passage."""
    if isinstance(raw, dict):
        items = list(raw.items())
    else:
        text = str(raw).strip()
        if not text:
            return {}
        if text.startswith("{"):
            items = list(json.loads(text).items())
        else:
            items = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for sep in ("=", ":", "\t", ","):
                    if sep in line:
                        key, _, value = line.partition(sep)
                        items.append((key, value))
                        break
                else:
                    raise SettingError(
                        f"{s.label}: cannot read {line!r}; use 'caterpillar = 260'")
    out: dict[str, int | None] = {}
    for key, value in items:
        name = str(key).strip().strip('"').lower()
        if not name:
            continue
        text = str(value).strip()
        if text in {"", "none", "null"}:
            out[name] = None
            continue
        try:
            out[name] = int(float(text))
        except ValueError:
            raise SettingError(f"{s.label}: {name!r} needs a whole number, got {text!r}")
    return out


def coerce_all(incoming: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Merge `incoming` onto `base` (defaults if omitted), coercing each value and
    collecting every error rather than stopping at the first, so the interface
    can highlight all the bad fields at once.
    """
    out = dict(base or defaults())
    errors: list[str] = []
    for name, raw in incoming.items():
        if name not in BY_NAME:
            continue  # ignore unknown keys: old saved files stay loadable
        try:
            out[name] = coerce(name, raw)
        except SettingError as exc:
            errors.append(str(exc))
    if errors:
        raise SettingError("; ".join(errors))
    return out


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def load(path: Path | None = None) -> dict[str, Any]:
    """Saved settings merged onto the defaults. Unreadable files fall back to
    the defaults rather than stopping the interface from opening."""
    target = Path(path) if path else SETTINGS_FILE
    values = defaults()
    if not target.is_file():
        return values
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return values
    if not isinstance(stored, dict):
        return values
    try:
        return coerce_all(stored, values)
    except SettingError:
        # keep whatever individual values do load
        for name, raw in stored.items():
            try:
                values[name] = coerce(name, raw)
            except SettingError:
                pass
        return values


def save(values: dict[str, Any], path: Path | None = None) -> Path:
    target = Path(path) if path else SETTINGS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {s.name: values.get(s.name, s.default) for s in SETTINGS}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def list_presets() -> list[str]:
    if not PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def preset_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise SettingError("preset name must contain a letter or a number")
    return PRESET_DIR / f"{safe}.json"


def schema() -> list[dict[str, Any]]:
    """The field list, as JSON for the interface."""
    return [
        {
            "name": s.name, "label": s.label, "kind": s.kind, "default": s.default,
            "section": s.section, "group": s.group, "help": s.help,
            "choices": [{"value": v, "label": l} for v, l in s.choices],
            "min": s.minimum, "max": s.maximum, "step": s.step,
            "advanced": s.advanced,
            "tasks": list(s.tasks),
            "dependsOn": s.depends_on, "dependsValue": s.depends_value,
        }
        for s in SETTINGS
    ]
