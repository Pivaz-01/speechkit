"""
Aligning produced phonemes to the reference text of a reading passage.

The algorithm from ``phonemes_alignment.py``, unchanged: Needleman-Wunsch over
token lists, a rough letter-to-sound guess used only to decide which word each
canonical phoneme belongs to, and the canonical sequence as the source of truth
for what gets printed.

What changed:

* the CONFIG block at the top (INPUT_DIRS, PASSAGES, PAUSE_THRESHOLD, VERIFY,
  MANUAL_PASSAGES, USER_DICTS) is now function parameters, supplied by the
  interface;
* `print` became a `log` callback, so progress can be streamed to the browser;
* espeak-ng, via the optional `phonemizer` package, can replace the built-in
  guesser for word-boundary assignment. This matters for any passage or
  language the built-in English table was not written for;
* when a recording has both a reviewed ``_edited.tsv`` and an unreviewed batch
  TSV, only one is used, so the same recording is not aligned twice.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

# =============================================================================
# Needleman-Wunsch global alignment over token lists.
# Returns (i, j) pairs; i indexes `a` (None for an insertion in b) and j indexes
# `b` (None for a deletion from a).
# =============================================================================


def nw_align(a, b, match=0, mismatch=1, gap=1):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        di = dp[i - 1]
        ci = dp[i]
        for j in range(1, m + 1):
            sub = di[j - 1] + (match if ai == b[j - 1] else mismatch)
            ci[j] = sub
            d = di[j] + gap
            if d < ci[j]:
                ci[j] = d
            ins = ci[j - 1] + gap
            if ins < ci[j]:
                ci[j] = ins
    i, j = n, m
    out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            match if a[i - 1] == b[j - 1] else mismatch
        ):
            out.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    out.reverse()
    return out


# =============================================================================
# Grapheme to phoneme, for word-boundary assignment only.
# =============================================================================

_G2P_MULTI = [
    ("tch", ["tʃ"]), ("dge", ["dʒ"]),
    ("sh", ["ʃ"]), ("ch", ["tʃ"]), ("th", ["θ"]), ("ph", ["f"]),
    ("ck", ["k"]), ("qu", ["k", "w"]), ("ng", ["ŋ"]), ("wh", ["w"]),
    ("oo", ["uː"]), ("ee", ["iː"]), ("ea", ["iː"]), ("ai", ["eɪ"]),
    ("ay", ["eɪ"]), ("oa", ["oʊ"]), ("ow", ["aʊ"]), ("ou", ["aʊ"]),
    ("igh", ["aɪ"]), ("ie", ["aɪ"]), ("oi", ["ɔɪ"]), ("oy", ["ɔɪ"]),
    ("ar", ["ɑːɹ"]), ("or", ["ɔːɹ"]), ("er", ["ɚ"]), ("ir", ["ɜː"]),
    ("ur", ["ɜː"]), ("aw", ["ɔː"]), ("au", ["ɔː"]),
]
_G2P_SINGLE = {
    "a": ["æ"], "b": ["b"], "c": ["k"], "d": ["d"], "e": ["ɛ"], "f": ["f"],
    "g": ["ɡ"], "h": ["h"], "i": ["ɪ"], "j": ["dʒ"], "k": ["k"], "l": ["l"],
    "m": ["m"], "n": ["n"], "o": ["ɑː"], "p": ["p"], "q": ["k"], "r": ["ɹ"],
    "s": ["s"], "t": ["t"], "u": ["ʌ"], "v": ["v"], "w": ["w"], "x": ["k", "s"],
    "y": ["j"], "z": ["z"], "'": [],
}


def rough_g2p(word: str) -> list[str]:
    """
    Scaffolding only: it has to be close enough that each canonical phoneme
    lands under the right word. Nothing it produces is printed.
    """
    w = word.lower()
    out: list[str] = []
    i = 0
    while i < len(w):
        for src, ipa in _G2P_MULTI:
            if w.startswith(src, i):
                out.extend(ipa)
                i += len(src)
                break
        else:
            out.extend(_G2P_SINGLE.get(w[i], []))
            i += 1
    if not out:
        out = ["ə"]
    return out


class G2P:
    """
    Word-to-phoneme guesser. `builtin` is the table above; `phonemizer` calls
    espeak-ng, which is considerably better and not English-only, and falls
    back to the table with a warning if the package is missing.
    """

    def __init__(self, backend: str = "builtin", language: str = "en-us",
                 log: Callable[[str], None] = print):
        self.backend = "builtin"
        self.language = language
        self._cache: dict[str, list[str]] = {}
        if backend == "phonemizer":
            try:
                from phonemizer.backend import EspeakBackend

                self._espeak = EspeakBackend(
                    language, preserve_punctuation=False, with_stress=False)
                self.backend = "phonemizer"
                log(f"[g2p] espeak-ng via phonemizer, language {language}")
            except Exception as exc:
                log(f"[g2p] phonemizer unavailable ({exc}); using the built-in "
                    f"English guesser")

    def __call__(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]
        if self.backend == "phonemizer":
            try:
                text = self._espeak.phonemize([word], strip=True)[0]
                tokens = [t for t in re.split(r"\s+|(?=\u02c8)|(?=\u02cc)", text) if t]
                result = tokens or rough_g2p(word)
            except Exception:
                result = rough_g2p(word)
        else:
            result = rough_g2p(word)
        self._cache[word] = result
        return result


# =============================================================================
# Splitting the canonical phoneme list across words.
# =============================================================================


def assign_words_to_phonemes(words, canon, g2p: Callable[[str], list[str]] = rough_g2p):
    """
    Returns per_word (parallel to `words`, each a list of canonical phonemes)
    and phoneme_word (canonical index -> word index).
    """
    guess: list[str] = []
    guess_word: list[int] = []
    for wi, w in enumerate(words):
        for p in g2p(w):
            guess.append(p)
            guess_word.append(wi)

    al = nw_align(guess, canon)
    phoneme_word: list[int | None] = [None] * len(canon)
    cur = 0
    for gi, ci in al:
        if gi is not None:
            cur = guess_word[gi]
        if ci is not None:
            phoneme_word[ci] = cur
    last = 0
    for k in range(len(phoneme_word)):
        if phoneme_word[k] is None or phoneme_word[k] < last:
            phoneme_word[k] = last
        last = phoneme_word[k]

    per_word: list[list[str]] = [[] for _ in words]
    for k, wi in enumerate(phoneme_word):
        per_word[wi].append(canon[k])
    return per_word, phoneme_word


def assign_with_dict(words, canon, user_dict,
                     g2p: Callable[[str], list[str]] = rough_g2p,
                     log: Callable[[str], None] = print):
    """
    Honour `user_dict` where a supplied pronunciation fits the canonical
    sequence at that position, and auto-segment everywhere else. Guaranteed to
    reconstruct `canon` exactly, falling back to pure automatic segmentation if
    a supplied dictionary cannot be reconciled.
    """
    auto_per, _ = assign_words_to_phonemes(words, canon, g2p)
    per = [list(chunk) for chunk in auto_per]

    out: list[tuple[int, str]] = []
    ci = 0
    overridden = 0
    for wi, w in enumerate(words):
        dict_pron = user_dict.get(w.lower())
        auto_slice = per[wi]
        if dict_pron is not None and canon[ci:ci + len(dict_pron)] == dict_pron:
            chunk = dict_pron
            if chunk != auto_slice:
                overridden += 1
        else:
            chunk = auto_slice
        out.extend((wi, p) for p in chunk)
        ci += len(chunk)

    rebuilt = [p for _, p in out]
    if rebuilt != canon:
        phoneme_word: list[int] = []
        per2: list[list[str]] = [[] for _ in words]
        for wi, chunk in enumerate(auto_per):
            for p in chunk:
                per2[wi].append(p)
                phoneme_word.append(wi)
        log("  [dict] supplied dictionary could not be reconciled with the "
            "phoneme sequence; using automatic segmentation.")
        return per2, phoneme_word

    if overridden:
        log(f"  [dict] applied {overridden} dictionary override(s); "
            f"reconstruction OK (exact)")
    phoneme_word = [wi for wi, _ in out]
    per_final: list[list[str]] = [[] for _ in words]
    for wi, p in out:
        per_final[wi].append(p)
    return per_final, phoneme_word


# =============================================================================
# File parsing
# =============================================================================


def read_words(text_path: str) -> list[str]:
    with open(text_path, encoding="utf-8") as f:
        return re.findall(r"[A-Za-z']+", f.read())


def read_canonical(phonemes_path: str) -> list[str]:
    with open(phonemes_path, encoding="utf-8") as f:
        raw = f.read()
    if "===" in raw:
        raw = raw.split("===")[-1]
    return raw.split()


def read_session(tsv_path: str) -> list[tuple[str, Any]]:
    """A list of ('ph', label) and ('pause', duration_seconds)."""
    seq: list[tuple[str, Any]] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if cols[0] == "type":
                continue
            kind = cols[0]
            label = cols[1] if len(cols) > 1 else ""
            try:
                start = float(cols[3])
                end = float(cols[4])
            except (IndexError, ValueError):
                start = end = 0.0
            if kind == "pause":
                seq.append(("pause", end - start))
            elif kind == "phoneme":
                seq.append(("ph", label))
    return seq


# =============================================================================
# Session to sibling file resolution, the "-0N" convention.
#
# If a session filename contains a "-0N" (dash plus digits), e.g.
# "caterpillar_20260413_post-03_edited.tsv", look for a same-numbered sibling
# next to the passage's default text and phoneme files, e.g.
# "caterpillar-03.txt" and "caterpillar_phonemes-03.txt". If it exists it is
# used for that session; otherwise the default is. Any "-0N" already on the
# default is stripped first, so segments never double up.
# =============================================================================

_DEFAULT_SUFFIX_RE = re.compile(r"-(\d+)$")
_SESSION_SUFFIX_RE = re.compile(r"-(\d+)\D*$")


def _strip_session_suffix(base: str) -> str:
    m = _DEFAULT_SUFFIX_RE.search(base)
    return base[: -len(m.group(0))] if m else base


def _sibling_for_session(session_path: str, default_path: str) -> str:
    stem = os.path.splitext(os.path.basename(session_path))[0]
    m = _SESSION_SUFFIX_RE.search(stem)
    if not m:
        return default_path
    suffix = f"-{m.group(1)}"
    folder, fname = os.path.split(default_path)
    base, ext = os.path.splitext(fname)
    base = _strip_session_suffix(base)
    candidate = os.path.join(folder, f"{base}{suffix}{ext}")
    return candidate if os.path.exists(candidate) else default_path


def phonemes_path_for_session(session_path, default_phonemes_path):
    return _sibling_for_session(session_path, default_phonemes_path)


def text_path_for_session(session_path, default_text_path):
    return _sibling_for_session(session_path, default_text_path)


# =============================================================================
# Building the two aligned lines for one session.
# =============================================================================


def build_block(words, canon, phoneme_word, session_seq, pause_threshold):
    hyp: list[str] = []
    long_pause_before: set[int] = set()
    count = 0
    for kind, val in session_seq:
        if kind == "ph":
            hyp.append(val)
            count += 1
        elif val >= pause_threshold and count > 0:
            long_pause_before.add(count)

    al = nw_align(canon, hyp)
    hyp_to_canon = {h: c for c, h in al if h is not None}

    word_tokens: list[list[str]] = [[] for _ in words]
    word_pause_after = [False] * len(words)
    cur = 0
    for h in range(len(hyp)):
        c = hyp_to_canon.get(h)
        if c is not None:
            cur = phoneme_word[c]
        word_tokens[cur].append(hyp[h])
        if (h + 1) in long_pause_before:
            word_pause_after[cur] = True

    ph_cells, tx_cells = [], []
    for wi, w in enumerate(words):
        ph_cells.append("".join(word_tokens[wi]))
        tx_cells.append(w)
        if word_pause_after[wi]:
            ph_cells.append("")
            tx_cells.append("...")

    line_ph, line_tx = [], []
    for pc, tc in zip(ph_cells, tx_cells):
        width = max(len(pc), len(tc))
        line_ph.append(pc.ljust(width))
        line_tx.append(tc.ljust(width))
    return " ".join(line_ph).rstrip(), " ".join(line_tx).rstrip()


def score_block(canon, session_seq) -> dict[str, Any]:
    """
    Phoneme-level accuracy for one session, from the same alignment the printed
    block uses. Reported so the interface can show more than a file count, and
    so the numbers can go in a table without re-parsing the text output.
    """
    hyp = [v for k, v in session_seq if k == "ph"]
    al = nw_align(canon, hyp)
    hits = subs = dels = ins = 0
    for c, h in al:
        if c is None:
            ins += 1
        elif h is None:
            dels += 1
        elif canon[c] == hyp[h]:
            hits += 1
        else:
            subs += 1
    total = len(canon)
    return {
        "canonical": total,
        "produced": len(hyp),
        "correct": hits,
        "substitutions": subs,
        "omissions": dels,
        "insertions": ins,
        "accuracy": hits / total if total else 0.0,
        "error_rate": (subs + dels + ins) / total if total else 0.0,
    }


def session_title(tsv_path: str) -> str:
    return os.path.splitext(os.path.basename(tsv_path))[0]


def get_alignment(text_path, phonemes_path, user_dict, cache, g2p,
                  verify=False, tag=None, log: Callable[[str], None] = print):
    """
    (words, canon, per_word, phoneme_word) for one text and phoneme file pair,
    cached so sessions sharing a pair do not redo the segmentation.
    """
    key = (text_path, phonemes_path)
    if key not in cache:
        words = read_words(text_path)
        canon = read_canonical(phonemes_path)
        if user_dict:
            per_word, phoneme_word = assign_with_dict(words, canon, user_dict, g2p, log)
        else:
            per_word, phoneme_word = assign_words_to_phonemes(words, canon, g2p)
        cache[key] = (words, canon, per_word, phoneme_word)

        if verify:
            rebuilt = [p for chunk in per_word for p in chunk]
            ok = rebuilt == canon
            label = tag or os.path.basename(text_path)
            log(f"  [verify:{label}] {len(words)} words, {len(canon)} canonical "
                f"phonemes, reconstruction {'OK (exact)' if ok else 'MISMATCH'}")
            if not ok:
                for k in range(min(len(rebuilt), len(canon))):
                    if rebuilt[k] != canon[k]:
                        log(f"           first diff at {k}: "
                            f"rebuilt={rebuilt[k]} canonical={canon[k]}")
                        break
    return cache[key]


# =============================================================================
# One passage, end to end.
# =============================================================================


def align_passage(name, text_path, phonemes_path, tsv_paths, out_path,
                  pause_threshold, verify, user_dict, g2p,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    cache: dict = {}
    get_alignment(text_path, phonemes_path, user_dict, cache, g2p, verify,
                  tag=name, log=log)

    if not tsv_paths:
        # Reporting "none found" alone sends people hunting for a bug that is
        # almost always a filename or a folder mismatch, so say what was looked
        # for and what is actually sitting there.
        folder = os.path.dirname(text_path) or "."
        log(f"  no session files found in {folder}")
        log(f"  looked for any .tsv whose name contains '{name}'")
        present = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".tsv")) if os.path.isdir(folder) else []
        if present:
            log(f"  the folder does contain {len(present)} .tsv file(s), but none "
                f"mention '{name}':")
            for f in present[:8]:
                log(f"      {f}")
            if len(present) > 8:
                log(f"      ... and {len(present) - 8} more")
            log(f"  the passage name has to appear somewhere in the filename, "
                f"e.g. {name}_speaker01_edited.tsv or "
                f"speaker01_{name}_edited.tsv")
        else:
            log("  no .tsv files in that folder at all. The phoneme stage writes "
                "them next to the audio, or into its own output folder, so copy "
                "or move them here first.")
        log("  nothing written for this passage.")
        return {"passage": name, "sessions": [], "output": None}

    blocks: list[str] = []
    scores: list[dict[str, Any]] = []
    for tsv in sorted(tsv_paths):
        session_text_path = text_path_for_session(tsv, text_path)
        session_phon_path = phonemes_path_for_session(tsv, phonemes_path)
        seg_tag = os.path.splitext(os.path.basename(session_text_path))[0]
        s_words, s_canon, _, s_phoneme_word = get_alignment(
            session_text_path, session_phon_path, user_dict, cache, g2p,
            verify, tag=seg_tag, log=log)

        tag_bits = []
        if session_text_path != text_path:
            tag_bits.append(os.path.basename(session_text_path))
        if session_phon_path != phonemes_path:
            tag_bits.append(os.path.basename(session_phon_path))
        tag = f"  [{', '.join(tag_bits)}]" if tag_bits else ""

        seq = read_session(tsv)
        ph_line, tx_line = build_block(s_words, s_canon, s_phoneme_word, seq,
                                       pause_threshold)
        title = session_title(tsv)
        blocks.append(f"=== {title} ===\n{ph_line}\n{tx_line}")

        score = score_block(s_canon, seq)
        score["session"] = title
        score["tsv"] = tsv
        scores.append(score)
        log(f"  {title:42} phonemes={score['produced']:4}  "
            f"accuracy={score['accuracy'] * 100:5.1f}%  "
            f"ellipses={tx_line.count('...'):3}{tag}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks).rstrip() + "\n")
    log(f"  -> wrote {out_path}  ({len(blocks)} session blocks)")
    return {"passage": name, "sessions": scores, "output": out_path}


# =============================================================================
# Deciding which passages to run.
# =============================================================================


def _find_segment_pairs(prefix: str, folder: str) -> dict[str, tuple[str, str]]:
    """
    Segmented text and phoneme pairs for `prefix`: "<prefix>-0N.txt" alongside
    "<prefix>_phonemes-0N.txt". Returns {digits: (text, phonemes)} for every N
    where both files exist.
    """
    pairs: dict[str, tuple[str, str]] = {}
    try:
        entries = os.listdir(folder)
    except OSError:
        return pairs
    text_re = re.compile(rf"^{re.escape(prefix)}-(\d+)\.txt$")
    for fn in entries:
        m = text_re.match(fn)
        if not m:
            continue
        digits = m.group(1)
        phon_fn = f"{prefix}_phonemes-{digits}.txt"
        if phon_fn in entries:
            pairs[digits] = (os.path.join(folder, fn), os.path.join(folder, phon_fn))
    return pairs



def find_session_files(prefix: str, folder: str,
                       log: Callable[[str], None] = lambda *a: None) -> list[str]:
    """
    Session TSVs belonging to `prefix`.

    The original convention was "<passage>_*.tsv", which requires recordings to
    be named passage-first. Speaker-first naming is at least as natural, and a
    file called speaker01_caterpillar_edited.tsv obviously belongs to the
    caterpillar passage, so the passage name is accepted anywhere in the stem.

    Prefix matches are preferred, and the looser search only runs when there are
    none, so an existing passage-first layout behaves exactly as before. A file
    naming two different passages is skipped rather than guessed at.
    """
    try:
        entries = [f for f in os.listdir(folder) if f.lower().endswith(".tsv")]
    except OSError:
        return []

    # the canonical phoneme file is not a session
    entries = [f for f in entries if not f.startswith(f"{prefix}_phonemes.")]

    strict = [f for f in entries if f.startswith(f"{prefix}_")]
    if strict:
        # Mixed conventions in one folder would otherwise drop sessions
        # silently: the prefix matches win and the rest are never mentioned.
        ignored = [f for f in entries
                   if f not in strict and prefix.lower() in os.path.splitext(f)[0].lower()]
        if ignored:
            log(f"  WARNING: {len(strict)} file(s) start with '{prefix}_' and are "
                f"being used, so {len(ignored)} other file(s) mentioning "
                f"'{prefix}' are being IGNORED:")
            for f in sorted(ignored):
                log(f"      {f}")
            log(f"  use one naming convention for the whole folder, otherwise "
                f"these sessions are left out of the results.")
        return [os.path.join(folder, f) for f in sorted(strict)]

    # Other passages present in this folder, so a file mentioning two of them
    # can be recognised as ambiguous instead of being assigned arbitrarily.
    others = set()
    try:
        for f in os.listdir(folder):
            m = re.match(r"^(.+)_phonemes(?:-\d+)?\.txt$", f)
            if m and m.group(1) != prefix:
                others.add(m.group(1).lower())
    except OSError:
        pass

    loose = []
    for f in entries:
        stem = os.path.splitext(f)[0].lower()
        if prefix.lower() not in stem:
            continue
        clash = sorted(o for o in others if o in stem)
        if clash:
            log(f"  skipping {f}: names both '{prefix}' and "
                f"'{clash[0]}', so which passage it belongs to is ambiguous. "
                f"Rename it to start with the passage it is.")
            continue
        loose.append(f)

    if loose:
        log(f"  matched {len(loose)} session file(s) by passage name rather than "
            f"by prefix")
    return [os.path.join(folder, f) for f in sorted(loose)]


def dedupe_sessions(tsvs, prefer_edited: bool = True) -> list[str]:
    """
    One TSV per recording.

    A recording reviewed in the editor produces "<stem>_edited.tsv" while the
    batch run produces "<stem>_auto.tsv". With both present the same recording
    would otherwise appear twice in the output, once reviewed and once not.
    """
    if not prefer_edited:
        return sorted(tsvs)
    by_stem: dict[str, list[str]] = {}
    for t in tsvs:
        stem = os.path.splitext(os.path.basename(t))[0]
        for suffix in ("_edited", "_auto"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        by_stem.setdefault(stem, []).append(t)
    out: list[str] = []
    for stem, group in by_stem.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        edited = [g for g in group if os.path.basename(g).rsplit(".", 1)[0].endswith("_edited")]
        out.append(sorted(edited)[0] if edited else sorted(group)[0])
    return sorted(out)


def discover_passage(prefix: str, folder: str, output_suffix: str,
                     prefer_edited: bool = True,
                     log: Callable[[str], None] = lambda *a: None):
    text = os.path.join(folder, f"{prefix}.txt")
    phon = os.path.join(folder, f"{prefix}_phonemes.txt")

    if not os.path.exists(text) or not os.path.exists(phon):
        # No bare pair. Bootstrap from a segmented pair instead, e.g.
        # "caterpillar-01.txt" with "caterpillar_phonemes-01.txt". That pair
        # only seeds the word and canonical-phoneme data; each session still
        # picks the segment matching its own "-0N" suffix.
        pairs = _find_segment_pairs(prefix, folder)
        if not pairs:
            if not os.path.exists(text):
                raise FileNotFoundError(text)
            raise FileNotFoundError(phon)
        first = sorted(pairs, key=lambda d: int(d))[0]
        text, phon = pairs[first]

    tsvs = find_session_files(prefix, folder, log)
    tsvs = dedupe_sessions(tsvs, prefer_edited)
    out = os.path.join(folder, f"{prefix}{output_suffix}")
    return (prefix, text, phon, tsvs, out)


def resolve_jobs(input_dirs, passages=None, output_suffix="_all_aligned.txt",
                 manual_passages=None, prefer_edited=True,
                 log: Callable[[str], None] = lambda *a: None):
    """Each job is (name, text_path, phonemes_path, [tsv_paths], out_path)."""
    if manual_passages:
        jobs = []
        for spec in manual_passages:
            name = spec["name"]
            out = spec.get("out", f"{name}{output_suffix}")
            jobs.append((name, spec["text"], spec["phonemes"],
                         list(spec["tsvs"]), out))
        return jobs

    if passages:
        jobs = []
        for folder in input_dirs:
            for prefix in passages:
                try:
                    jobs.append(discover_passage(prefix, folder, output_suffix,
                                             prefer_edited, log))
                except FileNotFoundError:
                    pass          # this prefix is simply not in this folder
        return jobs

    jobs = []
    bare_re = re.compile(r"^(.+)_phonemes\.txt$")
    seg_re = re.compile(r"^(.+)_phonemes-\d+\.txt$")
    for folder in input_dirs:
        prefixes: set[str] = set()
        try:
            entries = os.listdir(folder)
        except OSError:
            entries = []
        for fn in entries:
            m = bare_re.match(fn) or seg_re.match(fn)
            if m:
                prefixes.add(m.group(1))
        for prefix in sorted(prefixes):
            try:
                jobs.append(discover_passage(prefix, folder, output_suffix,
                                             prefer_edited, log))
            except FileNotFoundError:
                pass
    return jobs


# =============================================================================
# Settings-driven entry point
# =============================================================================


def run_alignment(cfg: dict[str, Any], log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    input_dirs = [str(d).strip() for d in cfg.get("INPUT_DIRS", []) if str(d).strip()]
    if not input_dirs:
        raise ValueError("no passage folders selected")

    raw_passages = str(cfg.get("PASSAGES", "") or "").strip()
    passages = [p.strip() for p in re.split(r"[,\n;]", raw_passages) if p.strip()] or None

    manual = cfg.get("MANUAL_PASSAGES") or None
    if isinstance(manual, dict):
        manual = [manual]

    output_suffix = cfg.get("OUTPUT_SUFFIX", "_all_aligned.txt") or "_all_aligned.txt"
    prefer_edited = bool(cfg.get("PREFER_EDITED_TSV", True))

    jobs = resolve_jobs(input_dirs, passages, output_suffix, manual,
                        prefer_edited, log)
    if not jobs:
        raise ValueError(
            "no passages found. Each folder needs <passage>.txt with the reference "
            "text and <passage>_phonemes.txt with the canonical phoneme sequence, or "
            "a numbered set such as caterpillar-01.txt with "
            "caterpillar_phonemes-01.txt. Session files are <passage>_*.tsv.")

    g2p = G2P(cfg.get("G2P_BACKEND", "builtin"),
              cfg.get("PHONEMIZER_LANGUAGE", "en-us"), log)
    user_dicts_all = cfg.get("USER_DICTS") or {}
    pause_threshold = float(cfg.get("PAUSE_THRESHOLD", 0.7))
    verify = bool(cfg.get("VERIFY", True))

    reports = []
    for name, text, phon, tsvs, out in jobs:
        log(f"Passage: {name}  ({os.path.dirname(text) or '.'})")
        raw = user_dicts_all.get(name, {})
        user_dict = {k.lower(): (v.split() if isinstance(v, str) else list(v))
                     for k, v in raw.items()}
        reports.append(align_passage(name, text, phon, tsvs, out, pause_threshold,
                                     verify, user_dict, g2p, log))
        log("")

    total = sum(len(r["sessions"]) for r in reports)
    written = [r["output"] for r in reports if r["output"]]
    if total == 0:
        # Discovering the passages but aligning nothing is not success, and
        # reporting it as "[done]" is what makes people look for a missing file.
        raise ValueError(
            f"found {len(reports)} passage(s) but no session files, so nothing "
            f"was written. A session file must be a .tsv sitting in the same "
            f"folder as <passage>.txt, with the passage name somewhere in its "
            f"filename. See the log above for what each passage looked for and "
            f"what was actually in the folder.")
    log(f"[done] {len(reports)} passage(s), {total} session(s) aligned, "
        f"{len(written)} file(s) written")
    for out in written:
        log(f"       {out}")
    return reports