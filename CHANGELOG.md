# Changelog — acoustic analysis module

This is the original changelog from the head of `praat_v10.py`, moved out of the
source file so the module starts with code. Nothing in it has been edited.

```text
Complete Speech Analysis Tool for Neurostimulation Studies
Comprehensive analysis matching and extending Praat's capabilities.

-----------------------------------------------------------------------------
v20 changelog - MERGE OF THE TWO INDEPENDENT v19 BRANCHES.

FILE NAMING: this file ships as praat_v11_merged.py to follow the external
naming of its inputs (praat_v9.py, praat_v10_chatty.py, praat_v10_claudio.py).
The INTERNAL changelog numbering is unbroken, so the merge is v20: both inputs
were independently numbered v19 on top of the same v18 base.

PROVENANCE. Both branches started from v18 (praat_v9.py) and both called
themselves v19, but they are not two attempts at the same list. A structural
diff gives 40 hunks each with only TWO overlapping regions, so this is a union
and not a choice between them. Every item below is tagged with where it came
from:

  [C]  from the v10_chatty branch    - measurement integrity and missing-value
                                       handling
  [K]  from the v10_claudio branch   - measures that were being taken on the
                                       wrong signal
  [B]  present in both               - reconciled, see item 0
  [N]  new here                      - bugs found while merging, in the two
                                       branches themselves

The base of this file is the claudio branch, because its changes are structural
(new functions, new columns, CANONICAL_METRICS 102 -> 103); the chatty changes
are localised and were ported onto it as 44 individually verified patches.

-----------------------------------------------------------------------------
 0. THE TWO OVERLAPS [B]. Only two regions were touched by both branches:

      SpectralMetrics NaN defaults  - identical intent, claudio's version kept
                                      (it also documents why cpp_db is not CPP).
      task_from_filename()          - both noticed that plain substring matching
                                      made "house" match "greenhouse". Claudio's
                                      version is kept: it not only matches at
                                      word boundaries but fixes the PRECEDENCE,
                                      asking the file stem before the parent
                                      folder and ranking an explicit task token
                                      ("sust", "mpt", "read") above a mere
                                      stimulus name from KNOWN_PASSAGE_SYLLABLES.
                                      Chatty's version fixed the matching but
                                      kept the old order, under which every file
                                      inside a folder called "Readings" was a
                                      reading file - "/study/Readings/P03_sust_a.wav"
                                      included.

-----------------------------------------------------------------------------
 1. ONE PRODUCTION PER FILE IS NOW THE DEFAULT [C]. The recording protocol is
    exactly one reading passage OR one sustained phonation per file, so there is
    nothing to segment into trials and no edit point worth splitting on: an
    "edit point" found inside a single take is a false positive that shortens
    MPT. single_token_per_file now defaults to True and split_at_splices to
    False, in the analyzer and in analyze_files(). An internal dropout stays
    inside the one production and is reported by the coverage ledger, the break
    ledger and sv_mpt_longest_uninterrupted_s instead of becoming a new token.
      - boundaries_source is labelled "single_production_per_file";
      - the token-count cross-check no longer expects "breath gaps + 1" under
        this mode (an internal breath is EXPECTED and is not a segmentation
        error), and never second-guesses user-supplied boundaries;
      - one_production_per_file is recorded in the audit block.

 2. SEMITONE SD WAS NOT AN SD [C]. std_semitones was
    12*log2((mean+sd_hz)/mean) - the semitone distance from the mean to one
    Hz-SD above it, i.e. a one-sided transform of the Hz SD. Because the
    Hz->semitone map is logarithmic the two differ, and the discrepancy grows
    with F0 variability, so the error was largest on exactly the unstable
    voices the measure exists to describe. Now the contour is converted to
    semitones first and the SD taken of that.
    DEVIATION FROM BOTH BRANCHES [N]: chatty also redefined mean_semitones
    relative to the median, which makes it ~0 by construction. It is kept here
    as a pitch LEVEL re 100 Hz (the v18 definition), which carries information.
    Neither field is exported to the CSV, so this affects no column.

 3. JITTER, SHIMMER, HNR AND CPPS FAIL INDEPENDENTLY [C]. In v18 an
    unmeasurable jitter (local) triggered `continue`, discarding the shimmer,
    HNR and autocorrelation that Praat had measured perfectly well on the same
    interval - biasing every other column towards the intervals where jitter
    happened to succeed. The interval is now kept if ANY metric survived.
    Related, in the sustained window loop: CPPS is computed FIRST and a window
    is dropped only when nothing at all could be measured on it. CPPS needs no
    pulse train, so a periodicity failure says nothing about whether the
    cepstral peak is trustworthy; the old order discarded the cepstral measures
    of precisely the roughest voices, i.e. selected on a correlate of the
    outcome. (This extends the v17 tiered-gate reasoning into the loop itself.)

 4. NO FAKE ZEROS [C]. IntensityMetrics, PitchMetrics, VoiceQualityMetrics,
    FormantMetrics and RecordingQualityMetrics defaulted their float fields to
    0.0, so a failed Praat call was indistinguishable from a measurement and
    entered the medians and the plots. All 68 such fields now default to NaN.
    (claudio had already done this for SpectralMetrics.) Also
    VoiceQualityMetrics.reliable and FormantMetrics.track_ok now default to
    False: both are assertions about a measurement that has not been taken on a
    default-constructed object, and track_ok is set explicitly to True on the
    success path, so only the early-exit and exception paths keep the default -
    where the track was precisely not ok.
    VoiceQualityMetrics.measured no longer hardcodes True; it reflects whether
    any of jitter/shimmer/HNR is finite.

 5. NaN MEANS MISSING, IN THE PLOTS AND IN THE CSVs [C].
      - the bar plots ran np.nan_to_num(values, nan=0.0), which DREW a missing
        value as a real zero-height bar. Missing slots are now empty (4 sites);
      - the CSV writers tested `isinstance(v, float) and np.isnan(v)`. That is
        False for np.float64 and every other NumPy scalar, so a NumPy NaN got
        past the test and was written out as the literal string "nan". Now
        `not np.isfinite(float(v))` (3 sites).

 6. THE COMPOSITE FLUENCY INDEX IS OFF BY DEFAULT [C]. overall_fluency is an
    unvalidated weighted sum of eight sub-scores with hand-chosen coefficients,
    exported next to real measurements and labelled with a "clinical_severity"
    string. Nothing here validates those weights against any outcome, and a
    number that looks like a severity rating gets read as one. Retained for
    continuity behind enable_fluency_index=False.

 7. THE USABILITY GATE IS REACHABLE ON SHORT TAKES [C]. Defaults are now
    2 valid windows / 4.0 s measured over 2.0 s windows (were 3 / 6.0 s over
    3.0 s). The old geometry demanded ~6.5 s of flawless phonation before
    anything about the voice was reported, which discarded usable takes from
    patients who cannot sustain longer. The gate also now screens SHIMMER,
    which was checked on neither count: a file could pass perturbation_usable
    with no shimmer at all, or with a window-to-window IQR exceeding its own
    median.

 8. NEGATIVE HARMONICITY IS A VALUE, NOT A FAILURE [N]. Both HNR frame loops
    filtered on `hv is not None and not np.isnan(hv) and hv > 0`, discarding
    every negative harmonicity - i.e. exactly the breathy and aperiodic frames
    that carry the clinical signal - and biasing the median upwards on the
    worst voices. Now `np.isfinite(hv)`. Neither branch caught this.

 9. TWO BUGS IN THE v10_chatty BRANCH ITSELF [N], not carried over:

      a) `Get jitter (rap)` was called with `mp` instead of `mpf` - an
         undefined name. The call sits inside q(lambda: ...), which swallows
         every exception and returns NaN, so jitter_rap_percent was silently
         NaN on every file, in every session, with nothing in the console or
         the audit block to say so.
      b) AUDIT_COLUMNS lost the comma after "cpps_is_praat", so Python's
         implicit string concatenation merged two entries into a single
         phantom column "cpps_is_praatsyllable_source_known". Confirmed by
         ast.literal_eval: 58 quoted strings, 57 actual list elements.
         cpps_is_praat and syllable_source_known were both destroyed.

10. ONE CHATTY CHANGE DELIBERATELY REJECTED [N]. That branch assigned
    mpt_longest_s = uninterrupted_s and mpt_mean_s = max(durs). This conflates
    two distinct metrics and destroys the invariant that the claudio branch
    establishes in item 1 of its own changelog - that the unbroken duration can
    never exceed the token it was measured in, because a value that exceeds it
    is a segmentation failure rather than a measurement. mpt_longest_s keeps
    its meaning (longest token) and mpt_mean_s stays the mean; under
    one-production-per-file there is one token, so the three coincide by
    construction and nothing is lost.

11. WARNINGS ARE NO LONGER BLANKET-SILENCED [C]. warnings.filterwarnings
    ('ignore') hid the librosa/NumPy warnings (empty slice, all-NaN axis,
    divide-by-zero) that are often the first sign a file is degenerate.
    Replaced with simplefilter("once"), which keeps a large batch readable
    without discarding the QC signal.

-----------------------------------------------------------------------------
EVERYTHING FROM THE claudio BRANCH IS PRESENT UNCHANGED [K]: the rewritten
longest_unbroken_phonation() and its break ledger, the CPPS scale and labelling
fix for cpp_db, the fixed 50-5000 Hz spectral-tilt band, the restriction of the
spectral and formant blocks to token spans and valid windows on sustained
files, intensity decay on the longest token plus its per-second twin, the
internally consistent exported pitch block, and the filename-routing
precedence. See the v19 changelog below for the reasoning on each.

CANONICAL_METRICS is 103 (unchanged from the claudio branch).
AUDIT_COLUMNS is 73 (was 72 there): + one_production_per_file.

VERIFICATION PERFORMED. The merge was applied as 44 scripted patches, each
asserting its own occurrence count, so a drifted anchor fails the build instead
of producing a half-merged file. The result compiles, imports cleanly with
stubbed audio libraries (so the module-level asserts actually run), and has:
103 unique canonical metrics all produced by _raw_metric_table, 73 unique audit
columns all produced by get_audit_table, no overlap between the two sets, and
no over-long name of the kind a missing comma produces.

NOT VERIFIED: no run on real audio. librosa, parselmouth and soundfile were
unavailable in the build environment. Before using this on the study batch,
re-run two or three files already analysed under v9 and diff the columns.
pitch_std_semitones, jitter_rap_percent, hnr_db, cpp_db, spectral_tilt_db and
sv_mpt_longest_uninterrupted_s are EXPECTED to change - if they do not, the
merge did not take effect.

-----------------------------------------------------------------------------
v19 changelog - THE UNBROKEN-PHONATION METRIC, and four measures that were
being taken on the wrong signal.

 1. sv_mpt_longest_uninterrupted_s WAS NOT MEASURING WHAT IT CLAIMED.
    "The longest phonation with no break in the middle" was computed by
    longest_uninterrupted_sound(), which was wrong three times over:
      - it searched the span from the FIRST token's start to the LAST token's
        end, so two productions separated by a breath that never fell below the
        silence line came back as ONE unbroken sound. The value could therefore
        exceed sv_mpt_longest_s, which is not a measurement but a segmentation
        failure, and on a multi-trial file it grew without limit;
      - it was ENERGY-ONLY: any frame above the threshold counted, so a breathy
        unvoiced exhale, a whispered tail or room noise at phonation level
        extended the run. Phonation requires fold vibration;
      - it used the global 2048-sample envelope (46 ms at 44.1 kHz), so a break
        SHORTER THAN ONE FRAME was averaged away with the phonation around it.
    Replaced by longest_unbroken_phonation(): measured inside a single token,
    clamped to that token, voicing required, splices treated as hard breaks,
    on its own 20 ms / 5 ms envelope with the threshold referenced to the LOCAL
    in-token phonation level. A BREAK is now defined explicitly - >= 50 ms
    below threshold or unvoiced (phonation_break_min_s / _drop_db) - because
    "no break in the middle" is not self-defining, and shorter dropouts must be
    bridged or the metric becomes a measure of tracker noise.
    New audit columns say what the rest of the take looked like, since the same
    unbroken duration with six breaks is a different finding from one with none:
      sv_n_phonation_breaks, sv_n_breaks_in_best_token, sv_unbroken_start_s,
      sv_unbroken_end_s, sv_unbroken_token_index, sv_break_threshold_db,
      sv_break_min_duration_s, sv_unbroken_voicing_required
      sv_unbroken_energy_only_s - the same measure with voicing IGNORED. A large
      gap between the two is APHONIC BREAKING: the airflow does not stop but the
      voice does, which is a finding in its own right and is now flagged on the
      console instead of silently shortening the headline number.

 2. cpp_db WAS NEVER CPP. analyze_spectral() calls PowerCepstrogram "Get CPPS",
    i.e. a SMOOTHED cepstral peak prominence, and stored it as cpp_db against
    thresholds (>8 "clear", >5 "good", >3 "moderate") that belong to the
    unsmoothed CPP. On the CPPS scale a healthy sustained vowel sits near
    13-20 dB, so "clear voice quality" printed for essentially every file,
    severely dysphonic ones included, and "breathy or rough" was unreachable.
    Worse, a Praat failure fell through to an FFT peak-to-mean cepstral ratio
    and a failure of THAT wrote 0.0 - three incommensurable quantities in one
    column with no way to tell them apart. Thresholds corrected and split per
    task, the fallback is labelled (cpp_source, cpp_is_praat_cpps in the audit
    block) and a total failure is NaN. The column name is kept for continuity
    but it is a SECOND CPPS, computed with different settings and over a
    different span than cpps_db - do not treat the two as independent.

 3. spectral_tilt_db_per_khz DEPENDED ON THE SAMPLE RATE. The regression ran
    over the whole linear axis, 0 Hz to Nyquist, so the same voice gave one
    tilt at 16 kHz and another at 48 kHz: the extra octave is almost pure noise
    floor and drags the slope down. v17 item 5 already warns that a
    recording-chain difference is indistinguishable from a treatment effect in
    every spectral measure - this was one such difference built into the metric.
    Now fitted over a fixed 50-5000 Hz band (clipped to Nyquist), reported as
    spectral_tilt_band_lo_hz / _hi_hz.

 4. THE SPECTRAL AND FORMANT BLOCKS RAN ON THE WHOLE SPEECH SPAN FOR BOTH
    TASKS. On a multi-token sustained file that span includes the inter-token
    silence and the breaths, so alpha_ratio_db, hammarberg_index,
    spectral_tilt_db_per_khz and cpp_db were measured partly on signal that is
    not phonation - and those four are exactly what this script tells you to
    fall back on when jitter and shimmer come back empty. F1/F2 came from every
    voiced frame in the span, so sv_f1f2_cloud_spread_hz, sold as ARTICULATORY
    STEADINESS on one held vowel, was inflated by the onset and offset
    transitions of every token.
    analyze() now runs the sustained analysis FIRST and hands its spans down:
    the spectral block gets the token spans, the formant block gets the valid
    steady-state windows - the same signal the perturbation medians come from.
    Connected speech is deliberately unchanged (whole speech span): fricatives
    and stops are part of the signal there, not gaps in it, so reading files
    stay comparable with earlier runs. spectral_span_intervals and
    formant_span_windows record which path was taken.

 5. intensity_decay_db ON A SUSTAINED FILE DESCRIBED THE SESSION, NOT THE VOWEL.
    The console has always called it "decay across the vowel" while the fit ran
    from speech_start to speech_end, so on a three-trial file it measured trial
    3 being quieter than trial 1, with the breaths contributing frames too. Now
    fitted on the LONGEST TOKEN only. The old value is also dB across the
    ANALYSED SPAN, so the same physiological fade rate produces a seven times
    larger number on a 70 s take than on a 10 s one; the duration-free twin is
    added as sv_intensity_decay_db_per_s (and intensity_decay_db_per_s for
    reading). The original column keeps its definition so a fixed-length
    passage stays comparable with previous runs.

 6. THE EXPORTED PITCH BLOCK WAS HALF WINDOWED, HALF WHOLE-SPAN. Only
    pitch_mean_hz was replaced by the window value on a sustained file, while
    the median, SD, min, max, range and CV stayed whole-span - so pitch_cv no
    longer equalled pitch_std_hz / pitch_mean_hz in the row a reader was
    looking at, and pitch_range_semitones spanned every token plus the
    tracker's excursions at the token edges, i.e. it measured the segmentation
    rather than the voice. All seven now switch together, from the same valid
    windows (pitch_block_from_windows in the audit block).
    voiced_percent stays whole-span ON PURPOSE: inside a valid window it would
    be ~100% by construction and would carry no information.

 7. FILENAME ROUTING SENT SUSTAINED FILES INTO THE READING PIPELINE.
    Matching was plain substring over parent-path + stem, with passage tokens
    checked before sustained ones, so "house" matched "greenhouse", "text"
    matched "context", and every file under a folder called "Readings" was a
    reading file regardless of its own name - "/study/Readings/P03_sust_a.wav"
    included. task_from_filename() now matches at word boundaries only
    (_token_present), asks the STEM before the FOLDER, and ranks the evidence:
    an explicit task token ("sust", "mpt", "read", "passage") outranks a mere
    STIMULUS name from KNOWN_PASSAGE_SYLLABLES, so "P03_sust_a_daily" is a
    sustained vowel recorded in a session that also used the daily passage.
    Contradictory names return None and let the acoustic classifier decide.

 8. SpectralMetrics NO LONGER DEFAULTS TO 0.0. A 0 dB alpha ratio, 0 dB
    Hammarberg index and 0 dB/kHz tilt are all physically possible, so
    initialising them to zero made a failed Praat call indistinguishable from a
    measurement - the same fake-zero class of bug as v9 item D and v16 item 9.

 9. Housekeeping: dead main_vq / best_len removed from the window loop (v15
    superseded them with panel_vq); the "a sustained file carries 73 and a
    reading file 78" note had not been true since v16 and now reads 82 / 81;
    and the note records that cpps_iqr_db, hnr_iqr_db, jitter_iqr and
    shimmer_iqr are structurally blank on reading files - they are
    window-to-window dispersions and only the sustained path measures in
    windows, so an empty cell there is not a failed measurement.

 CANONICAL_METRICS is 103 (was 102): + sv_intensity_decay_db_per_s.

 KNOWN LIMITATION, UNCHANGED: sv_mpt_longest_uninterrupted_s can only be as
 good as the token boundaries. Supply them via <stem>_tokens.csv when you have
 them - inference is a fallback, not a feature.

-----------------------------------------------------------------------------
v18 changelog - HOW MUCH OF THE VOWEL WAS ACTUALLY USED.

 The window machinery already discarded onsets, offsets and unsteady patches
 and reported the median over the survivors, but the only trace of how much it
 had to throw away was a COUNT of windows (sv_n_windows_valid / sv_window_yield).
 A count cannot answer the clinical question - "did this person hold a steady
 /a/ from beginning to end, or was the take full of unstable patches?" - because
 the same count comes out of a 7 s take and a 25 s one, and because a window is
 lost to plain geometry (edge trim, and a remainder shorter than one window) as
 easily as to a wobbling voice.

 v18 therefore accounts for the phonation in SECONDS, twice over, and adds a
 window-free frame-level steadiness measure.

 1. HOW MUCH SURVIVED (new metrics, all sustained-only):
      sv_analyzed_fraction      seconds measured / seconds phonated. The headline
                                number: 1.00 = every phonated second is behind
                                the medians, 0.35 = two thirds were discarded.
      sv_window_yield_s         seconds measured / seconds the tiling geometry
                                COULD have measured. The duration-weighted twin
                                of sv_window_yield, so a low value means an
                                unsteady voice and not a short take.
      sv_steady_frame_fraction  fraction of phonated frames on which a valid
                                window COULD be centred, evaluated frame by
                                frame. Free of window quantisation, so it does
                                not fall just because the take is short.
      sv_longest_steady_run_s   longest uninterrupted steady stretch, in seconds.
                                A take with 80% steady frames scattered in 300 ms
                                islands is a different finding from one steady
                                block, and only this column separates them.

 2. WHERE THE REST WENT (new audit columns). Two exact decompositions of
    sv_total_phonation_s, so the loss can be attributed instead of guessed:
      by CAUSE      measured + unsteady + quantisation + edge-trim
      by POSITION   measured + onset + interior + offset (+ dead tokens)
    An onset/offset-only loss is normal and expected; interior loss, and a high
    sv_n_interior_gaps, is the signature of a take that kept breaking down.

 3. The console prints the ledger and a one-line verdict per file
    (sv_coverage_note, also written to the CSV), so "the windows were not good
    enough" now comes with the seconds that explain it.

 4. Two smaller fixes in the same area:
      - the "no window could be measured by Praat" exit dropped the rejection
        tally and the splice times, so exactly the file that needed the audit
        trail was the one that had none;
      - vowel_window_hop_s has been dead since v8 (the candidate grid is
        hard-wired to win/2). It is still accepted, and still ignored, but the
        grid step is now settable on purpose via vowel_window_grid_hop_s
        (default None = win/2, i.e. unchanged behaviour). A finer grid, e.g.
        0.5 s, lets windows slide around a short wobble instead of losing the
        whole 2 s slot, which is the cheapest way to raise the yield on takes
        that keep failing the usability gate.

-----------------------------------------------------------------------------
v17 changelog - found by running v16 on a real 20-file DBS sustained-/a/ batch.

 1. THE USABILITY GATE IS NOW TIERED, because the single flag was discarding
    cepstral measures for a PERIODICITY failure. CPPS needs no pulse train, so
    "the jitter estimate is unstable" says nothing about whether the CPPS
    estimate is trustworthy - yet sv_usable=0 removed both from the session
    table. Measured cost on the real batch: in session 20260311 two of three
    takes were dropped for jitter IQR, the survivor happened to have the highest
    CPPS, and the session median moved 16.8 -> 18.9 dB. That is ~45% of the
    entire between-session CPPS range in that dataset, manufactured purely by
    selecting on a correlate of the outcome.
      sv_windows_usable      - enough steady signal to trust ANY window median
      sv_perturbation_usable - jitter/shimmer additionally stable, signal type OK
      sv_usable              - the AND of the two (old, conservative meaning)
    write_session_csv() now aggregates in three tiers (rep_basis records which):
    all / windows_usable / perturbation_usable.

 2. PERTURBATION NO LONGER LEAKS FROM THE WHOLE-FILE FALLBACK. When the
    sustained analysis finds ZERO valid windows, analyze() falls back to the
    whole-file Praat Voice Report. The console printed "Jitter/Shimmer/HNR: NOT
    MEASURABLE" while the CSV quietly received jitter, shimmer, NHR and
    autocorrelation from that fallback - on one real file, a 22%-voiced,
    3.3 dB-HNR signal carried ppq5=1.619%, apq11=6.300%, NHR=0.2148. Those are
    the tracker's variability over the handful of frames it accepted. Cycle-based
    measures are now NaN unless the windowed analysis actually measured them;
    everything periodicity-free (CPPS, CPP, token HNR, spectral shape, voiced
    fraction) is untouched, so an unmeasurable voice stays a finding WITH numbers.

 3. classify_signal_type() NO LONGER FLIPS ON A HUNDREDTH. The rule was
    `vf < 0.55 or hnr < 7 or cpps < 12` -> type 3. A real file with voiced
    fraction 0.54, HNR 17.2 dB and CPPS 13.2 dB was therefore declared
    "perturbation UNDEFINED" - and then measured a perfectly good jitter of
    0.658%, while the next console line said "the THRESHOLD was the limit and the
    periodicity is strong". HNR and CPPS (properties of the harmonic structure)
    still trigger type 3 alone; low voiced fraction triggers it only when the
    harmonic evidence does not contradict it. The voicing probe now runs BEFORE
    classification and is passed in, so a threshold artifact is recognised as one.

 4. THE WINDOW LEDGER BALANCES. Windows discarded by the non-overlapping dedup
    (candidates are tiled at hop = win/2) and by max_windows_measured were
    dropped silently, so the printout read e.g. "6 valid of 11" with no
    rejections listed and 5 windows apparently vanished. They are now counted as
    overlap_dedup / over_cap, and the console states which denominator is which
    (n_windows_total counts overlapping candidates; window_yield does not).

 5. BATCH CHANNEL-HETEROGENEITY CHECK. A recording-chain difference between
    sessions is indistinguishable from a treatment effect in every spectral and
    cepstral measure. On the real batch the session with the best CPPS (19.1 dB)
    was also the only one at ~3000 Hz effective bandwidth against ~1400-1600 Hz
    elsewhere. analyze_files() now reports sample-rate spread, effective-bandwidth
    ratio and spectral-edge spread across the batch, and when bandwidth varies by
    >=1.5x it prints the diagnostic sequence needed to tell a channel artifact
    from a real voice change. It does not decide for you - a clearer voice really
    does push more energy into the upper harmonics - it tells you to check.

 6. The session summary no longer claims to be "only sv_usable reps" while some
    columns were aggregated over ALL takes (which is why a 0/3-usable session
    still printed a pitch value). Every cell now carries its basis and the n
    actually used, e.g. 16.80( 1.27)[win3].

 7. Cosmetic: the "STILL COMPARABLE" hint named sv_voiced_fraction, a column
    renamed to voiced_fraction in v16, so it pointed at something not in the CSV.

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
v16 changelog - METRIC SET RE-SELECTED FOR ROBUSTNESS, plus four bug fixes.

Motivation: jitter and shimmer are not reliably computable. They need a
period-by-period pulse train, so on a rough, breathy or diplophonic voice - the
voices this study is about - Praat returns nothing, and a metric that is missing
on the bad days cannot be used to compare good days with bad ones. That is a
property of the measure, not of the recording.

  1. THE METRIC SET IS 97 AGAIN (v6 parity), BUT RE-CHOSEN.
     v8 had grown to 163 columns by pure addition - it was a strict superset of
     v6, so the "focus on voice quality for sustained vowels" happened only in
     the plot lists, never in the metric set. The union across both tasks is now
     exactly 97 (asserted at import), selected for robustness:
       - CPPS leads the voice-quality block. It is a cepstral property, needs no
         pulse train and no periodicity decision, and is the best-validated
         single correlate of dysphonia severity - so it still produces a number
         on exactly the files where jitter/shimmer come back blank.
       - jitter_ddp and shimmer_dda were DELETED as information-free: in Praat
         DDP == 3 x RAP and DDA == 3 x APQ3, by definition.
       - of the remaining perturbation variants only the most-smoothed survive
         (PPQ5, APQ11) alongside the conventional reporting pair
         (jitter_local, shimmer_local_dB); RAP/APQ3/APQ5 dropped as >0.95
         collinear with those.
       - EVERY perturbation median now ships with its dispersion:
         jitter_iqr, shimmer_iqr, hnr_iqr_db, cpps_iqr_db. This is the column
         that says whether a value is stable enough to compare across days.
       - voiced_fraction promoted to a first-class metric: always computable, so
         an unmeasurable voice becomes a finding instead of an empty row.
     See CANONICAL_METRICS for the full rationale block.

  2. THE sv_ TWINS ARE GONE. On a sustained file v8 assigned the robust sv_
     medians into the generic columns AND kept the sv_ names, so seven pairs
     (jitter_local, shimmer_local, hnr_db, cpps_db, pitch_mean_hz,
     hnr_median_perinterval_db, intensity_decay_db) held identical numbers under
     two names. The sv_ prefix is now reserved for what only a sustained vowel
     HAS: MPT, token structure, steady-state F0, tremor, token-level fallbacks.

  3. PROVENANCE SPLIT OUT OF THE METRIC TABLE (get_audit_table, AUDIT_COLUMNS).
     Settings-actually-used, octave verdict, window-rejection counts, the
     usability gate and recording properties are why v9-v15 can be trusted, but
     they are not measurements of the voice. They now go to their own labelled
     block in the CSV: never plotted, never counted as metrics, never aggregated
     as if they were outcomes. Read that block FIRST when two sessions disagree -
     a change in pitch_floor_used_hz explains a "finding" that is not one.

  4. BUG: extract_date_from_filename() only matched 8 CONSECUTIVE digits, so
     "2024-03-14_..." returned "99999999". That value is the SESSION KEY, so
     every ISO-dated file collapsed into one pseudo-session: different days were
     silently averaged, and the across-repetition SD (the whole point of the
     session table) became an SD across unrelated sessions. Separated forms
     (YYYY-MM-DD, YYYY_MM_DD, DD.MM.YYYY) are now parsed and normalised.

  5. BUG: _f0_wide_median leaked between files. It was initialised in __init__
     only, so if estimate_f0_range() raised, the PREVIOUS file's value was used
     for the subharmonic-lock test and written out as this file's
     f0_wide_median_hz. Reset per file with the rest of the state, along with
     _octave, _f0_subharm_lock, _f0_lock_st and _silence_threshold_used.

  6. BUG: direction guidance was global, so shared metrics were mislabelled on
     one task. On a sustained-vowel file the "pauses" are the gaps BETWEEN
     TRIALS, so pause_count is the trial count minus one - and the plots said
     "fewer pauses = more fluent", i.e. that recording two /a/ tokens instead of
     three was a clinical improvement. _pick_best_on_per_metric() was also
     optimising these. See TASK_DIRECTION_OVERRIDES / metric_direction().

  7. BUG: write_consolidated_csv() called write_long_csv() as a side effect, so
     calling it alone wrote and announced a second file. Decoupled;
     analyze_files() now calls both explicitly.

  8. intensity_cv removed. It had no METRIC_DIRECTION entry (so it printed "no
     directional guidance") and was always blank by construction - a coefficient
     of variation is undefined on a logarithmic dB scale.

  9. BUG (found on a second audit pass): FAKE-ZERO HNR SURVIVED v9's OWN FIX.
     hnr_db and mean_autocorrelation were initialised to 0.0. v9 item D fixed the
     PARSING path (`x or 0.0`) but not the EXCEPTION path: if the "Voice report"
     call itself raised, `except: pass` left hnr_db at 0.0 - and 0.0 is finite, so
     the `if not np.isfinite(hnr_db)` harmonicity fallback never ran and HNR was
     reported as exactly 0.0 dB. It also raised a spurious "Low HNR (0.0 dB)"
     pathology flag. Both now start at NaN, and the fallback's own failure path
     no longer writes 0.0 either.

 10. BUG: NHR was `10**(-hnr/10) if hnr > 0 else inf`, wrong twice. A MISSING HNR
     (NaN) failed the `> 0` test and became +inf, which then propagated into
     medians, means and bar plots as if measured; and an HNR of exactly 0 dB or
     negative - physically real for a very noisy voice - also became +inf instead
     of 1.0 / >1.0. Now `10**(-hnr/10)` at any sign, NaN only when HNR is missing.

 11. BUG: the formant plausibility check blanked ONLY F3/F4, whatever had failed.
     When the implausible formant was F1 or F2, the bad values were still returned
     as measurements AND still fed vowel_space_area, vowel_dispersion_logarea,
     vowel_cloud_spread_hz and sv_f1f2_cloud_spread_hz - the articulation outcomes
     actually compared between sessions - with only formant_track_ok=0 to hint at
     it. Failures are now tiered: a low-tier (F1/F2) failure invalidates the F1-F2
     plane and everything derived from it.

 12. "Voice quality within normal limits" could be printed when the measure that
     would have raised the flag was never obtained (NaN fails every threshold
     test silently). Unmeasurable is not normal; the assessment now names what
     could not be measured.

 13. KNOWN_PASSAGE_SYLLABLES NOW SELF-CHECKS. The constants are hand-entered and
     cannot be validated from the audio, and a wrong one rescales speech_rate and
     articulation_rate for every file of that passage by the SAME factor - so the
     error is invisible within-subject and only shows against published norms. The
     envelope estimate is now computed even when the passage is known and compared
     against the constant: a ratio outside 0.70-1.40 prints a warning and is
     exported as syllable_count_estimated / syllable_count_agreement in the AUDIT
     block. Too noisy to replace the constant, quite good enough to catch a typo
     or a factor-of-two error.

  STILL YOUR JOB, NOT THE SCRIPT'S: verify KNOWN_PASSAGE_SYLLABLES against your
  exact passage wording. A wrong constant biases the rate of every file of that
  passage. speaking_time_fraction is independent of it and stays valid.
-----------------------------------------------------------------------------
v9 changelog - correctness fixes found while auditing a real DBS dataset in
which the SAME subject was reported at ~95 Hz in some sessions and ~190 Hz in
others (an exact 2:1 split, with identical formants - i.e. an octave tracking
error, not a physiological change):

  A. OCTAVE VERIFICATION FROM THE HARMONIC COMB (root-cause fix).
     estimate_f0_range() used to anchor the analysis range to the median of a
     wide first pass (med/1.7 .. med*1.7). If that median was the SUBHARMONIC
     (common in rough/diplophonic dysarthric voices), the range excluded the
     true F0 for good and repair_octave_jumps() then "repaired" every frame
     TOWARDS the wrong octave, making the error internally consistent and
     invisible. v9 verifies the candidate against the harmonic comb of the
     signal itself: if the odd harmonics of the candidate are missing, the
     candidate is a subharmonic and the range is doubled (and vice versa).
     Reported in the CSV as f0_octave_* columns; strong disagreement is
     auto-corrected, marginal disagreement is flagged for manual review.

  B. PER-SUBJECT FIXED PITCH RANGE. pitch_range=(floor, ceiling) freezes the
     analysis range across all sessions of one subject. Previously floor/ceiling
     were re-estimated per file (56-163 Hz .. 113-326 Hz on the same subject),
     which also changed the jitter/shimmer period bounds and the
     PowerCepstrogram window - so part of any PRE/POST difference was a
     settings difference, not a voice difference. USE THIS for longitudinal
     designs.

  C. TREMOR EXTENT WAS WRONG. It was 2*sqrt(2)*SD of the whole detrended F0
     contour, i.e. all F0 instability (creak, jumps, drift residual), which is
     why an unstable file reported "tremor 5.89 st". It is now measured on the
     BAND-LIMITED 2-12 Hz component only, and both rate and extent are NaN
     unless the band peak is prominent against the rest of the band (before,
     argmax always returned some number).

  D. NO MORE FAKE ZEROS. agg() returned 0.0 when every Praat call had failed,
     so an unmeasurable jitter/shimmer entered the CSV and the plots as a
     perfect 0.0. It now returns NaN. Aggregation is the MEDIAN (was: a
     one-sided top-decile trim, which biased every value downwards by an amount
     that depended on the number of intervals). HNR of exactly 0 / negative is
     no longer swallowed by `or 0.0`.

  E. SUSTAINED FILES NO LONGER REPORT TWO DIFFERENT JITTERS. jitter_local_percent
     (single longest window) and sv_jitter_local_percent (median over all valid
     windows) coexisted in the CSV under near-identical names. On a sustained
     file the generic columns are now the robust sv_ medians.

  F. USABILITY GATE. sv_usable is 0 when the summary rests on too little signal
     (min_valid_windows / min_measured_s), or when window-to-window spread
     exceeds the value itself. One file in the audited set was summarised from
     2 valid windows out of 11 and looked exactly like the ones based on 16.

  G. SPLICE OVER-DETECTION. The "trimmed silence" cue fired on any 15 dB dip
     below the FILE PEAK, so amplitude dips inside a tremulous vowel were read
     as edit points: three files were split into 5 tokens instead of 3, which
     understated MPT by ~5 s. The cue is now referenced to the LOCAL phonation
     level (drop_db, default 25 dB) and requires min_gap_s of consecutive
     below-threshold frames; the F0-plateau cue defaults to 1.5 st.

  H. TOKEN BOUNDARIES CAN BE SUPPLIED. boundaries_dir/boundaries_suffix reads a
     per-file CSV (onset,offset in seconds) or Praat TextGrid interval bounds.
     When you cut the tokens yourself, inference is pointless - pass the truth.

  I. AUDIT TRAIL IN THE CSV. pitch_floor_used, pitch_ceiling_used,
     silence_threshold_db_used, snr_reference (none/leading-trailing/internal),
     rec_digital_silence, sv_rej_* counts, sv_token_count_mismatch,
     sv_measured_total_s, f0_octave_*. Plus a tidy/long CSV
     (<prefix>_long.csv) for mixed models in R/Python.

  J. FORMANT PLAUSIBILITY. "F3 > F2 + 250 Hz" passed an obvious failure
     (F3=1686 Hz on /a/). Absolute per-formant plausibility bounds added.

  K. ON AGGREGATION. "best ON" took the best value of each metric
     independently, i.e. a column that no single recording ever produced. The
     default is now on_aggregation="all" (one column per ON recording);
     "best"/"median" remain available and are labelled as derived.

  L. PointProcess is computed once per file instead of once per analysis window
     (~20x fewer Praat calls on a 70 s multi-token file).
-----------------------------------------------------------------------------
v8 (TASK-AWARE) changelog - fixes the systematic false positives that appeared
when sustained-vowel recordings were run through a connected-speech pipeline:

  1. TASK ROUTING. Every file is classified as `sustained_vowel` or `reading`
     (folder/filename tokens first, then an acoustic classifier: voicing
     fraction, 3-8 Hz envelope modulation, spectral-centroid stability). Pass
     task="sustained_vowel" or task="reading" to analyze_files() to force it.
     Metrics that cannot apply to the task are emitted as NaN, so they are
     skipped in the plots and left blank in the CSV.

  2. SNR IS NO LONGER MEASURED AGAINST THE VOICE. The old noise floor was the
     median of the quietest 15% of frames of the whole file; with phonation
     running edge to edge that floor IS the vowel, so studio recordings scored
     5-10 dB and were flagged "noisy", which in turn flagged jitter/shimmer/HNR
     UNRELIABLE. The floor now comes from a genuine silent reference (frames
     outside the speech span, else below-threshold frames inside it); with no
     silent reference, SNR is reported as NOT ESTIMABLE and penalises nothing.

  3. BAND-LIMIT DETECTION TIGHTENED. The old test (edge <= 4200 Hz, 15 dB cliff,
     <0.5% energy above) fires on any vowel, which has no consonant energy above
     ~3.5 kHz. It now also requires the high band to be at the numerical floor
     (< -60 dB) and a 25 dB cliff - the signature of a real codec cut.

  4. OCTAVE-ERROR REPAIR + NARROW F0 RANGE FOR VOWELS. Frames an exact octave
     from the median are corrected before any statistic is taken, and for
     sustained vowels the search range is anchored to the median (median/1.7 to
     median*1.7). This removes the "median 101 Hz but mean 140 Hz, CV 0.354"
     pattern that also inflated jitter and shimmer.

  5. PERTURBATION FROM STEADY STATE, NOT FROM THE WHOLE FILE. A multi-trial
     sustained-vowel file is segmented into individual tokens; MPT is per token,
     and jitter/shimmer/HNR/CPPS are measured inside the mid-window of each
     token (onset/offset trimmed), never averaged across breath resets.

  6. NEW SUSTAINED-VOWEL METRICS: MPT (longest/mean), token count, steady-state
     F0 SD in semitones, F0 drift, 2-12 Hz frequency tremor rate and extent,
     intensity SD in the window, and across-token SD of jitter/shimmer/HNR as a
     within-file reproducibility check.

  7. FLUENCY SCORE SUPPRESSED FOR SUSTAINED VOWELS. Its inputs (pauses, fillers,
     rate, rhythm) are undefined there; it was reporting "Moderate-severe
     impairment" purely from artifacts.

  8. FORMANT SANITY CHECK. F3 must exceed F2 by 250 Hz; otherwise the LPC track
     has failed and F3/F4 are set to NaN with a warning instead of being
     reported as measurements.

  9. SILENCE THRESHOLD for sustained vowels is placed 15 dB below the phonation
     level instead of 6 dB above a vowel-contaminated "noise floor", so vibrato
     and taper are no longer counted as pauses.

 10. Removed the always-NaN intensity_cv metric; added task, task_source,
     rec_snr_estimable, formant_track_ok and pitch_octave_repair_fraction to the
     CSV so every gating decision is auditable.
-----------------------------------------------------------------------------

This version produces, instead of a TXT report:
  - One bar plot per metric (X axis = files, Y axis = metric value), saved as PNG
  - One consolidated CSV file with all metrics for all files

-----------------------------------------------------------------------------
v8 changelog (READING-FOCUSED additions for DBS / post-stroke dysarthria):
  1. SYLLABLE-RATE BUG FIXED. The old estimate_syllables clamped the count to
     [1*duration, 8*duration], coupling rate to duration and able to MASK
     bradylalia (a slow reader's count was pulled up toward the 1*duration
     floor). Reading passages are KNOWN, so we now use a reference syllable
     count per passage (Caterpillar=262, etc.) identified from the filename,
     giving EXACT speech/articulation rates. When the passage is unknown we fall
     back to an UN-CLAMPED peak-picking estimate. A 'syllable_source' flag marks
     which path was used (known_passage | estimated).
  2. NEW connected-speech metrics sensitive to hypokinetic dysarthria, all
     computable from reading without phoneme/vowel segmentation:
        - speaking_time_fraction (articulation_time / total_speech_time, 0-1):
          separates true articulatory slowing from PAUSING.
        - Envelope Modulation Spectrum energy in the 3-8 Hz syllabic band (EMS),
          plus peak modulation frequency and a syllabic/slow ratio.
        - Segmentation-free vowel-space dispersion (log-area of the F1-F2
          covariance ellipse) as a centralization proxy, more robust than
          convex-hull VSA on connected speech.
        - Intensity decay across the passage (hypokinetic fading).
        - Per-interval MEDIAN HNR (robust complement to the Voice Report HNR).
        - Hardened CPPS (Praat PowerCepstrogram first; clearly-labeled FFT
          fallback only if Praat fails).
  All new metrics flow automatically into the plots, the overview grid, the
  consolidated CSV, and the per-stim aggregation, via get_metric_table() and
  METRIC_DIRECTION.

  (v6/v7 changes retained: active-speech intensity, spectral-cliff band-limit
   detection, adaptive F0 ceiling, graded voice-quality scoring, directional
   plot guidance, per-stim OFF/ON aggregation.)

NOTE: jitter/shimmer/HNR/voice-break/CPPS extraction depends on parselmouth
(Praat). Validate on real audio in an environment where parselmouth + librosa
are installed.
-----------------------------------------------------------------------------

Requirements:
    pip install numpy scipy librosa soundfile matplotlib parselmouth praat-parselmouth

Usage:
    python speech_analysis_plots.py
    Modify the WAV_FILES and OUTPUT_FOLDER variables at the bottom of the script.
```
