# Metrics

This is the full inventory of what speechkit measures, stage by stage. The
acoustic lists are copied from `speechkit/acoustics.py` (`_M_*` lists,
`CANONICAL_METRICS`, `AUDIT_COLUMNS`), the alignment scores from
`score_block()` in `speechkit/alignment.py`, and the DDK columns from
`master_rows()` and `summary_rows()` in `speechkit/ddk.py`. If a count here disagrees with the
code, the code is right and this file needs updating.

## Summary

| Stage | Unit | Count |
|---|---|---|
| Acoustic analysis | one recording | 103 metrics (60 shared + 21 reading-only + 22 sustained-only), plus 73 audit columns |
| Phoneme recognition | one recording | 4 header values, 5 fields per segment |
| Alignment | one session | 8 scores |
| DDK analysis | one trial | 12 core measures (4 measures x 3 analyses), 22 measurement columns in all; per session, 6 means for each cue plus AMR |

## Acoustic analysis

The recording type decides which block a file fills. Every file gets the 60
shared metrics plus the block for its own task, so a reading file carries
81 metrics and a sustained-vowel file 82. The other task's block is left blank
on purpose: an empty cell there means "does not apply", not "failed".

`acoustics.py` asserts at import that the set is exactly 103 and that no name
is repeated, so a change to the lists that is not reflected in that assert
stops the module from loading.


### Shared by both tasks (60)

| Group | n | Metrics |
|---|---|---|
| Durations | 4 | `duration_total_s`, `duration_speech_s`, `articulation_time_s`, `phonation_time_s` |
| Intensity | 5 | `intensity_active_mean_db`, `intensity_active_std_db`, `intensity_nucleus_std_db`, `intensity_range_db`, `intensity_decay_db` |
| Pitch | 10 | `pitch_mean_hz`, `pitch_median_hz`, `pitch_std_hz`, `pitch_min_hz`, `pitch_max_hz`, `pitch_range_hz`, `pitch_range_semitones`, `pitch_cv`, `voiced_percent`, `unvoiced_percent` |
| Voice quality, periodicity-free or robust | 10 | `cpps_db`, `cpps_iqr_db`\*, `cpp_db`, `hnr_db`, `hnr_iqr_db`\*, `hnr_median_perinterval_db`, `nhr`, `mean_autocorrelation`, `voice_breaks_count`, `voice_breaks_degree_percent` |
| Voice quality, cycle-to-cycle perturbation | 7 | `jitter_local_percent`, `jitter_ppq5_percent`, `jitter_iqr`\*, `shimmer_local_db`, `shimmer_apq11_percent`, `shimmer_iqr`\*, `num_pulses` |
| Spectral shape | 7 | `spectral_centroid_hz`, `spectral_spread_hz`, `spectral_skewness`, `spectral_kurtosis`, `spectral_tilt_db_per_khz`, `alpha_ratio_db`, `hammarberg_index` |
| Formants | 6 | `f1_mean_hz`, `f1_std_hz`, `f2_mean_hz`, `f2_std_hz`, `f3_mean_hz`, `f3_std_hz` |
| Pitch-tracking stability | 3 | `voiced_fraction`, `f0_subharmonic_lock`, `f0_lock_semitones` |
| Pauses | 5 | `pause_count`, `pauses_per_minute`, `pause_total_duration_s`, `pause_mean_duration_s`, `pause_ratio_to_speech` |
| Recording quality | 3 | `rec_quality_score`, `rec_snr_db`, `rec_clipping_fraction` |

\* Window-to-window dispersions. Only the sustained path measures in windows,
so these four are always empty on reading files. They stay in the shared block
so the CSV keeps one fixed shape. In practice, a reading file has 56 values
that can be filled from this block, a sustained file 60.

### Reading passages only (21)

| Group | n | Metrics |
|---|---|---|
| Rate | 4 | `speech_rate_syll_per_s`, `articulation_rate_syll_per_s`, `syllable_count_used`, `speaking_time_fraction` |
| Fillers | 4 | `filler_count`, `fillers_per_minute`, `filler_total_duration_s`, `filler_ratio_to_speech` |
| Rhythm and envelope modulation | 4 | `ems_3_8hz_ratio`, `ems_peak_freq_hz`, `ems_syllabic_to_slow_ratio`, `npvi` |
| Vowel space | 3 | `vowel_space_area`, `vowel_dispersion_logarea`, `vowel_cloud_spread_hz` |
| Fluency index | 6 | `fluency_intensity_stability`, `fluency_pitch_stability`, `fluency_rhythm_regularity`, `fluency_voice_quality`, `fluency_articulation`, `fluency_overall` |

The six `fluency_*` columns are an unvalidated composite with hand-chosen
weights. The index is off by default (`enable_fluency_index=False`) and should
not be read as a severity rating.

`speech_rate_syll_per_s` and `articulation_rate_syll_per_s` are exact only when
the passage syllable count is right for your wording. `speaking_time_fraction`
does not depend on it.

### Sustained vowels only (22)

| Group | n | Metrics |
|---|---|---|
| Maximum phonation time and tokens | 5 | `sv_mpt_longest_s`, `sv_mpt_longest_uninterrupted_s`, `sv_mpt_mean_s`, `sv_total_phonation_s`, `sv_n_tokens` |
| Steady-state pitch and loudness | 3 | `sv_f0_sd_semitones`, `sv_f0_drift_st_per_s`, `sv_intensity_sd_db` |
| Tremor | 2 | `sv_tremor_rate_hz`, `sv_tremor_extent_st` |
| Token-level voice quality | 2 | `sv_cpps_token_db`, `sv_hnr_token_db` |
| Window yield | 1 | `sv_window_yield` |
| Intensity decay | 1 | `sv_intensity_decay_db_per_s` |
| Coverage: how much of the vowel was measured | 4 | `sv_analyzed_fraction`, `sv_window_yield_s`, `sv_steady_frame_fraction`, `sv_longest_steady_run_s` |
| Articulatory steadiness | 1 | `sv_f1f2_cloud_spread_hz` |
| Reproducibility across tokens | 3 | `sv_jitter_sd_across_tokens`, `sv_shimmer_sd_across_tokens`, `sv_hnr_sd_across_tokens` |

### Audit columns (73)

These are written to a separate block of the CSV and are not metrics. They
record which settings were actually used, the octave check, the window
rejection counts, the usability gates and the recording properties. They are
never plotted and never aggregated as outcomes. When two sessions disagree,
read this block first: a change in `pitch_floor_used_hz` can explain a
"finding" that is not one.

`task_is_sustained_vowel`, `one_production_per_file`, `vq_measured`, `cpps_is_praat`, `syllable_source_known`, `syllable_count_estimated`, `syllable_count_agreement`, `formant_track_ok`, `formant_ceiling_hz_used`, `pitch_octave_repair_fraction`, `pitch_floor_used_hz`, `pitch_ceiling_used_hz`, `silence_threshold_db_used`, `f0_octave_ok`, `f0_octave_corrected`, `f0_octave_score_ref_db`, `f0_octave_score_half_db`, `f0_wide_median_hz`, `sv_measured`, `sv_usable`, `sv_windows_usable`, `sv_perturbation_usable`, `sv_signal_type`, `sv_n_windows_valid`, `sv_n_windows_total`, `sv_measured_total_s`, `sv_n_tokens_expected`, `sv_token_count_mismatch`, `sv_boundaries_from_file`, `sv_n_splices_detected`, `sv_analysis_window_s`, `sv_rej_low_voicing`, `sv_rej_f0_outlier`, `sv_rej_f0_step`, `sv_rej_splice`, `sv_rej_other`, `sv_analyzable_total_s`, `sv_discard_unsteady_s`, `sv_discard_edge_s`, `sv_discard_quantisation_s`, `sv_discard_onset_s`, `sv_discard_interior_s`, `sv_discard_offset_s`, `sv_discard_dead_token_s`, `sv_n_interior_gaps`, `sv_n_steady_stretches`, `sv_voiced_at_045`, `sv_voiced_at_020`, `sv_voiced_low_floor`, `sv_tremor_peak_ratio`, `sv_f1f2_dispersion_logarea`, `sv_n_phonation_breaks`, `sv_n_breaks_in_best_token`, `sv_unbroken_start_s`, `sv_unbroken_end_s`, `sv_unbroken_token_index`, `sv_unbroken_energy_only_s`, `sv_unbroken_voicing_required`, `sv_break_threshold_db`, `sv_break_min_duration_s`, `cpp_is_praat_cpps`, `spectral_span_intervals`, `formant_span_windows`, `spectral_tilt_band_lo_hz`, `spectral_tilt_band_hi_hz`, `pitch_block_from_windows`, `rec_effective_bandwidth_hz`, `rec_spectral_edge_hz`, `rec_sample_rate_hz`, `rec_is_bandlimited`, `rec_vq_reliable`, `rec_snr_estimable`, `rec_digital_silence`

## Phoneme recognition

One TSV per recording. The comment header holds 4 values: duration, phoneme
count, speech rate and articulation rate. Each row is one segment with 5 fields:
type (phoneme or pause), label, confidence, start and end.

## Alignment

`score_block()` returns 8 values per session, from the same Needleman-Wunsch
alignment that produces the printed output:

| Score | Meaning |
|---|---|
| `canonical` | phonemes in the reference sequence |
| `produced` | phonemes in the session TSV |
| `correct` | aligned pairs that match |
| `substitutions` | aligned pairs that differ |
| `omissions` | reference phonemes with nothing produced |
| `insertions` | produced phonemes with no reference counterpart |
| `accuracy` | correct / canonical |
| `error_rate` | (substitutions + omissions + insertions) / canonical |

`error_rate` can exceed 1 when there are many insertions; `accuracy` cannot.

## DDK analysis

The column names below are taken from `master_rows()` and `summary_rows()` in
`speechkit/ddk.py`, with the default settings.

### Core measures (12 per trial)

Each trial is analysed three ways, and each analysis reports the same four
measures:

| Measure | Column suffix | Definition |
|---|---|---|
| Rate | `DDK_Rate` | syllables per second (see the table below for the denominator) |
| ISI mean | `ISI_mean_ms` | mean inter-syllable interval, in ms |
| ISI CV | `ISI_CV` | SD / mean of the inter-syllable intervals |
| InstRate CV | `InstRate_CV` | SD / mean of the local rate over sliding 3-syllable windows (`DDK_TRIPLET_WIN`) |

| Analysis | Prefix | Portion of the trial | Rate |
|---|---|---|---|
| Whole trial | `Whole_` | every detected envelope peak | syllables / active-speech duration |
| Best window | `Best3s_` | the 3-s window with the most syllables (ties: lowest ISI CV, then earliest) | syllables in window / window length; needs >= 2 syllables, ISI CV needs >= 3 |
| Pierce, Cotton and Perry (2013) | `Pierce9_` | onset of glottal pulsing of syllable 1 to that of syllable 10 | 9 / duration, duration to 4 dp and rate to 1 dp, as in the paper |

With the default `DDK_PIERCE_VAR_TIMES = onsets`, the Pierce variability
measures use the ten voicing onsets, so `Pierce9_ISI_mean_ms` is the duration
divided by 9. With `peaks` they use the envelope peaks of the same syllables,
the same definition as `Whole_` and `Best3s_`.

### All measurement columns on the Trials sheet (22)

| Analysis | n | Columns |
|---|---|---|
| Whole trial | 8 | `Whole_SyllableCount`, `Whole_DDK_Rate`, `Whole_ActiveDurationSec`, `Whole_SpeechStartSec`, `Whole_SpeechEndSec`, `Whole_ISI_mean_ms`, `Whole_ISI_CV`, `Whole_InstRate_CV` |
| Best window | 7 | `Best3s_WindowStartSec`, `Best3s_WindowEndSec`, `Best3s_SyllableCount`, `Best3s_DDK_Rate`, `Best3s_ISI_mean_ms`, `Best3s_ISI_CV`, `Best3s_InstRate_CV` |
| Pierce | 7 | `Pierce9_Onset1Sec`, `Pierce9_Onset10Sec`, `Pierce9_DurationSec`, `Pierce9_DDK_Rate`, `Pierce9_ISI_mean_ms`, `Pierce9_ISI_CV`, `Pierce9_InstRate_CV` |

The rest of the row is identifiers and bookkeeping, not measurements:
`RelativePath`, `Session`, `Filename`, `Subject`, `Date`, `Day`, `AudioNum`,
`Stim`, any extra metadata columns, `Trial`, `CueLabel`, `SampleRate`,
`FileDurationSec`, `Pierce9_Included` (meets the paper's inclusion criteria:
at least 10 syllables and 4 s of active speech), the two `_SegmentFile` columns
and the three `Notes` columns.

The prefixes follow the settings. A 4-second window gives `Best4s_`, the
`first` criterion gives `First3s_`, and `DDK_PIERCE_N_SYLL = 12` gives
`Pierce12_` with `Onset13Sec`.

### Sessions sheet

One row per session (date x audio number) and cue, for each of the cues in
`DDK_CUES` (`pataka`, `tatata`, `papapa`, `kakaka`), plus an `AMR` row. Each row
has `n_trials` and 6 means, rounded to 3 dp:

`Whole_DDK_Rate`, `Whole_ISI_CV`, `Best3s_DDK_Rate`, `Best3s_ISI_CV`,
`Pierce9_DDK_Rate`, `Pierce9_ISI_CV`

The AMR row is the mean of the `tatata`, `papapa` and `kakaka` means
(`DDK_AMR_CUES`). These means average every trial of the cue with a finite
value, including Pierce trials with `Pierce9_Included = False`. They are a
descriptive summary; the statistics use the stricter selection below.

A trial whose `CueLabel` is not in `DDK_CUES` (compared ignoring case and
surrounding spaces) is on the Trials sheet but in neither the Sessions sheet
nor the statistics.

### Statistics

They run on the measures in `DDK_STATS_METRICS`, by default only
`Pierce9_DDK_Rate`. The unit is one value per session and cue: the first valid
trial (`DDK_TRIAL_AGG = first`); for `Pierce9_` measures a trial is valid only
when `Pierce9_Included` is true. By default only Stim `OFF` and `ON` sessions
are kept, and values outside 1.5 x IQR are removed before the tests.

| Sheet | Contents |
|---|---|
| `Normality` | skewness and kurtosis with SPSS standard errors and z, before and after outlier removal, plus a distribution figure |
| `Table3` | n, mean (1 dp), SD (2 dp) and range per cue and group, in the layout of the paper's Table 3 |
| `Levene` | Levene's F test for equal variances between the groups of `DDK_LEVENE_FACTOR` |
| `ANOVA` | independent-groups ANOVA per cue, Type III SS, with partial eta squared |
| `Descriptives` | long-format descriptives per cue and group, with totals |
| `Reliability_ICC`, `Reliability_trials` | only when a filled-in reliability sheet is given: ICC(2,1) and ICC(3,1), automatic vs manual |
