"""
The local interface.

One Flask app serving one page per stage plus the phoneme editor, which is the
same editor that used to be embedded in `phonemes_from_audio.py` as a 545-line
string. Its four endpoints keep their original paths (/api/filelist, /api/data,
/api/save_segments, /api/export) so the page works unchanged.

Bound to 127.0.0.1 by default. There is no authentication and the path browser
can see the whole filesystem, which is exactly what a local research tool
needs and exactly what must not be exposed on a network interface.
"""

from __future__ import annotations

import os
import string
import threading
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from .. import environment, settings
from .._files import collect_audio
from ..jobs import JobRunner

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

RUNNER = JobRunner()

# Current settings, held in memory so a run and the page agree on the values,
# and written to disk whenever they change.
VALUES: dict[str, Any] = settings.load()

# The last phoneme recognition result, which is what the editor reads.
FILES: list[dict[str, Any]] = []


# ==========================================================================
# Pages
# ==========================================================================


@app.route("/")
def index():
    return render_template("index.html", version=environment()["version"])


@app.route("/editor")
def editor():
    return render_template("editor.html")


# ==========================================================================
# Settings
# ==========================================================================


@app.route("/api/schema")
def api_schema():
    return jsonify({
        "schema": settings.schema(),
        "values": VALUES,
        "sections": settings.SECTION_TITLES,
        "presets": settings.list_presets(),
        "settingsFile": str(settings.SETTINGS_FILE),
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global VALUES
    incoming = request.get_json(silent=True) or {}
    try:
        merged = settings.coerce_all(incoming, VALUES)
    except settings.SettingError as exc:
        return jsonify({"error": str(exc)}), 400
    VALUES = merged
    settings.save(VALUES)
    return jsonify({"values": VALUES, "saved": str(settings.SETTINGS_FILE)})


@app.route("/api/settings/reset", methods=["POST"])
def api_settings_reset():
    global VALUES
    section = (request.get_json(silent=True) or {}).get("section")
    if section:
        for s in settings.SETTINGS:
            if s.section == section:
                VALUES[s.name] = settings.defaults()[s.name]
    else:
        VALUES = settings.defaults()
    settings.save(VALUES)
    return jsonify({"values": VALUES})


@app.route("/api/presets", methods=["GET", "POST", "DELETE"])
def api_presets():
    global VALUES
    if request.method == "GET":
        name = request.args.get("name", "")
        if not name:
            return jsonify({"presets": settings.list_presets()})
        try:
            loaded = settings.load(settings.preset_path(name))
        except settings.SettingError as exc:
            return jsonify({"error": str(exc)}), 400
        if not settings.preset_path(name).is_file():
            return jsonify({"error": f"no preset called {name!r}"}), 404
        VALUES = loaded
        settings.save(VALUES)
        return jsonify({"values": VALUES, "loaded": name})

    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    try:
        target = settings.preset_path(name)
    except settings.SettingError as exc:
        return jsonify({"error": str(exc)}), 400

    if request.method == "DELETE":
        target.unlink(missing_ok=True)
        return jsonify({"presets": settings.list_presets()})

    settings.save(VALUES, target)
    return jsonify({"presets": settings.list_presets(), "saved": name})


# ==========================================================================
# Environment
# ==========================================================================


@app.route("/api/env")
def api_env():
    return jsonify(environment())


# ==========================================================================
# Path browser
#
# A browser file input hands back file contents, not paths, and the analysis
# needs paths. So the picking happens server-side: this lists directories and
# counts the .wav files in them, and the page walks the tree.
# ==========================================================================


def _drives() -> list[dict[str, str]]:
    if os.name != "nt":
        return []
    out = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            out.append({"name": root, "path": root})
    return out


@app.route("/api/browse")
def api_browse():
    raw = request.args.get("path", "").strip()
    show_files = request.args.get("files", "1") != "0"

    if not raw:
        home = Path.home()
        roots = _drives()
        return jsonify({
            "path": str(home),
            "parent": str(home.parent) if home.parent != home else None,
            "roots": roots,
            "dirs": _list_dirs(home),
            "files": _list_wavs(home) if show_files else [],
            "error": None,
        })

    target = Path(raw).expanduser()
    if not target.exists():
        return jsonify({"error": f"{target} does not exist", "path": raw,
                        "dirs": [], "files": [], "roots": _drives()}), 404
    if target.is_file():
        target = target.parent
    try:
        dirs = _list_dirs(target)
        files = _list_wavs(target) if show_files else []
    except PermissionError:
        return jsonify({"error": f"no permission to read {target}", "path": str(target),
                        "dirs": [], "files": [], "roots": _drives()}), 403

    parent = target.parent
    return jsonify({
        "path": str(target),
        "parent": str(parent) if parent != target else None,
        "roots": _drives(),
        "dirs": dirs,
        "files": files,
        "error": None,
    })


def _list_dirs(folder: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        try:
            n_wav = sum(1 for c in p.iterdir() if c.is_file() and c.suffix.lower() == ".wav")
        except (PermissionError, OSError):
            n_wav = -1
        out.append({"name": p.name, "path": str(p), "wavs": n_wav})
    return out


def _list_wavs(folder: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in {".wav", ".txt", ".tsv", ".csv", ".json"}:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            out.append({"name": p.name, "path": str(p), "size": size,
                        "kind": p.suffix.lower().lstrip(".")})
    return out


@app.route("/api/preview_inputs", methods=["POST"])
def api_preview_inputs():
    """How many recordings the current selection actually resolves to, and
    which entries do not exist. Cheaper than starting a run to find out."""
    body = request.get_json(silent=True) or {}
    inputs = body.get("inputs", [])
    recursive = bool(body.get("recursive", False))
    from .._files import missing_inputs

    wavs = collect_audio(inputs, recursive=recursive)
    return jsonify({
        "count": len(wavs),
        "sample": [Path(w).name for w in wavs[:12]],
        "missing": missing_inputs(inputs),
    })


# ==========================================================================
# Running a stage
# ==========================================================================


def _stage_acoustics(cfg, log, should_stop):
    from ..acoustics import run_acoustics

    return run_acoustics(cfg, log)


def _stage_phonemes(cfg, log, should_stop):
    from ..phonemes import run_phonemes

    records = run_phonemes(cfg, log, write_files=True, keep_audio=True,
                           should_stop=should_stop)
    FILES.clear()
    FILES.extend(records)
    log(f"[editor] {len(FILES)} file(s) ready to review at /editor")
    return [{"wav": r["wav_path"], "metrics": r["metrics"],
             "tsv": r.get("tsv_path")} for r in records]


def _stage_alignment(cfg, log, should_stop):
    from ..alignment import run_alignment

    return run_alignment(cfg, log)


STAGES = {
    "acoustics": _stage_acoustics,
    "phonemes": _stage_phonemes,
    "alignment": _stage_alignment,
}


@app.route("/api/run/<stage>", methods=["POST"])
def api_run(stage: str):
    global VALUES
    if stage not in STAGES:
        return jsonify({"error": f"unknown stage {stage!r}"}), 404

    incoming = request.get_json(silent=True) or {}
    try:
        VALUES = settings.coerce_all(incoming, VALUES)
    except settings.SettingError as exc:
        return jsonify({"error": str(exc)}), 400
    settings.save(VALUES)

    if stage == "acoustics" and not str(VALUES.get("TASK", "")).strip():
        return jsonify({
            "error": "Choose what is in these recordings first: a reading passage, "
                     "a sustained vowel, or a mixed folder. The two tasks are "
                     "measured differently, so there is no safe default."
        }), 400

    env = environment()["stages"].get(stage, {})
    if not env.get("ready", True):
        return jsonify({
            "error": f"{stage} needs {', '.join(env['missing'])}. "
                     f"Install with: pip install {' '.join(env['missing'])}"
        }), 400

    try:
        RUNNER.start(stage, STAGES[stage], dict(VALUES))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"started": stage})


@app.route("/api/job")
def api_job():
    since = int(request.args.get("since", 0))
    status = RUNNER.status(since)
    job = RUNNER.current
    if job is not None and job.state == "done":
        status["summary"] = _summarise(job)
    return jsonify(status)


@app.route("/api/job/cancel", methods=["POST"])
def api_job_cancel():
    return jsonify({"cancelled": RUNNER.cancel()})


def _summarise(job) -> dict[str, Any]:
    """A small result table per stage, so the page shows numbers rather than
    only the last line of the log."""
    if job.stage == "phonemes" and isinstance(job.result, list):
        return {
            "kind": "phonemes",
            "rows": [
                {"file": Path(r["wav"]).name,
                 "phonemes": r["metrics"]["n_phonemes"],
                 "duration": round(r["metrics"]["duration_s"], 2),
                 "speech_rate": round(r["metrics"]["speech_rate"], 2),
                 "articulation_rate": round(r["metrics"]["articulation_rate"], 2),
                 "tsv": r.get("tsv")}
                for r in job.result
            ],
            "editorReady": len(FILES) > 0,
        }
    if job.stage == "alignment" and isinstance(job.result, list):
        rows = []
        for report in job.result:
            for s in report["sessions"]:
                rows.append({
                    "passage": report["passage"], "session": s["session"],
                    "canonical": s["canonical"], "produced": s["produced"],
                    "accuracy": round(s["accuracy"] * 100, 1),
                    "substitutions": s["substitutions"],
                    "omissions": s["omissions"], "insertions": s["insertions"],
                })
        return {"kind": "alignment", "rows": rows,
                "outputs": [r["output"] for r in job.result if r["output"]]}
    if job.stage == "acoustics":
        folder = VALUES.get("OUTPUT_FOLDER", "")
        produced: list[dict[str, Any]] = []
        if folder and Path(folder).is_dir():
            for p in sorted(Path(folder).iterdir()):
                if p.is_file():
                    produced.append({"name": p.name, "path": str(p),
                                     "size": p.stat().st_size})
        return {"kind": "acoustics",
                "files": len(job.result or []),
                "outputs": produced,
                "folder": folder}
    return {"kind": job.stage}


@app.route("/api/download")
def api_download():
    raw = request.args.get("path", "")
    target = Path(raw).expanduser()
    if not target.is_file():
        return jsonify({"error": f"{raw} is not a file"}), 404
    return send_file(str(target), as_attachment=True, download_name=target.name)


# ==========================================================================
# Phoneme editor endpoints.
# Same four paths and payloads as the original embedded app, so the editor page
# needed no changes.
# ==========================================================================


@app.route("/api/filelist")
def api_filelist():
    return jsonify([
        {"index": i, "name": os.path.basename(f["wav_path"]),
         "duration": f["duration"], "n_segments": len(f["segments"])}
        for i, f in enumerate(FILES)
    ])


@app.route("/api/data")
def api_data():
    idx = int(request.args.get("file", 0))
    if idx < 0 or idx >= len(FILES):
        return jsonify({"error": "invalid index"}), 404
    f = FILES[idx]
    return jsonify({
        "segments": f["segments"],
        "rms": f["rms"],
        "duration": f["duration"],
        "threshold": f["threshold"],
        "audio_b64": f["audio_b64"],
        "wav_path": f["wav_path"],
    })


@app.route("/api/save_segments", methods=["POST"])
def api_save_segments():
    body = request.get_json(silent=True) or {}
    idx = int(body.get("file_index", 0))
    segs = body.get("segments", [])
    if 0 <= idx < len(FILES):
        FILES[idx]["segments"] = segs
        return jsonify({"status": "ok"})
    return jsonify({"error": "invalid index"}), 404


@app.route("/api/export", methods=["POST"])
def api_export():
    from ..phonemes import tsv_path_for, write_tsv

    if not FILES:
        return jsonify({"error": "nothing to export; run phoneme recognition first"}), 400
    body = request.get_json(silent=True) or {}
    segs = body.get("segments", [])
    idx = int(body.get("file_index", 0))
    idx = idx if 0 <= idx < len(FILES) else 0
    record = FILES[idx]
    record["segments"] = segs

    out_path = tsv_path_for(record["wav_path"], VALUES.get("PHONEME_OUTPUT_DIR", ""),
                            "_edited.tsv")
    metrics = write_tsv(out_path, segs, record["duration"], record["threshold"])
    return jsonify({
        "status": "ok",
        "path": metrics["path"],
        "speech_rate": round(metrics["speech_rate"], 4),
        "articulation_rate": round(metrics["articulation_rate"], 4),
    })


# ==========================================================================
# Entry point
# ==========================================================================


def create_app() -> Flask:
    return app


def serve(host: str | None = None, port: int | None = None,
          open_browser: bool | None = None) -> None:
    host = host or VALUES.get("HOST", "127.0.0.1")
    port = int(port if port is not None else VALUES.get("PORT", 7331))
    if open_browser is None:
        open_browser = bool(VALUES.get("OPEN_BROWSER", True))

    check = environment()["python_check"]
    if check["ok"] is not True:
        print(f"warning: {check['message']}")

    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"speechkit interface at {url}")
    print(f"settings file: {settings.SETTINGS_FILE}")
    if host not in ("127.0.0.1", "localhost"):
        print("warning: bound to a non-local address. This interface has no "
              "authentication and can browse the filesystem.")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
