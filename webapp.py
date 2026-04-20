"""
Webapp locale pour le générateur de Reels.

Drag-and-drop une vidéo dans le browser → progress en temps réel → grid
de previews vidéo avec hooks/captions/hashtags.

Usage:
    source venv312/bin/activate
    python3 webapp.py
    → ouvre http://localhost:5151

Conçu pour tourner en local seulement, pas exposé sur internet.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "webapp_jobs"
JOBS_DIR.mkdir(exist_ok=True)

VENV_PYTHON = ROOT / "venv312" / "bin" / "python3"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

app = Flask(__name__, static_folder=str(ROOT / "webapp_static"))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB max upload

# job_id -> {"status": str, "progress": int, "step": str, "log": [..],
#            "clips": [..], "error": str|None, "video_name": str}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Pipeline runner (background thread)
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("Transcription", r"Transcription en cache|Transcription Whisper|✅ \d+ mots"),
    ("Analyse audio", r"Analyse audio en cache|Analyse de l'énergie audio|secondes 'peak'"),
    ("Détection clips", r"Détection des clips topicaux|clips candidats|viral arcs construits"),
    ("LLM judge", r"LLM judge|/25\] |/20\]"),
    ("Génération hooks", r"LLM génère hook"),
    ("Rendu vidéo", r"🎞️.*reel_"),
    ("Terminé", r"✨ Terminé"),
]


def _update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _append_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["log"].append(line)
            if len(JOBS[job_id]["log"]) > 400:
                JOBS[job_id]["log"] = JOBS[job_id]["log"][-300:]


def _detect_step(line: str, current_step_idx: int) -> int:
    for i in range(current_step_idx, len(PIPELINE_STEPS)):
        _, pattern = PIPELINE_STEPS[i]
        if re.search(pattern, line):
            return i
    return current_step_idx


def _scan_clips(out_dir: Path) -> list:
    """Lit summary.json + scanne les .mp4/.txt/.srt pour le job."""
    if not out_dir.exists():
        return []
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return []

    clips = []
    for entry in summary:
        fname = entry.get("file") or ""
        mp4 = out_dir / fname
        if not mp4.exists():
            continue
        clips.append({
            "file": fname,
            "hook": entry.get("hook", ""),
            "caption": entry.get("caption", ""),
            "hashtags": entry.get("hashtags", []),
            "duration": entry.get("duration", 0),
            "score": entry.get("score", 0),
            "reason": entry.get("reason", ""),
        })
    return clips


def _run_pipeline(job_id: str, video_path: Path, out_dir: Path,
                  max_clips: int, max_len: float) -> None:
    """Lance make_reels_v3.py en subprocess et stream le stdout vers le job."""
    _update_job(job_id, status="running", progress=2, step="Démarrage…")

    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "make_reels_v3.py"),
        str(video_path),
        "--out", str(out_dir),
        "--max-clips", str(max_clips),
        "--max-len", str(max_len),
        "--shortlist", "25",
        # English-only pipeline: lock language + use the EN-tuned Whisper model.
        # Skips language detection (~30% faster) and uses a model trained
        # exclusively on English (better WER than the multilingual 'small').
        "--lang", "en",
        "--model", "small.en",
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(job_id, status="error", error=str(exc), progress=0)
        return

    current_step_idx = 0
    rendered_count = 0
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if not line:
            continue
        _append_log(job_id, line)

        new_idx = _detect_step(line, current_step_idx)
        if new_idx != current_step_idx:
            current_step_idx = new_idx
            step_name = PIPELINE_STEPS[current_step_idx][0]
            progress = int(((current_step_idx + 1) / len(PIPELINE_STEPS)) * 100)
            _update_job(job_id, step=step_name, progress=progress)

        if "🎞️" in line and "→" in line:
            rendered_count += 1
            base = int(((len(PIPELINE_STEPS) - 1) / len(PIPELINE_STEPS)) * 100)
            extra = min(rendered_count, max_clips) / max_clips * (100 - base)
            _update_job(job_id, progress=int(base + extra),
                        step=f"Rendu {rendered_count}/{max_clips}")

    proc.wait()
    if proc.returncode != 0:
        _update_job(job_id, status="error",
                    error=f"Pipeline a échoué (code {proc.returncode})",
                    progress=0)
        return

    clips = _scan_clips(out_dir)
    _update_job(job_id, status="done", progress=100, step="Terminé",
                clips=clips)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "no filename"}), 400

    max_clips = int(request.form.get("max_clips", "8"))
    max_len = float(request.form.get("max_len", "90"))

    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", f.filename)
    src_path = job_dir / safe_name
    f.save(str(src_path))

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "step": "En attente…",
            "log": [],
            "clips": [],
            "error": None,
            "video_name": safe_name,
            "out_dir": str(out_dir),
            "started_at": time.time(),
        }

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, src_path, out_dir, max_clips, max_len),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        return jsonify({
            "status": job["status"],
            "progress": job["progress"],
            "step": job["step"],
            "log_tail": job["log"][-15:],
            "clips": job.get("clips", []),
            "error": job.get("error"),
            "video_name": job.get("video_name", ""),
            "elapsed": int(time.time() - job["started_at"]),
        })


@app.route("/clip/<job_id>/<path:filename>")
def clip(job_id: str, filename: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        out_dir = Path(job["out_dir"])
    return send_from_directory(str(out_dir), filename, as_attachment=False)


@app.route("/export/<job_id>", methods=["POST"])
def export_to_desktop(job_id: str):
    """Copie tous les clips d'un job dans ~/Desktop/Reels - <video name>/."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        out_dir = Path(job["out_dir"])
        video_name = job.get("video_name", "video")

    base_name = Path(video_name).stem
    dest = Path.home() / "Desktop" / f"Reels - {base_name}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for entry in out_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in {".mp4", ".srt", ".txt", ".json"}:
            shutil.copy2(str(entry), str(dest / entry.name))
    return jsonify({"path": str(dest), "count": len(list(dest.iterdir()))})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("REELS_PORT", "5151"))
    url = f"http://localhost:{port}"
    print("=" * 64)
    print(f"🎬 Reels Generator — webapp lancée")
    print(f"   → Ouvre dans ton browser : {url}")
    print(f"   (Ctrl+C pour stopper)")
    print("=" * 64)

    if os.environ.get("REELS_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
