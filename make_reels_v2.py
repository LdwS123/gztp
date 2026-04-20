#!/usr/bin/env python3
"""
Générateur de Reels v2 — backed by ClipsAI + WhisperX.

Pipeline :
  1. Transcription via WhisperX (clipsai.Transcriber) → timestamps précis au mot
  2. Détection de clips par ClipFinder (TextTiling sur embeddings de phrases)
     → frontières TOUJOURS sur début/fin de phrase (= cuts propres)
  3. Pour chaque clip candidat, on calcule notre score viral
     (texte + énergie audio = peaks / laughter)
  4. On retient les meilleurs et on génère les MP4 verticaux 9:16
     avec fond flouté + .srt + .txt (hook + caption + hashtags).

Tu n'as qu'à fournir la vidéo :

    python make_reels_v2.py video.mp4
    python make_reels_v2.py video.mp4 --max-clips 5 --out ~/Desktop/reels
    python make_reels_v2.py video.mp4 --model small --horizontal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

# Avoid OMP / torchcodec warnings cluttering output
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
warnings.filterwarnings("ignore")

# Re-use everything we already have for audio analysis, scoring, hooks…
from make_reels import (  # noqa: E402
    Word,
    Segment,
    AudioAnalysis,
    analyze_audio_energy,
    get_or_make_audio_analysis,
    score_segment,
    select_top_segments,
    extract_hook,
    extract_keywords,
    make_hashtags,
    make_caption,
    enrich_segments,
    trim_segment_edges,
    make_clip,
    words_to_srt,
    slugify,
    extract_audio,
    ffprobe_duration,
)


# ---------------------------------------------------------------------------
# Transcription + clip finding (clipsai)
# ---------------------------------------------------------------------------

def transcribe_with_clipsai(audio_path: Path, model_size: str, language: Optional[str] = "en"):
    """Lance WhisperX via clipsai. Retourne un objet Transcription.

    `language` doit être un code ISO 639-1 (ex: 'en', 'fr', 'es').
    Par défaut 'en' (skip la détection de langue → +30% vitesse, +précision).
    Mettre None pour auto-detect.
    """
    lang_str = language or "auto"
    print(f"🗣️  Transcription WhisperX (modèle '{model_size}', langue: {lang_str})…")
    from clipsai import Transcriber

    transcriber = Transcriber(model_size=model_size, device="cpu", precision="int8")
    transcription = transcriber.transcribe(str(audio_path), iso6391_lang_code=language)
    try:
        n_words = len(transcription.words)
    except Exception:
        n_words = "?"
    try:
        n_sent = len(transcription.sentences)
    except Exception:
        n_sent = "?"
    print(f"   ✅ {n_words} mots, {n_sent} phrases")
    return transcription


def transcription_to_words(transcription) -> List[Word]:
    """Convertit clipsai.Transcription → List[Word] (notre type)."""
    out: List[Word] = []
    for w in transcription.words:
        if w.start_time is None or w.end_time is None:
            continue
        out.append(Word(text=w.text or "", start=float(w.start_time), end=float(w.end_time)))
    return out


def find_clips_clipsai(
    transcription,
    min_len: float,
    max_len: float,
) -> List[Tuple[float, float]]:
    """ClipFinder de clipsai : retourne une liste de (start, end) propres."""
    print("🔍 Détection des clips topicaux (ClipFinder)…")
    from clipsai import ClipFinder

    finder = ClipFinder(
        device="cpu",
        min_clip_duration=int(min_len),
        max_clip_duration=int(max_len),
        cutoff_policy="average",  # 'low' = + de candidats, 'high' = – de candidats
    )
    clips = finder.find_clips(transcription)
    print(f"   ✅ {len(clips)} clips candidats")
    return [(float(c.start_time), float(c.end_time)) for c in clips]


def clips_to_segments(
    clip_ranges: List[Tuple[float, float]],
    words: List[Word],
) -> List[Segment]:
    """Construit nos Segment à partir des frontières clipsai + nos Word."""
    segments: List[Segment] = []
    if not words:
        return segments

    for start, end in clip_ranges:
        # Garde les mots dont le début est dans la fenêtre
        seg_words = [w for w in words if w.start >= start - 0.05 and w.end <= end + 0.05]
        if not seg_words:
            continue
        text = " ".join(w.text for w in seg_words).strip()
        if not text:
            continue
        segments.append(
            Segment(
                start=seg_words[0].start,
                end=seg_words[-1].end,
                text=text,
                words=seg_words,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Cache (transcript JSON)
# ---------------------------------------------------------------------------

def transcript_cache_path(video: Path, cache_dir: Path, lang: str = "en") -> Path:
    # On inclut la langue dans le nom du cache pour ne pas mélanger les
    # transcriptions EN et FR du même fichier.
    return cache_dir / f"{video.stem}.{lang}.clipsai.json"


def get_or_make_transcript(video: Path, cache_dir: Path, model_size: str,
                           language: Optional[str] = "en"):
    """Renvoie un objet Transcription clipsai (depuis cache ou fraîchement transcrit).

    `language` : ISO 639-1 ('en' par défaut). None = auto-detect.
    """
    from clipsai.transcribe.transcription import Transcription

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = transcript_cache_path(video, cache_dir, lang=language or "auto")

    # Compat ascendante : on accepte aussi l'ancien cache sans langue dans le nom
    legacy = cache_dir / f"{video.stem}.clipsai.json"
    if not cache_file.exists() and legacy.exists():
        cache_file = legacy

    if cache_file.exists():
        print(f"📂 Transcription en cache: {cache_file.name}")
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return Transcription(data)

    audio = cache_dir / f"{video.stem}.wav"
    if not audio.exists():
        print("🎧 Extraction audio…")
        extract_audio(video, audio)

    transcription = transcribe_with_clipsai(audio, model_size, language=language)
    transcription.store_as_json_file(str(cache_file))
    print(f"💾 Transcription sauvée : {cache_file.name}")
    return transcription


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère des Reels viraux avec ClipsAI + scoring perso.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Vidéo source (mp4, mov, mkv…)")
    p.add_argument("--out", type=Path, default=Path("reels_output"),
                   help="Dossier de sortie (défaut: reels_output/)")
    p.add_argument("--max-clips", type=int, default=5,
                   help="Nombre de clips finaux à garder (défaut: 5)")
    p.add_argument("--min-len", type=float, default=15.0,
                   help="Durée min d'un clip en secondes (défaut: 15)")
    p.add_argument("--max-len", type=float, default=60.0,
                   help="Durée max d'un clip en secondes (défaut: 60)")
    p.add_argument("--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Modèle Whisper (défaut: base)")
    p.add_argument("--horizontal", action="store_true",
                   help="Garde le format horizontal 16:9 (sinon 9:16 fond flouté)")
    p.add_argument("--dry-run", action="store_true",
                   help="Affiche juste la sélection sans découper la vidéo")
    p.add_argument("--cache-dir", type=Path, default=Path("transcripts"),
                   help="Dossier de cache transcription/audio (défaut: transcripts/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists() and not args.dry_run:
        print(f"❌ Vidéo introuvable: {args.video}")
        sys.exit(1)

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(args.video) if args.video.exists() else 0
    fmt = "16:9 horizontal" if args.horizontal else "9:16 fond flouté"
    print("=" * 64)
    print(f"🎬 Vidéo:    {args.video.name}")
    if duration:
        print(f"   Durée:    {int(duration)}s")
    print(f"🎯 Cible:    {args.max_clips} reels, {int(args.min_len)}-{int(args.max_len)}s, {fmt}")
    print(f"🧠 Engine:   ClipsAI (TextTiling) + scoring viral perso")
    print("=" * 64)

    # 1) Transcription
    transcription = get_or_make_transcript(args.video, cache_dir, args.model)
    words = transcription_to_words(transcription)

    # 2) Audio analysis (peaks, laughter)
    audio_analysis = get_or_make_audio_analysis(args.video, cache_dir) if args.video.exists() else None

    # 3) ClipFinder → frontières propres
    clip_ranges = find_clips_clipsai(transcription, args.min_len, args.max_len)
    candidates = clips_to_segments(clip_ranges, words)
    if not candidates:
        print("❌ Aucun clip détecté.")
        sys.exit(1)

    # 4) Scoring viral + sélection
    chosen = select_top_segments(candidates, args.max_clips, audio=audio_analysis)
    chosen = [trim_segment_edges(seg) for seg in chosen]
    enrich_segments(chosen)

    # 5) Affichage
    print(f"\n✅ {len(chosen)} clips retenus :\n")
    for i, seg in enumerate(chosen, 1):
        print(f"  [{i}] {seg.start:7.1f}s → {seg.end:7.1f}s  ({seg.end - seg.start:.1f}s)  score={seg.score:5.1f}")
        if seg.hook:
            print(f"      🎯 HOOK     : {seg.hook}")
        if seg.caption:
            print(f"      📝 CAPTION  : {seg.caption}")
        if seg.hashtags:
            print(f"      🏷️  HASHTAGS: {' '.join(seg.hashtags)}")
        snippet = seg.text[:120].replace("\n", " ")
        print(f"      💬          « {snippet}{'…' if len(seg.text) > 120 else ''} »")
        print()

    if args.dry_run:
        print("🛑 --dry-run: pas de découpage vidéo.")
        return

    # 6) Génération vidéo + fichiers compagnons
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, seg in enumerate(chosen, 1):
        slug = slugify(seg.text, max_len=35)
        base = f"reel_{i:02d}_{slug}"
        mp4 = args.out / f"{base}.mp4"
        srt = args.out / f"{base}.srt"
        txt = args.out / f"{base}.txt"

        print(f"🎞️  [{i}/{len(chosen)}] → {mp4.name}")
        make_clip(args.video, mp4, seg.start, seg.end, vertical=not args.horizontal)

        srt.write_text(words_to_srt(seg.words, offset=seg.start), encoding="utf-8")
        txt.write_text(
            f"HOOK:\n{seg.hook}\n\nCAPTION:\n{seg.caption}\n\nHASHTAGS:\n{' '.join(seg.hashtags)}\n",
            encoding="utf-8",
        )
        summary.append({
            "file": mp4.name,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": round(seg.end - seg.start, 2),
            "score": round(seg.score, 2),
            "hook": seg.hook,
            "caption": seg.caption,
            "hashtags": seg.hashtags,
        })

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 64)
    print(f"✨ Terminé! {len(chosen)} reels dans: {args.out}/")
    print("   • .mp4 = clip vidéo prêt (sans sous-titres incrustés)")
    print("   • .srt = sous-titres prêts à importer dans CapCut/Submagic")
    print("   • .txt = hook + caption + hashtags à copier dans ton post")
    print(f"📄 Récap global: {args.out}/summary.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
