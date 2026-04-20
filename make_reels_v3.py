#!/usr/bin/env python3
"""
Générateur de Reels v3 — full stack viral.

Pipeline complet :
  1. WhisperX (timestamps précis au mot)
  2. ClipsAI ClipFinder (frontières sur fin de phrase)
  3. Pré-filtre rapide (texte + énergie audio) → top 20
  4. LLM Judge local (Ollama / llama3.2) → score viral hook/payoff/emotion/clarity/share
  5. LLM génère hook + caption + hashtags + start/end excerpts EXACTS du transcript
  6. Trim auto : on cale le clip pile sur la phrase du hook (start_excerpt → end_excerpt)
  7. ffmpeg : MP4 vertical 9:16 fond flouté + .srt + .txt

Usage:
    python make_reels_v3.py video.mp4
    python make_reels_v3.py video.mp4 --max-clips 5 --shortlist 20
    python make_reels_v3.py video.mp4 --no-llm     # pas de LLM, scoring heuristique seul
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from make_reels import (  # noqa: E402
    Word,
    Segment,
    AudioAnalysis,
    get_or_make_audio_analysis,
    score_segment,
    select_top_segments,
    enrich_segments,
    trim_segment_edges,
    make_clip,
    words_to_srt,
    slugify,
    extract_audio,
    ffprobe_duration,
)

from make_reels_v2 import (  # noqa: E402
    transcription_to_words,
    find_clips_clipsai,
    clips_to_segments,
    get_or_make_transcript,
)

import llm_judge  # noqa: E402
import viral_arcs  # noqa: E402
import viral_signals  # noqa: E402
import viral_patterns  # noqa: E402

# speaker_diarization est OPTIONNEL : si le module n'est pas installé
# (pas de HF_TOKEN, pas de pyannote dispo, etc.), on désactive juste
# les features dialogue-aware. Le pipeline reste fonctionnel.
try:
    import speaker_diarization  # noqa: E402
    _HAS_DIAR = True
except ImportError:
    speaker_diarization = None  # type: ignore[assignment]
    _HAS_DIAR = False


# ---------------------------------------------------------------------------
# Trim auto : aligner le clip vidéo pile sur le start/end excerpt du LLM
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() or c == " " else " " for c in text.lower()).split())


def _find_words_window(words: List[Word], excerpt: str, near_end: bool = False) -> Optional[int]:
    """Trouve l'index de début (ou de fin si near_end) de la fenêtre de mots qui matche
    le mieux 'excerpt'. Tolérant aux différences mineures (fuzzy).
    """
    excerpt_norm = _normalize(excerpt)
    if not excerpt_norm:
        return None
    target_tokens = excerpt_norm.split()
    if not target_tokens:
        return None
    n = len(target_tokens)

    word_texts = [_normalize(w.text) for w in words]
    if not word_texts:
        return None

    best_idx, best_score = None, 0.0
    rng = range(len(word_texts) - n + 1) if not near_end else range(len(word_texts) - n, -1, -1)
    for i in rng:
        window = " ".join(word_texts[i : i + n])
        if not window:
            continue
        score = SequenceMatcher(None, window, excerpt_norm).ratio()
        if score > best_score:
            best_score = score
            best_idx = i
            if score > 0.95:
                break
    if best_score < 0.55:
        return None
    return best_idx


_STRONG_END_PUNCT = {".", "!", "?"}


def _snap_end_to_sentence(words: List[Word], end_idx: int,
                          look_ahead: int = 8, look_back: int = 4) -> int:
    """Ajuste end_idx pour tomber pile sur la fin d'une phrase (. ! ?).

    On regarde d'abord en avant (jusqu'à `look_ahead` mots), puis en arrière
    (jusqu'à `look_back`). Si rien trouvé on garde end_idx.
    """
    if end_idx <= 0 or end_idx > len(words):
        return end_idx

    last = words[end_idx - 1].text.strip()
    if last and last[-1] in _STRONG_END_PUNCT:
        return end_idx

    for k in range(1, look_ahead + 1):
        i = end_idx + k - 1
        if i >= len(words):
            break
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1

    for k in range(1, look_back + 1):
        i = end_idx - 1 - k
        if i < 0:
            break
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1

    return end_idx


def _snap_start_to_sentence(words: List[Word], start_idx: int,
                            look_back: int = 6, look_ahead: int = 4) -> int:
    """Ajuste start_idx pour tomber juste APRÈS la fin d'une phrase précédente,
    ce qui garantit qu'on commence en début de phrase.
    """
    if start_idx <= 0 or start_idx >= len(words):
        return max(0, start_idx)

    for k in range(1, look_back + 1):
        i = start_idx - k
        if i < 0:
            break
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1

    for k in range(1, look_ahead + 1):
        i = start_idx + k - 1
        if i >= len(words) - 1:
            break
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1

    return start_idx


def trim_to_excerpts(
    seg: Segment,
    start_excerpt: str,
    end_excerpt: str,
    min_len: float = 15.0,
    max_len: float = 70.0,
) -> Segment:
    """Re-cadre les bornes du segment pour démarrer/finir aux excerpts donnés par le LLM.

    Garde-fous:
      - Snap sur ponctuation forte (. ! ?) côté start ET côté end
      - Si résultat < min_len, on étend (priorité : garder la fin punchline)
      - Si résultat > max_len, on tronque côté début
      - Si matching échoue, on retourne le segment original
    """
    words = seg.words
    if not words:
        return seg

    start_idx = _find_words_window(words, start_excerpt, near_end=False) if start_excerpt else None
    end_idx_start = _find_words_window(words, end_excerpt, near_end=True) if end_excerpt else None

    if start_idx is None and end_idx_start is None:
        return seg

    new_start = start_idx if start_idx is not None else 0
    if end_idx_start is not None:
        end_tokens = len(_normalize(end_excerpt).split())
        new_end = min(len(words), end_idx_start + end_tokens)
    else:
        new_end = len(words)

    if new_start >= new_end:
        return seg

    new_start = _snap_start_to_sentence(words, new_start)
    new_end = _snap_end_to_sentence(words, new_end)

    if new_start >= new_end:
        return seg

    duration = words[new_end - 1].end - words[new_start].start
    if duration < min_len:
        i = new_start
        while i > 0 and (words[new_end - 1].end - words[i - 1].start) < min_len:
            i -= 1
        new_start = _snap_start_to_sentence(words, i)
        j = new_end
        while j < len(words) and (words[j - 1].end - words[new_start].start) < min_len:
            j += 1
        new_end = _snap_end_to_sentence(words, j)

    duration = words[new_end - 1].end - words[new_start].start
    if duration > max_len:
        # Stratégie : on garde la PUNCHLINE (la fin) à tout prix. On rogne le début
        # SAUF si on perd la phrase d'ouverture du hook. On accepte un slack de +15%
        # avant de vraiment tronquer, parce que les vrais moments viraux dépassent
        # souvent un peu le max théorique.
        if duration <= max_len * 1.15:
            return Segment(
                start=words[new_start].start,
                end=words[new_end - 1].end,
                text=" ".join(w.text for w in words[new_start:new_end]).strip(),
                words=words[new_start:new_end],
                score=seg.score,
                reason=seg.reason,
                hook=seg.hook,
                caption=seg.caption,
                hashtags=seg.hashtags,
            )
        # Vraiment trop long : on rogne côté début, en s'assurant de tomber pile
        # APRÈS une fin de phrase (jamais en plein milieu).
        i = new_start
        while i < new_end - 1 and (words[new_end - 1].end - words[i].start) > max_len:
            i += 1
        new_start = _snap_start_to_sentence(words, i)

    new_words = words[new_start:new_end]
    if not new_words:
        return seg
    text = " ".join(w.text for w in new_words).strip()
    return Segment(
        start=new_words[0].start,
        end=new_words[-1].end,
        text=text,
        words=new_words,
        score=seg.score,
        reason=seg.reason,
        hook=seg.hook,
        caption=seg.caption,
        hashtags=seg.hashtags,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère des Reels viraux (ClipsAI + WhisperX + LLM judge local).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("reels_v3_output"))
    p.add_argument("--max-clips", type=int, default=5)
    p.add_argument("--shortlist", type=int, default=20,
                   help="Nb de candidats à passer au LLM (défaut: 20)")
    p.add_argument("--min-len", type=float, default=15.0)
    p.add_argument("--max-len", type=float, default=90.0,
                   help="Durée max d'un clip en s (défaut: 90 — sweet spot Reels). "
                        "Le viral arc detector peut générer des clips plus longs si "
                        "la réaction le justifie (cap à max_len * 1.15).")
    p.add_argument("--model", default="small.en",
                   help="Modèle Whisper. Par défaut 'small.en' (EN-only, "
                        "+50%% rapide et +précis que 'small' multilingue). "
                        "Options EN-only: tiny.en, base.en, small.en, medium.en. "
                        "Multilingue: tiny, base, small, medium, large-v3.")
    p.add_argument("--lang", default="en",
                   help="Langue source (ISO 639-1). Défaut 'en'. "
                        "Mettre 'auto' pour détection automatique (-30%% vitesse).")
    p.add_argument("--llm-model", default=llm_judge.DEFAULT_MODEL,
                   help=f"Modèle Ollama (défaut: {llm_judge.DEFAULT_MODEL})")
    p.add_argument("--horizontal", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-llm", action="store_true",
                   help="Désactive le LLM judge (scoring heuristique seul)")
    p.add_argument("--cache-dir", type=Path, default=Path("transcripts"))
    # ====== Détection viralité avancée ======
    p.add_argument("--no-viral-arcs", action="store_true",
                   help="Désactive le viral arc detector (clips construits "
                        "autour des pics audio).")
    p.add_argument("--setup-max", type=float, default=60.0,
                   help="Durée max du 'setup' avant un pic viral en secondes "
                        "(défaut: 60). Augmenter pour les anecdotes longues.")
    p.add_argument("--after-laugh", type=float, default=8.0,
                   help="Durée à garder après le pic (réaction/rire). Défaut: 8s.")
    p.add_argument("--emotion", action="store_true",
                   help="Active la détection émotion par seconde "
                        "(speechbrain wav2vec2 IEMOCAP — lourd mais précis).")
    p.add_argument("--diarization", action="store_true",
                   help="Active la diarisation (pyannote-audio). "
                        "Nécessite HF_TOKEN + accepter la licence du modèle.")
    p.add_argument("--prosody", action="store_true",
                   help="Active l'analyse prosodie (librosa pitch + energy).")
    p.add_argument("--all-signals", action="store_true",
                   help="Raccourci pour --emotion --diarization --prosody.")
    # ====== v3.1 — Speaker-aware + patterns viraux (ON par défaut) ======
    p.add_argument("--no-speakers", action="store_true",
                   help="Désactive la diarization speaker-aware "
                        "(par défaut on essaie WhisperX/pyannote, fallback MFCC).")
    p.add_argument("--no-patterns", action="store_true",
                   help="Désactive les patterns viraux (laughter detector, "
                        "punchline pattern, duo dynamic).")
    p.add_argument("--n-speakers", type=int, default=2,
                   help="Nb de speakers attendu pour le fallback heuristique "
                        "(défaut: 2 = duo podcast).")
    p.add_argument("--force-fallback-diar", action="store_true",
                   help="Force le fallback heuristique MFCC+KMeans même si "
                        "HF_TOKEN est dispo (utile pour tester).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists() and not args.dry_run:
        print(f"❌ Vidéo introuvable: {args.video}")
        sys.exit(1)

    use_llm = not args.no_llm
    if use_llm and not llm_judge.is_ollama_available(args.llm_model):
        print(f"⚠️  Ollama indisponible ou modèle '{args.llm_model}' absent.")
        print("   Lance: brew services start ollama && ollama pull " + args.llm_model)
        print("   → on continue en mode heuristique seul.")
        use_llm = False

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Activation des signaux avancés
    if args.all_signals:
        args.emotion = True
        args.diarization = True
        args.prosody = True

    duration = ffprobe_duration(args.video) if args.video.exists() else 0
    fmt = "16:9" if args.horizontal else "9:16 fond flouté"
    extra_engines = []
    if not args.no_viral_arcs:
        extra_engines.append("viral-arcs")
    if not args.no_speakers:
        extra_engines.append("speakers")
    if not args.no_patterns:
        extra_engines.append("patterns(laugh+punch+duo)")
    if args.emotion:
        extra_engines.append("emotion")
    if args.diarization:
        extra_engines.append("diarization-adv")
    if args.prosody:
        extra_engines.append("prosody")

    print("=" * 64)
    print(f"🎬 Vidéo:    {args.video.name}")
    if duration:
        print(f"   Durée:    {int(duration)}s")
    print(f"🎯 Cible:    {args.max_clips} reels (shortlist {args.shortlist}), {int(args.min_len)}-{int(args.max_len)}s, {fmt}")
    engine_str = "ClipsAI + WhisperX"
    if use_llm:
        engine_str += f" + LLM {args.llm_model}"
    if extra_engines:
        engine_str += " + " + ", ".join(extra_engines)
    print(f"🧠 Engine:   {engine_str}")
    print("=" * 64)

    # 1) Transcription
    lang = None if args.lang.lower() == "auto" else args.lang.lower()
    # Garde-fou : si on demande un modèle .en, on FORCE language='en'
    if args.model.endswith(".en") and lang and lang != "en":
        print(f"⚠️  Modèle '{args.model}' est EN-only mais --lang={lang}. "
              f"Bascule sur language='en'.")
        lang = "en"
    transcription = get_or_make_transcript(args.video, cache_dir, args.model, language=lang)
    words = transcription_to_words(transcription)

    # 1b) Diarization speaker-aware (ON par défaut, fallback heuristique
    # si pas de HF_TOKEN). Modifie words in-place pour leur donner un .speaker.
    diar = None
    if not args.no_speakers and _HAS_DIAR and args.video.exists():
        audio_wav = cache_dir / f"{args.video.stem}.wav"
        if audio_wav.exists():
            diar = speaker_diarization.diarize(
                audio_wav,
                cache_dir=cache_dir,
                stem=args.video.stem,
                fallback_speakers=args.n_speakers,
                force_fallback=args.force_fallback_diar,
            )
            if diar:
                n_assigned = speaker_diarization.assign_speakers_to_words(words, diar)
                pct = (100.0 * n_assigned / max(1, len(words)))
                print(f"   👥 {n_assigned}/{len(words)} mots attribués "
                      f"({pct:.0f}%) à {len(diar.speakers)} speakers "
                      f"[{diar.source}]")

    # 2) Audio analysis (RMS + pics + rires soutenus)
    audio_analysis = get_or_make_audio_analysis(args.video, cache_dir) if args.video.exists() else None

    # 2b) Signaux audio avancés (emotion, diarisation, prosodie) — optionnels
    advanced_signals = viral_signals.ViralSignals()
    if (args.emotion or args.diarization or args.prosody) and args.video.exists():
        audio_wav = cache_dir / f"{args.video.stem}.wav"
        if audio_wav.exists():
            advanced_signals = viral_signals.compute_all_signals(
                audio_wav,
                duration_seconds=int(duration) if duration else 0,
                cache_dir=cache_dir,
                video_stem=args.video.stem,
                enable_emotion=args.emotion,
                enable_diarization=args.diarization,
                enable_prosody=args.prosody,
            )

    # 2c) Patterns viraux : laughter detector + punchlines + duo dynamic
    # ON par défaut, désactivable via --no-patterns. Léger en CPU/RAM.
    patterns = viral_patterns.ViralPatterns()
    if not args.no_patterns and args.video.exists():
        audio_wav = cache_dir / f"{args.video.stem}.wav"
        if audio_wav.exists():
            patterns = viral_patterns.compute_all_patterns(
                audio_wav,
                words=words,
                total_duration=int(duration) if duration else 0,
                cache_dir=cache_dir,
                stem=args.video.stem,
                enable_laughter=True,
                enable_punchlines=bool(diar),  # nécessite speakers
                enable_duo=bool(diar),
            )

    # 3) Sources de candidats : ClipsAI (topiques) + viral arcs (émotionnels)
    clip_ranges = find_clips_clipsai(transcription, args.min_len, args.max_len)
    clipsai_candidates = clips_to_segments(clip_ranges, words)

    arc_candidates: List[Segment] = []
    if not args.no_viral_arcs and audio_analysis:
        print("🌋 Construction des viral arcs autour des pics audio…")
        arc_candidates = viral_arcs.build_viral_arc_candidates(
            words, audio_analysis,
            min_len=args.min_len,
            max_len=args.max_len,
            setup_max=args.setup_max,
            after_laugh=args.after_laugh,
        )
        print(f"   ✅ {len(arc_candidates)} viral arcs construits")

    candidates = viral_arcs.merge_arcs_with_clipsai(arc_candidates, clipsai_candidates)
    if not candidates:
        print("❌ Aucun clip détecté.")
        sys.exit(1)
    print(f"📦 Total candidats: {len(candidates)} "
          f"({len(arc_candidates)} arcs + {len(clipsai_candidates)} clipsai, fusionnés)")

    # 4) Pré-filtre heuristique → shortlist (avec bonus signaux + patterns)
    for c in candidates:
        sig_bonus, _ = viral_signals.signals_bonus_for_segment(c.start, c.end, advanced_signals)
        pat_bonus, _ = viral_patterns.patterns_bonus_for_segment(c.start, c.end, patterns)
        c._signals_bonus = sig_bonus  # type: ignore[attr-defined]
        c._patterns_bonus = pat_bonus  # type: ignore[attr-defined]
        # Annotate segment with speaker info (utilisé pour scoring + LLM)
        if diar and _HAS_DIAR:
            spks, turns = speaker_diarization.speakers_in_segment(words, c.start, c.end)
            c.speakers = spks
            c.speaker_turns = turns

    shortlist = select_top_segments(candidates, args.shortlist, audio=audio_analysis)
    # Réinjecte les bonus après le scoring heuristique (qui a écrasé .score)
    for s in shortlist:
        sig_bonus = getattr(s, "_signals_bonus", 0.0)
        pat_bonus = getattr(s, "_patterns_bonus", 0.0)
        total_bonus = sig_bonus + pat_bonus
        if total_bonus > 0:
            s.score += total_bonus
            extras = []
            if sig_bonus > 0:
                extras.append(f"signals+{sig_bonus:.1f}")
            if pat_bonus > 0:
                extras.append(f"patterns+{pat_bonus:.1f}")
            s.reason += " + " + " ".join(extras)
    shortlist.sort(key=lambda s: s.score, reverse=True)
    shortlist = shortlist[: args.shortlist]
    print(f"\n📋 Shortlist (heuristique + signaux + patterns): {len(shortlist)} candidats\n")

    # 5) LLM judge sur la shortlist (dialogue-aware si diarization OK)
    if use_llm:
        print(f"🤖 LLM judge ({args.llm_model}) en cours "
              f"{'(dialogue-aware)' if diar else ''}…")
        scored: List[Tuple[Segment, llm_judge.ViralScore]] = []
        for i, seg in enumerate(shortlist, 1):
            # Préparer les signaux dialogue-aware pour le LLM
            seg_dialogue = ""
            seg_has_laugh = False
            seg_has_punch = False
            seg_duo_int = 0.0

            if diar and _HAS_DIAR and seg.speakers:
                seg_dialogue = speaker_diarization.words_to_dialogue(seg.words)
            if patterns.laughter and patterns.laughter.laugh_score_per_second:
                track = patterns.laughter.laugh_score_per_second
                window = track[int(seg.start): min(int(seg.end), len(track))]
                seg_has_laugh = bool(window) and max(window) >= 0.55
            if patterns.punchlines:
                seg_has_punch = any(
                    h.setup_start >= seg.start - 1 and h.reaction_end <= seg.end + 1
                    for h in patterns.punchlines
                )
            if patterns.duo and patterns.duo.duo_intensity_per_second:
                track = patterns.duo.duo_intensity_per_second
                window = track[int(seg.start): min(int(seg.end), len(track))]
                if window:
                    seg_duo_int = sum(window) / len(window)

            score = llm_judge.score_clip(
                seg.text, model=args.llm_model,
                dialogue=seg_dialogue,
                speaker_turns=seg.speaker_turns,
                has_laughter=seg_has_laugh,
                has_punchline=seg_has_punch,
                duo_intensity=seg_duo_int,
            )
            if score is None:
                print(f"   [{i:2d}/{len(shortlist)}] ❌ skip (parse fail)")
                continue

            # Score final = LLM (0-10) * 10 + signaux audio (0-15) + patterns (0-20)
            # Les bonus pèsent lourd car ce sont des signaux RÉELS (réaction du
            # public, dynamique de dialogue), pas seulement du texte.
            sig_bonus, sig_reason = viral_signals.signals_bonus_for_segment(
                seg.start, seg.end, advanced_signals,
            )
            pat_bonus, pat_reason = viral_patterns.patterns_bonus_for_segment(
                seg.start, seg.end, patterns,
            )
            seg.score = score.overall * 10 + sig_bonus + pat_bonus
            seg.reason = (
                f"LLM={score.overall:.1f} (hook {score.hook:.0f}/payoff {score.payoff:.0f}"
                f"/emo {score.emotion:.0f}/clar {score.clarity:.0f}/share {score.shareability:.0f}) "
                f"— {score.reasoning}"
            )
            if sig_bonus > 0:
                seg.reason += f" | signals+{sig_bonus:.1f} [{sig_reason}]"
            if pat_bonus > 0:
                seg.reason += f" | patterns+{pat_bonus:.1f} [{pat_reason}]"
            scored.append((seg, score))
            extras_parts = []
            if sig_bonus > 0:
                extras_parts.append(f"sig+{sig_bonus:.0f}")
            if pat_bonus > 0:
                extras_parts.append(f"pat+{pat_bonus:.0f}")
            extras = (" " + " ".join(extras_parts)) if extras_parts else ""
            print(f"   [{i:2d}/{len(shortlist)}] {score.overall:4.1f}/10  hook={score.hook:.0f} share={score.shareability:.0f}{extras}  {score.reasoning[:55]}")

        # Tri par score combiné (LLM + signals), pas LLM seul
        scored.sort(key=lambda x: x[0].score, reverse=True)

        def _too_close(seg: Segment, picked: List[Segment]) -> bool:
            return any(
                abs(seg.start - c.start) < 90
                or not (seg.end <= c.start or seg.start >= c.end)
                for c in picked
            )

        # Sélection en 2 passes :
        #   Passe 1 : meilleur clip de chaque angle (diversification émotionnelle)
        #   Passe 2 : on remplit avec les meilleurs scores restants
        chosen: List[Segment] = []
        seen_angles = set()
        for seg, vs in scored:
            if vs.angle in seen_angles:
                continue
            if _too_close(seg, chosen):
                continue
            chosen.append(seg)
            seen_angles.add(vs.angle)
            if len(chosen) >= args.max_clips:
                break

        if len(chosen) < args.max_clips:
            for seg, _ in scored:
                if seg in chosen:
                    continue
                if _too_close(seg, chosen):
                    continue
                chosen.append(seg)
                if len(chosen) >= args.max_clips:
                    break

        if seen_angles:
            print(f"\n🎭 Angles couverts: {', '.join(sorted(seen_angles))}")
    else:
        chosen = shortlist[: args.max_clips]

    # 6) Génération de contenu (LLM hook/caption/hashtags + trim aux excerpts)
    if use_llm:
        print(f"\n✍️  LLM génère hook/caption/hashtags + trim auto…")
        used_hooks: List[str] = []
        for i, seg in enumerate(chosen, 1):
            content = llm_judge.generate_content(
                seg.text, model=args.llm_model, avoid_hooks=used_hooks,
            )
            if content:
                used_hooks.append(content.hook)
                seg.hook = content.hook
                seg.caption = content.caption
                seg.hashtags = content.hashtags
                seg = trim_to_excerpts(
                    seg, content.start_excerpt, content.end_excerpt,
                    min_len=args.min_len, max_len=args.max_len,
                )
                chosen[i - 1] = seg
            else:
                print(f"   [{i}] ⚠️  LLM content fail → fallback regex")
        # Re-applique nos fallbacks pour ce qui manque
        for seg in chosen:
            if not seg.hook or not seg.caption:
                enrich_segments([seg])
    else:
        chosen = [trim_segment_edges(s) for s in chosen]
        enrich_segments(chosen)

    # 7) Affichage
    print(f"\n✅ {len(chosen)} clips retenus :\n")
    for i, seg in enumerate(chosen, 1):
        print(f"  [{i}] {seg.start:7.1f}s → {seg.end:7.1f}s  ({seg.end - seg.start:.1f}s)  {seg.reason or f'score={seg.score:.1f}'}")
        if seg.hook:
            print(f"      🎯 HOOK     : {seg.hook}")
        if seg.caption:
            print(f"      📝 CAPTION  : {seg.caption}")
        if seg.hashtags:
            print(f"      🏷️  HASHTAGS: {' '.join(seg.hashtags)}")
        snippet = seg.text[:140].replace("\n", " ")
        print(f"      💬          « {snippet}{'…' if len(seg.text) > 140 else ''} »")
        print()

    if args.dry_run:
        print("🛑 --dry-run: pas de découpage vidéo.")
        return

    # 8) Génération vidéo
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, seg in enumerate(chosen, 1):
        slug = slugify(seg.hook or seg.text, max_len=35)
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
            "reason": seg.reason,
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
    print("=" * 64)


if __name__ == "__main__":
    main()
