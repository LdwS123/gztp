#!/usr/bin/env python3
"""
Générateur de Reels viraux à partir d'un podcast (vidéo).

Tu fournis JUSTE la vidéo. Le script:
  1. Transcrit l'audio (Whisper local, gratuit)
  2. Détecte les meilleurs moments (scoring viral)
  3. Découpe en clips courts verticaux 9:16 (fond flouté + speaker centré)
  4. Génère pour chaque clip un fichier .txt avec hook + caption + hashtags

Tu rajouteras les sous-titres après dans ton outil préféré (CapCut, Submagic…).

Usage:
    python make_reels.py video.mp4
    python make_reels.py video.mp4 --max-clips 12 --min-len 15 --max-len 40
    python make_reels.py video.mp4 --dry-run               # juste l'analyse
    python make_reels.py video.mp4 --transcript subs.srt   # si tu as déjà une transcription
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Heuristiques de score viral
# ---------------------------------------------------------------------------

VIRAL_TRIGGERS = {
    # ====== Emotion / reaction ======
    "amazing": 3, "incredible": 3, "insane": 4, "crazy": 4, "wild": 3,
    "shocked": 3, "shocking": 3, "blown": 3, "mind-blown": 4, "unbelievable": 4,
    "ridiculous": 3, "absurd": 3, "savage": 4, "brutal": 3, "wholesome": 2,
    "love": 1, "hate": 2, "scared": 2, "stressful": 2, "stressed": 2,
    "obsessed": 3, "addicted": 2, "afraid": 2, "embarrassing": 3,
    "humiliating": 3, "terrifying": 3, "devastating": 3,
    # ====== Strong claims / storytelling ======
    "secret": 3, "truth": 3, "honestly": 2, "literally": 1, "actually": 1,
    "never": 2, "nobody": 3, "everyone": 1, "always": 1,
    "discovered": 2, "realized": 2, "figured out": 3, "turns out": 3,
    "the thing is": 3, "here's the thing": 4, "the real reason": 4,
    "biggest mistake": 4, "what nobody": 4, "the catch": 3,
    "plot twist": 4, "the worst part": 4, "the best part": 3,
    # ====== Hooks / commands ======
    "listen": 2, "imagine": 2, "watch": 2, "stop": 2,
    "let me tell you": 4, "you won't believe": 5, "wait until": 3,
    "did you know": 3, "what if": 3, "hot take": 4, "controversial": 3,
    # ====== Money / business / tech ======
    "million": 3, "billion": 3, "money": 2, "rich": 2, "broke": 2,
    "wealthy": 2, "investor": 1, "startup": 2, "founder": 2,
    "raised": 2, "exit": 2, "scam": 3, "scammed": 3, "viral": 2,
    "openai": 2, "youtube": 1, "tiktok": 1, "instagram": 1,
    "ceo": 1, "elon": 2, "trump": 2,
    # ====== Profanity (high virality on EN content) ======
    "fuck": 3, "fucking": 3, "shit": 2, "damn": 2, "bullshit": 3,
    "asshole": 2, "bitch": 2, "wtf": 3,
}

# Phrases d'ouverture qui font un BON début de reel (EN only)
HOOK_STARTERS = (
    "you won't", "you wont", "listen", "imagine", "watch this", "stop",
    "here's", "heres", "the secret", "the truth", "the reason",
    "what if", "what nobody", "did you know", "let me tell you",
    "nobody talks", "everyone thinks", "i've never", "ive never",
    "the craziest", "the biggest", "the worst", "the best",
    "i was", "i am", "i'm", "im", "when i", "before i",
    "it's fucking", "its fucking", "this is",
    "okay so", "look", "real talk", "not gonna lie",
    "in my opinion", "hot take", "controversial",
    "people don't", "people dont",
)

# Connecteurs en début de phrase = mid-pensée → mauvais début de reel (EN only)
BAD_STARTERS = (
    "and", "or", "so", "but", "because", "however", "therefore",
    "yeah", "yes", "no", "okay", "ok", "well", "uh", "um", "uhm",
    "that's why", "for example", "like for", "i mean", "you know",
    "kind of", "sort of", "basically", "anyway", "anyways",
)


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # ex: "SPEAKER_00", "SPEAKER_01"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word]
    score: float = 0.0
    reason: str = ""
    hook: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)  # speakers présents
    speaker_turns: int = 0                              # nb de changements
    dialogue: str = ""                                  # transcript "SPK_A: ... SPK_B: ..."

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Helpers shell / ffmpeg
# ---------------------------------------------------------------------------

def run(cmd: List[str], quiet: bool = True) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        check=False,
    )
    if result.returncode != 0:
        if quiet and result.stderr:
            sys.stderr.write(result.stderr.decode(errors="replace")[-2000:])
        sys.exit(f"❌ Commande échouée: {' '.join(cmd[:4])}")


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def extract_audio(video: Path, audio_out: Path) -> None:
    print(f"🎧 Extraction audio…")
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_out),
    ])


# ---------------------------------------------------------------------------
# Transcription (Whisper local)
# ---------------------------------------------------------------------------

def transcribe_whisper(audio: Path, model_size: str = "base") -> Tuple[str, List[Word]]:
    print(f"🗣️  Transcription Whisper ({model_size}) — ça peut prendre quelques minutes…")
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio),
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    print(f"   Langue détectée: {info.language} (proba {info.language_probability:.2f})")

    words: List[Word] = []
    last_log = 0.0
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            txt = (w.word or "").strip()
            if not txt:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), text=txt))
        # Petit log de progression toutes les 30s d'audio transcrit
        if seg.end - last_log > 30:
            print(f"   … {seg.end:.0f}s transcrits", flush=True)
            last_log = seg.end

    print(f"   ✅ {len(words)} mots transcrits")
    return info.language, words


# ---------------------------------------------------------------------------
# Analyse audio : énergie par seconde + détection de pics (rires, surprise…)
# ---------------------------------------------------------------------------

@dataclass
class AudioAnalysis:
    """Énergie audio (RMS) par seconde, et liste des secondes 'peak'."""
    duration: float
    energies: List[float]              # 1 valeur par seconde
    peak_seconds: List[int]            # secondes où l'énergie dépasse le seuil
    laughter_seconds: List[int]        # secondes avec pic 'rire' (long & fort)
    mean: float
    std: float


def analyze_audio_energy(video: Path) -> AudioAnalysis:
    """Extrait l'audio à 8kHz mono int16, calcule l'énergie RMS par seconde,
    puis identifie les pics (= moments de réaction forte: rires, surprise…)."""
    print("🔊 Analyse de l'énergie audio (détection des pics de réaction)…")

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-vn", "-f", "s16le", "-ac", "1", "-ar", "8000", "-",
        ],
        check=True, capture_output=True,
    )
    raw = proc.stdout
    sr = 8000
    bytes_per_sample = 2
    samples_per_window = sr  # 1 seconde

    energies: List[float] = []
    for i in range(0, len(raw), samples_per_window * bytes_per_sample):
        chunk = raw[i: i + samples_per_window * bytes_per_sample]
        if len(chunk) < samples_per_window * bytes_per_sample:
            break
        n = len(chunk) // 2
        samples = struct.unpack(f"<{n}h", chunk)
        # RMS
        s = 0.0
        for v in samples:
            s += v * v
        rms = math.sqrt(s / n)
        energies.append(rms)

    if not energies:
        return AudioAnalysis(0.0, [], [], [], 0.0, 0.0)

    # Stats
    mean = sum(energies) / len(energies)
    var = sum((e - mean) ** 2 for e in energies) / len(energies)
    std = math.sqrt(var)

    # Seuil : pic = moment > mean + 1.0*std (modéré pour capturer les vraies réactions)
    threshold = mean + 1.0 * std
    peak_seconds = [i for i, e in enumerate(energies) if e > threshold]

    # Rire / réaction forte = pic SOUTENU (au moins 2 secondes consécutives au-dessus
    # de mean + 1.5*std). C'est le signal le plus fort de virabilité.
    high_threshold = mean + 1.5 * std
    laughter_seconds: List[int] = []
    i = 0
    while i < len(energies):
        if energies[i] > high_threshold:
            run_start = i
            while i < len(energies) and energies[i] > high_threshold:
                i += 1
            run_len = i - run_start
            if run_len >= 2:
                laughter_seconds.extend(range(run_start, run_start + run_len))
        else:
            i += 1

    print(f"   Énergie moy={mean:.0f} std={std:.0f}")
    print(f"   ⚡ {len(peak_seconds)} secondes 'peak' (réaction)")
    print(f"   😂 {len(laughter_seconds)} secondes 'fort & soutenu' (rire/surprise)")

    return AudioAnalysis(
        duration=float(len(energies)),
        energies=energies,
        peak_seconds=peak_seconds,
        laughter_seconds=laughter_seconds,
        mean=mean,
        std=std,
    )


def detect_scene_changes(video: Path, threshold: float = 0.3) -> List[float]:
    """Détecte les changements de scène (cuts éditoriaux). Souvent les cuts marquent
    des moments importants dans un montage de podcast."""
    print(f"🎬 Détection des changements de scène (seuil {threshold})…")
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "info", "-i", str(video),
                "-filter:v", f"select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("   (timeout — analyse de scène ignorée)")
        return []

    times: List[float] = []
    for line in proc.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            times.append(float(m.group(1)))
    print(f"   {len(times)} changements de scène détectés")
    return times


def get_or_make_audio_analysis(video: Path, cache_dir: Path) -> Optional[AudioAnalysis]:
    cache = cache_dir / f"{video.stem}.audio.json"
    if cache.exists():
        print(f"📂 Analyse audio en cache: {cache.name}")
        d = json.loads(cache.read_text())
        return AudioAnalysis(**d)

    if not video.exists():
        return None

    a = analyze_audio_energy(video)
    cache.write_text(json.dumps(asdict(a)))
    print(f"💾 Cache audio: {cache}")
    return a


# ---------------------------------------------------------------------------
# Parseurs de transcription externes (optionnel via --transcript)
# ---------------------------------------------------------------------------

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_vtt(path: Path) -> List[Word]:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    blocks: List[Tuple[float, float, str]] = []
    for match in _TS_RE.finditer(raw):
        start = _ts_to_sec(*match.groups()[:4])
        end = _ts_to_sec(*match.groups()[4:])
        rest = raw[match.end():]
        text_lines: List[str] = []
        for line in rest.split("\n")[1:]:
            if not line.strip():
                break
            text_lines.append(line.strip())
        text = re.sub(r"<[^>]+>|\{[^}]+\}", "", " ".join(text_lines))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            blocks.append((start, end, text))

    words: List[Word] = []
    for start, end, text in blocks:
        toks = text.split()
        if not toks:
            continue
        per = max((end - start) / len(toks), 0.05)
        for i, tok in enumerate(toks):
            words.append(Word(start=start + i * per, end=start + (i + 1) * per, text=tok))
    return words


def parse_whisper_json(path: Path) -> List[Word]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif "words" in data:
        items = data["words"]
    elif "segments" in data:
        items = [w for seg in data["segments"] for w in seg.get("words", [])]
    else:
        sys.exit(f"❌ Format JSON non reconnu: {path}")

    words: List[Word] = []
    for w in items:
        text = (w.get("text") or w.get("word") or "").strip()
        if not text:
            continue
        try:
            words.append(Word(start=float(w["start"]), end=float(w["end"]), text=text))
        except (KeyError, TypeError, ValueError):
            continue
    return words


def load_external_transcript(path: Path) -> List[Word]:
    suffix = path.suffix.lower()
    if suffix in (".srt", ".vtt"):
        return parse_srt_vtt(path)
    if suffix == ".json":
        return parse_whisper_json(path)
    sys.exit(f"❌ Format non supporté: {suffix}")


# ---------------------------------------------------------------------------
# Détection des candidats + scoring
# ---------------------------------------------------------------------------

def _is_clean_start_word(word_text: str) -> bool:
    """Un démarrage 'propre' = pas un connecteur / filler / acquiescement."""
    clean = word_text.strip().lower().rstrip(",.")
    bad_first = {
        "and", "or", "but", "so", "because", "however", "therefore",
        "et", "ou", "mais", "donc", "alors", "car", "parce", "puis",
        "yeah", "yes", "no", "okay", "ok", "well", "uh", "um", "euh",
        "ouais", "non", "bah", "ben", "bon", "like",
    }
    return clean not in bad_first


def words_to_candidates(
    words: List[Word],
    min_len: float,
    max_len: float,
    pause_threshold: float = 0.8,
) -> List[Segment]:
    """Construit des segments candidats avec des FRONTIÈRES STRICTES :
      • début = soit le premier mot, soit après une grosse pause (≥ 0.8s)
                ET le mot précédent finit par .!? ou la pause est ≥ 1.2s
      • fin = doit finir sur .!? OU être suivi d'une pause ≥ 0.8s
      • on rejette tout segment qui commence par un mot 'sale' (and/so/yeah…)
    """
    if not words:
        return []

    # Détection des frontières "propres"
    boundaries: List[int] = [0]
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        prev_text = words[i - 1].text.strip()
        prev_ends_sentence = prev_text.endswith((".", "!", "?", "…"))
        # Frontière acceptable si:
        #   pause >= 1.2s  OU  (pause >= 0.8s ET phrase précédente terminée)
        if gap >= 1.2 or (gap >= pause_threshold and prev_ends_sentence):
            boundaries.append(i)

    candidates: List[Segment] = []
    seen: set = set()

    for start_idx in boundaries:
        # Skip si le mot de départ est "sale"
        if not _is_clean_start_word(words[start_idx].text):
            continue

        for end_idx in range(start_idx + 1, len(words) + 1):
            duration = words[end_idx - 1].end - words[start_idx].start
            if duration < min_len:
                continue
            if duration > max_len:
                # IMPORTANT: on ne `break` pas — on a juste dépassé pour CE start_idx,
                # mais d'autres end_idx pourraient encore donner des candidats valides
                # si on continue de chercher une fin propre dans la fenêtre.
                # On stoppe quand même cette branche, mais on autorise un slack de
                # +20% pour ne pas couper une punchline qui dépasse de 5-10s.
                if duration > max_len * 1.20:
                    break
                # Sinon : on accepte si la fin est PROPRE (sinon on continue à chercher)
                last_word = words[end_idx - 1].text
                if not last_word.endswith((".", "!", "?", "…", '"')):
                    continue

            # Une fin propre = ponctuation finale OU prochaine pause ≥ 0.8s
            last_word = words[end_idx - 1].text
            ends_clean = last_word.endswith((".", "!", "?", "…", '"'))
            if not ends_clean and end_idx < len(words):
                next_gap = words[end_idx].start - words[end_idx - 1].end
                if next_gap < pause_threshold:
                    continue

            key = (start_idx, end_idx)
            if key in seen:
                continue
            seen.add(key)

            seg_words = words[start_idx:end_idx]
            text = re.sub(r"\s+", " ", " ".join(w.text for w in seg_words)).strip()
            candidates.append(
                Segment(
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    text=text,
                    words=seg_words,
                )
            )
    return candidates


def trim_segment_edges(seg: Segment) -> Segment:
    """Nettoie les premiers/derniers mots 'sales' du segment (yeah, you know, uh…).

    Modifie aussi seg.start / seg.end pour que le découpage vidéo s'aligne.
    """
    words = seg.words[:]

    # Mots/expressions à retirer s'ils sont au DÉBUT
    leading_junk_words = {
        "uh", "um", "euh", "heu", "yeah", "yes", "yep", "no", "nope",
        "okay", "ok", "well", "so", "and", "but", "or", "like",
        "ouais", "non", "bah", "ben", "bon",
    }

    # Trim leading
    while words:
        first = words[0].text.strip().lower().rstrip(",.")
        if first in leading_junk_words:
            words = words[1:]
        else:
            break

    # Trim trailing — mots peu utiles à la toute fin (mais on garde la ponctuation)
    trailing_junk_words = {
        "uh", "um", "euh", "yeah", "you", "i", "and", "but", "or", "so",
        "the", "a", "to", "of", "in", "on", "for", "with",
    }
    while len(words) > 5:
        last = words[-1].text.strip().lower().rstrip(",.!?")
        if last in trailing_junk_words:
            words = words[:-1]
        else:
            break

    if not words:
        return seg

    text = re.sub(r"\s+", " ", " ".join(w.text for w in words)).strip()
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=words,
        score=seg.score,
        reason=seg.reason,
        hook=seg.hook,
        caption=seg.caption,
        hashtags=seg.hashtags,
    )


def score_segment(seg: Segment, audio: Optional[AudioAnalysis] = None) -> Tuple[float, str]:
    """Scoring orienté Insta/TikTok virality.

    Principes:
      - PUNIR sévèrement les segments qui commencent mid-pensée (and, so, but…)
      - RÉCOMPENSER les vrais hooks d'ouverture
      - Récompenser densité de mots viraux, questions, gros mots, chiffres
      - Pénaliser les longs blablas mous (fillers, faible densité)
    """
    text_lower = seg.text.lower()
    score = 0.0
    reasons: List[str] = []

    # ── 1. OUVERTURE (les 3 premières secondes décident tout sur TikTok) ──
    first_3s_words = [w.text for w in seg.words if w.start - seg.start <= 3.5]
    opener = " ".join(first_3s_words).lower().strip()

    # Pénalité TRÈS forte si le clip commence mid-pensée
    bad_start = False
    for bad in BAD_STARTERS:
        if opener.startswith(bad + " ") or opener == bad:
            score -= 8
            reasons.append(f"BAD_START({bad})")
            bad_start = True
            break

    if not bad_start:
        for hook in HOOK_STARTERS:
            if opener.startswith(hook):
                score += 6
                reasons.append(f"HOOK({hook})")
                break

    # Bonus si l'ouverture contient un trigger fort
    opening_triggers = sum(
        1 for trig in VIRAL_TRIGGERS
        if re.search(rf"\b{re.escape(trig)}\w*\b", opener)
    )
    if opening_triggers:
        score += 3 * opening_triggers
        reasons.append(f"open_trig={opening_triggers}")

    # ── 2. Longueur (15-35s = sweet spot TikTok) ──
    if 15 <= seg.duration <= 25:
        length_score = 5.0       # ultra-court = top
    elif 25 < seg.duration <= 35:
        length_score = 4.0
    elif 35 < seg.duration <= 45:
        length_score = 2.0
    else:
        length_score = 0.0
    score += length_score
    reasons.append(f"len={seg.duration:.0f}s(+{length_score:.1f})")

    # ── 3. Triggers viraux dans tout le segment ──
    trigger_score = 0.0
    trigger_hits = 0
    for trigger, weight in VIRAL_TRIGGERS.items():
        matches = len(re.findall(rf"\b{re.escape(trigger)}\w*\b", text_lower))
        if matches:
            trigger_score += weight * min(matches, 2)  # cap par trigger
            trigger_hits += matches
    score += trigger_score
    if trigger_hits:
        reasons.append(f"triggers={trigger_hits}(+{trigger_score:.1f})")

    # ── 4. Engagement: questions et exclamations ──
    questions = text_lower.count("?")
    if questions:
        score += 2.5 * min(questions, 3)
        reasons.append(f"?={questions}")
    exclam = text_lower.count("!")
    if exclam:
        score += 1.5 * min(exclam, 2)
        reasons.append(f"!={exclam}")

    # ── 5. Storytelling (je / I / un nom) → contenu personnel ==> partage ──
    storytelling_words = (
        " i ", " i'm ", " i was ", " i had ", " i saw ", " i felt ", " i think ",
        " i realized ", " i discovered ", " i told ", " when i ", " before i ",
        " je ", " j'ai ", " j'étais ", " quand j ",
    )
    story_hits = sum(text_lower.count(w) for w in storytelling_words)
    if story_hits >= 2:
        score += 2
        reasons.append(f"story={story_hits}")

    # ── 6. Chiffres concrets ──
    numbers = re.findall(r"\b\d+\b", seg.text)
    if numbers:
        # Les BIG numbers (> 10) sont plus virals
        big_nums = sum(1 for n in numbers if int(n) >= 10)
        score += min(len(numbers), 3) + big_nums
        reasons.append(f"nums={len(numbers)}")

    # ── 7. Densité de parole ──
    wpm = len(seg.words) / max(seg.duration, 1) * 60
    if 140 <= wpm <= 220:
        score += 2
    elif wpm < 100:
        score -= 2  # trop mou
    reasons.append(f"wpm={wpm:.0f}")

    # ── 8. Pénalité fillers ──
    fillers = sum(
        text_lower.count(f) for f in (
            " euh ", " heu ", " bah ", " uh ", " um ", " like, ", " you know, ",
            " i mean ", " sort of ", " kind of ", " basically ",
        )
    )
    filler_density = fillers / max(len(seg.words), 1)
    if filler_density > 0.05:
        penalty = filler_density * 30
        score -= penalty
        reasons.append(f"filler-{fillers}({-penalty:.1f})")

    # ── 9. Bonus si la fin "résout" (point, point d'interrogation/exclamation) ──
    last_word = seg.words[-1].text if seg.words else ""
    if last_word.endswith((".", "!", "?")):
        score += 1.5
        reasons.append("clean_end")

    # ── 10. Pénalité si le segment est juste de la conversation/intro ──
    intro_phrases = (
        "welcome", "thanks for", "subscribe", "follow us", "let's start",
        "first of all", "for those of you", "before we",
    )
    if any(p in text_lower[:80] for p in intro_phrases):
        score -= 3
        reasons.append("intro_talk")

    # ── 11. SIGNAL AUDIO : pics de réaction (le plus important pour le viral) ──
    if audio and audio.energies:
        s_start = int(seg.start)
        s_end = min(int(seg.end), len(audio.energies))
        if s_end > s_start:
            # Pics modérés (réactions générales)
            peaks_in = sum(1 for s in audio.peak_seconds if s_start <= s < s_end)
            # Rires / surprises (énergie soutenue)
            laughs_in = sum(1 for s in audio.laughter_seconds if s_start <= s < s_end)

            if peaks_in:
                bonus = min(peaks_in * 1.5, 8)
                score += bonus
                reasons.append(f"audio_peaks={peaks_in}(+{bonus:.1f})")

            if laughs_in:
                # Énorme bonus pour les rires/réactions fortes (signal #1 de viral)
                bonus = min(laughs_in * 2.5, 15)
                score += bonus
                reasons.append(f"REACTION={laughs_in}(+{bonus:.1f})⭐")

            # Bonus supplémentaire si la réaction arrive en FIN de clip (= punchline)
            last_third_start = s_start + int((s_end - s_start) * 0.66)
            climax_laughs = sum(
                1 for s in audio.laughter_seconds
                if last_third_start <= s < s_end
            )
            if climax_laughs:
                score += 4
                reasons.append("CLIMAX_LAUGH(+4)")

            # Bonus si la SECONDE 0-3 a un pic (= hook visuel/audio fort dès le début)
            opening_peak = any(s_start <= s < s_start + 3 for s in audio.peak_seconds)
            if opening_peak:
                score += 2
                reasons.append("OPENING_PEAK(+2)")

    return score, " ".join(reasons)


def select_top_segments(
    candidates: List[Segment],
    max_clips: int,
    audio: Optional[AudioAnalysis] = None,
) -> List[Segment]:
    for c in candidates:
        c.score, c.reason = score_segment(c, audio=audio)
    candidates.sort(key=lambda s: s.score, reverse=True)

    chosen: List[Segment] = []
    for c in candidates:
        if any(not (c.end <= other.start - 1 or c.start >= other.end + 1) for other in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= max_clips:
            break
    chosen.sort(key=lambda s: s.start)
    return chosen


# ---------------------------------------------------------------------------
# Génération du contenu textuel : hook / caption / hashtags
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # FR
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais",
    "donc", "que", "qui", "quoi", "dont", "où", "ce", "cette", "ces", "se",
    "sa", "son", "ses", "mon", "ma", "mes", "ton", "ta", "tes", "il", "elle",
    "on", "nous", "vous", "ils", "elles", "je", "tu", "me", "te", "lui", "leur",
    "y", "en", "à", "au", "aux", "avec", "pour", "par", "sur", "dans", "sans",
    "pas", "ne", "plus", "moins", "très", "trop", "bien", "tout", "tous",
    "toutes", "toute", "même", "aussi", "alors", "puis", "comme", "fait",
    "faire", "être", "avoir", "est", "sont", "était", "été", "ai", "as",
    "avons", "avez", "ont", "vais", "vas", "va", "ça", "cela", "vraiment",
    "juste", "déjà", "encore", "voilà", "voici", "ouais",
    # EN
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "as", "if", "so", "not", "no", "yes",
    "just", "really", "very", "quite", "actually", "basically", "literally",
    "like", "well", "okay", "ok", "yeah", "yep", "nope", "im", "ive", "ill",
    "youre", "youve", "youll", "hes", "shes", "theyre", "weve", "wed", "lets",
    "dont", "doesnt", "didnt", "wont", "cant", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "thats", "whats", "whos", "wheres", "whens",
    "hows", "theres", "heres", "gonna", "wanna", "gotta", "kinda", "sorta",
    "know", "think", "thought", "said", "say", "says", "see", "saw", "go",
    "went", "going", "get", "got", "make", "made", "way", "thing", "things",
    "first", "last", "right", "left", "good", "bad", "big", "small", "new",
    "old", "every", "everyone", "someone", "anyone", "nobody", "anything",
    "something", "nothing", "all", "any", "some", "many", "much", "few",
    "more", "less", "such", "only", "own", "same", "other", "another",
    "also", "back", "again", "now", "then", "there", "here", "when", "where",
    "why", "how", "what", "which", "who", "whose", "whom",
}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _truncate_to_words(text: str, max_words: int = 9) -> str:
    """Tronque proprement à max_words mots (pour un hook visuellement court)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "…"


def extract_hook(seg: Segment) -> str:
    """Cherche la phrase la plus punchy. Hook final TOUJOURS court (≤ 9 mots)
    et qui ne commence PAS par un connecteur / filler."""
    sentences = _split_sentences(seg.text)
    candidates: List[str] = list(sentences)

    for sent in sentences:
        for chunk in re.split(r"(?<=[?!])\s+", sent):
            chunk = chunk.strip()
            if chunk and chunk not in candidates:
                candidates.append(chunk)

    best: Tuple[float, str] = (-1.0, "")
    for sent in candidates:
        words = sent.split()
        wc = len(words)
        if wc < 3:
            continue

        sent_lower = sent.lower()
        first_word = words[0].lower().rstrip(",.")

        # ❌ REJETER si la phrase commence par un mot 'sale'
        if not _is_clean_start_word(first_word):
            continue

        # ❌ REJETER si elle se termine par un fragment ouvert
        last_word = words[-1].lower().rstrip(",.!?")
        if last_word in {"and", "or", "but", "the", "a", "to", "of", "i", "you", "we"}:
            continue

        score = 0.0

        if 4 <= wc <= 8:
            score += 6
        elif wc <= 10:
            score += 3
        elif wc <= 14:
            score += 1
        else:
            score -= (wc - 14) * 0.5

        if "?" in sent:
            score += 4    # les questions = excellents hooks
        if "!" in sent:
            score += 2

        for trig, w in VIRAL_TRIGGERS.items():
            if re.search(rf"\b{re.escape(trig)}\w*\b", sent_lower):
                score += w

        if any(sent_lower.startswith(h) for h in HOOK_STARTERS):
            score += 5

        # Bonus si la phrase contient "I" / "we" (storytelling personnel)
        if re.search(r"\b(i|we|my|our|i'm|we're)\b", sent_lower):
            score += 1.5

        # Pénalité fillers/fragments
        if any(f in sent_lower for f in (" euh ", " heu ", " bah ", " uh ", " um ")):
            score -= 2

        if score > best[0]:
            best = (score, sent)

    if not best[1]:
        # Fallback: prendre les premiers mots du segment, en sautant les junks
        clean_words = []
        for w in seg.words:
            t = w.text.strip()
            if not clean_words and not _is_clean_start_word(t):
                continue
            clean_words.append(t)
            if len(clean_words) >= 7:
                break
        if not clean_words:
            return ""
        chosen = " ".join(clean_words)
    else:
        chosen = best[1]

    chosen = chosen.strip().rstrip(".").rstrip(",")
    return _truncate_to_words(chosen, max_words=9).upper()


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Garde les mots significatifs (>= 4 lettres, hors stopwords)."""
    tokens = re.findall(r"[A-Za-zÀ-ÿ]{4,}", text.lower())
    freq: dict = {}
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    # Bonus pour les mots qui apparaissent plusieurs fois
    return [t for t, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def make_hashtags(text: str) -> List[str]:
    tags = ["#podcast", "#reels", "#shorts"]
    for kw in extract_keywords(text, top_n=5):
        clean = re.sub(r"[^a-z0-9]", "", kw)
        if len(clean) < 4:
            continue
        tag = "#" + clean
        if tag not in tags:
            tags.append(tag)
    return tags[:7]


def make_caption(seg: Segment) -> str:
    hook = seg.hook or extract_hook(seg)
    return hook.capitalize().rstrip(".") + " 👇"


def enrich_segments(segments: List[Segment]) -> None:
    for seg in segments:
        seg.hook = extract_hook(seg)
        seg.caption = make_caption(seg)
        seg.hashtags = make_hashtags(seg.text)


# ---------------------------------------------------------------------------
# Découpage vidéo (clip propre, sans sous-titres ni hook incrustés)
# ---------------------------------------------------------------------------

def make_clip(source: Path, out_path: Path, start: float, end: float, vertical: bool = True) -> None:
    """Découpe un clip et le recadre en 9:16 (ou laisse le format d'origine)."""
    duration = end - start

    if vertical:
        vf = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=luma_radius=30:luma_power=2[bg2];"
            "[fg]scale=1080:-2[fg2];"
            "[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}",
            "-filter_complex", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
    run(cmd)


def words_to_srt(words: List[Word], offset: float, max_chars: int = 22) -> str:
    """Génère un .srt compagnon (mots par mots) que tu pourras importer dans CapCut."""
    def ts(t: float) -> str:
        if t < 0:
            t = 0
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    idx = 1
    buf: List[Word] = []

    def flush() -> None:
        nonlocal idx
        if not buf:
            return
        start = max(buf[0].start - offset, 0)
        end = max(buf[-1].end - offset, start + 0.05)
        text = " ".join(w.text for w in buf).strip()
        lines.append(str(idx))
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
        buf.clear()

    for w in words:
        candidate = (" ".join(x.text for x in buf) + " " + w.text).strip()
        if len(candidate) > max_chars and buf:
            flush()
        buf.append(w)
    flush()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s-]+", "-", text)
    return text[:max_len].strip("-") or "clip"


def get_or_make_transcript(
    video: Path,
    cache_dir: Path,
    model_size: str,
    external: Optional[Path],
) -> Tuple[str, List[Word]]:
    if external:
        print(f"📄 Transcription fournie: {external.name}")
        return ("?", load_external_transcript(external))

    cache_path = cache_dir / f"{video.stem}.json"
    if cache_path.exists():
        print(f"📂 Transcription en cache: {cache_path.name}")
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return (data.get("language", "?"), [Word(**w) for w in data["words"]])

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        extract_audio(video, audio)
        language, words = transcribe_whisper(audio, model_size)

    cache_path.write_text(
        json.dumps({"language": language, "words": [asdict(w) for w in words]}, ensure_ascii=False)
    )
    print(f"💾 Transcription sauvée en cache: {cache_path}")
    return (language, words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des reels viraux à partir d'un podcast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=Path, help="Vidéo source (.mp4)")
    parser.add_argument(
        "--transcript", "-t", type=Path, default=None,
        help="(Optionnel) transcription déjà faite (.srt/.vtt/.json). "
             "Si absent, le script transcrit lui-même avec Whisper.",
    )
    parser.add_argument(
        "--model", default="base",
        help="Modèle Whisper: tiny / base / small / medium / large-v3 (défaut: base)",
    )
    parser.add_argument("--max-clips", type=int, default=10)
    parser.add_argument("--min-len", type=float, default=15.0,
                        help="Durée min en s (défaut: 15 — court & punchy)")
    parser.add_argument("--max-len", type=float, default=45.0,
                        help="Durée max en s (défaut: 45)")
    parser.add_argument("--out", type=Path, default=Path("reels_output"))
    parser.add_argument("--horizontal", action="store_true",
                        help="Garde le format d'origine au lieu de recadrer en 9:16.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les moments choisis SANS découper la vidéo.")
    args = parser.parse_args()

    cache_dir = Path("transcripts")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{args.video.stem}.json"

    # En dry-run avec transcription cachée, on n'a pas besoin du fichier vidéo
    can_skip_video = args.dry_run and (args.transcript or cache_path.exists())

    if not args.video.exists() and not can_skip_video:
        sys.exit(f"❌ Vidéo introuvable: {args.video}")

    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"🎬 Vidéo:    {args.video.name}")
    if args.video.exists():
        print(f"   Durée:    {ffprobe_duration(args.video):.0f}s")
    else:
        print("   (vidéo absente — dry-run sur transcription cachée)")
    print(f"🎯 Cible:    {args.max_clips} reels, {args.min_len:.0f}-{args.max_len:.0f}s, "
          f"{'9:16' if not args.horizontal else 'format original'}")
    print("=" * 64)

    language, words = get_or_make_transcript(args.video, cache_dir, args.model, args.transcript)

    audio_analysis = get_or_make_audio_analysis(args.video, cache_dir)
    if audio_analysis is None:
        print("⚠️  Pas de fichier vidéo accessible → analyse audio désactivée "
              "(utilise SEULEMENT le scoring textuel)")

    print("\n🔍 Recherche des moments candidats…")
    candidates = words_to_candidates(words, args.min_len, args.max_len)
    print(f"   {len(candidates)} segments candidats")

    chosen = select_top_segments(candidates, args.max_clips, audio=audio_analysis)
    chosen = [trim_segment_edges(seg) for seg in chosen]
    enrich_segments(chosen)

    print(f"\n✅ {len(chosen)} clips retenus :\n")
    for i, seg in enumerate(chosen, 1):
        preview = (seg.text[:140] + "…") if len(seg.text) > 140 else seg.text
        print(f"  [{i}] {seg.start:7.1f}s → {seg.end:7.1f}s  ({seg.duration:4.1f}s)  score={seg.score:5.1f}")
        print(f"      🎯 HOOK     : {seg.hook}")
        print(f"      📝 CAPTION  : {seg.caption}")
        print(f"      🏷️  HASHTAGS: {' '.join(seg.hashtags)}")
        print(f"      💬          « {preview} »\n")

    if args.dry_run:
        print("🛑 --dry-run: pas de découpage vidéo.")
        return

    for i, seg in enumerate(chosen, 1):
        slug = slugify(" ".join(seg.text.split()[:6]))
        out_clip = args.out / f"reel_{i:02d}_{slug}.mp4"
        print(f"🎞️  [{i}/{len(chosen)}] → {out_clip.name}")
        make_clip(args.video, out_clip, seg.start, seg.end, vertical=not args.horizontal)

        # Fichier .srt compagnon (à importer dans CapCut/Submagic pour styliser tes subs)
        srt_path = out_clip.with_suffix(".srt")
        srt_path.write_text(words_to_srt(seg.words, offset=seg.start), encoding="utf-8")

        # Fichier .txt avec hook / caption / hashtags
        txt = (
            f"HOOK (le titre/accroche à mettre en gros à l'écran):\n"
            f"  {seg.hook}\n\n"
            f"CAPTION (légende du post Insta/TikTok):\n"
            f"  {seg.caption}\n\n"
            f"HASHTAGS:\n  {' '.join(seg.hashtags)}\n\n"
            f"--- Transcription brute du clip ({seg.duration:.1f}s) ---\n{seg.text}\n"
        )
        out_clip.with_suffix(".txt").write_text(txt, encoding="utf-8")

    summary = [
        {
            "rank": i,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": round(seg.duration, 2),
            "score": round(seg.score, 2),
            "hook": seg.hook,
            "caption": seg.caption,
            "hashtags": seg.hashtags,
            "reason": seg.reason,
            "text": seg.text,
        }
        for i, seg in enumerate(chosen, 1)
    ]
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 64)
    print(f"✨ Terminé! {len(chosen)} reels dans: {args.out}/")
    print(f"   • .mp4 = clip vidéo prêt (sans sous-titres, à toi de les ajouter)")
    print(f"   • .srt = sous-titres prêts à importer dans CapCut/Submagic")
    print(f"   • .txt = hook + caption + hashtags à copier dans ton post")
    print(f"📄 Récap global: {args.out / 'summary.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
