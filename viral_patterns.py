"""
Patterns viraux : laughter detection + dialog patterns.

Trois familles de signaux complémentaires aux audio_signals existants :

  1. LAUGHTER DETECTOR
     Vraie détection de rire (vs simple peak RMS) :
       - librosa onset detection (attaque rapide = clap, rire éclatant)
       - bande haute-fréquence (4-8 kHz) où les rires sont énergétiques
       - ratio HF/LF élevé + onset = rire probable
     → liste de "laughter_seconds" beaucoup plus précis que celle de
       AudioAnalysis (qui ne fait que de l'énergie globale).

  2. PUNCHLINE PATTERN
     Détecte le pattern classique d'une vanne :
       - Speaker A parle (setup, 2-15s)
       - Speaker B réagit court / rit / coupe (≤ 3s)
       - retour à Speaker A (chute, optionnel)
     → liste de (peak_time, score) où score reflète la "punchline-ness".

  3. DUO DYNAMIC
     Densité de turn-taking : passages où deux speakers s'enchaînent vite
     (banter, débat, interruption). On regarde un sliding window de 30s :
       - nb de tours par seconde
       - équilibre du temps de parole entre les 2 speakers
     → `duo_intensity_per_second` (0-1).

Toutes ces données s'utilisent comme bonus dans le scoring final
(via `patterns_bonus_for_segment`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class LaughterTrack:
    """Pour chaque seconde : score 0-1 de probabilité de rire."""
    duration: int
    laugh_score_per_second: List[float]
    laugh_seconds: List[int]  # secondes où le score dépasse le seuil
    source: str = "librosa-hf"


@dataclass
class PunchlineHit:
    """Un pattern setup→punch détecté."""
    setup_start: float
    setup_end: float
    punch_time: float        # moment de la chute (typiquement debut réaction)
    reaction_end: float
    score: float             # 0-1 force de la punchline
    speakers: List[str] = field(default_factory=list)


@dataclass
class DuoTrack:
    """Densité 'banter' par seconde + speakers actifs."""
    duration: int
    duo_intensity_per_second: List[float]  # 0-1
    dominant_speaker_per_second: List[str]


@dataclass
class ViralPatterns:
    laughter: Optional[LaughterTrack] = None
    punchlines: List[PunchlineHit] = field(default_factory=list)
    duo: Optional[DuoTrack] = None


# ---------------------------------------------------------------------------
# 1. LAUGHTER DETECTOR (librosa)
# ---------------------------------------------------------------------------

def compute_laughter_track(audio_wav: Path,
                           threshold: float = 0.55) -> Optional[LaughterTrack]:
    """Détecte les rires par seconde via :
      - librosa.onset.onset_strength (attaque rapide)
      - ratio d'énergie haute-fréquence (4-8 kHz) sur basse-fréquence
      - facteur de "burstiness" (énergie max/moy dans la fenêtre)

    Retourne un score 0-1 par seconde. >= `threshold` = rire probable.
    """
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
    except Exception as exc:
        print(f"   ⚠️  librosa indispo, laughter detection désactivée: {exc}")
        return None

    print("😂 Détection des rires (librosa onset + bande HF)…")
    try:
        y, sr = librosa.load(str(audio_wav), sr=16000, mono=True)
    except Exception as exc:
        print(f"   ⚠️  Lecture audio échouée: {exc}")
        return None

    duration = int(len(y) / sr)
    hop = 512
    frames_per_sec = sr / hop

    # 1) Onset strength (mesure l'attaque)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onset_max = float(onset_env.max() or 1.0)
        onset_norm = onset_env / onset_max
    except Exception:
        onset_norm = np.zeros(int(len(y) / hop))

    # 2) Énergies par bande (basse vs haute fréquence)
    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = (freqs >= 100) & (freqs < 1500)
        high_mask = (freqs >= 4000) & (freqs < 8000)
        low_e = S[low_mask].sum(axis=0) + 1e-9
        high_e = S[high_mask].sum(axis=0)
        hf_ratio = high_e / low_e
        hf_norm = np.clip(hf_ratio / (hf_ratio.mean() * 3 + 1e-9), 0, 1)
    except Exception:
        hf_norm = np.zeros_like(onset_norm)

    # 3) Combine onset + HF + variance pour score brut par frame
    n_frames = min(len(onset_norm), len(hf_norm))
    onset_norm = onset_norm[:n_frames]
    hf_norm = hf_norm[:n_frames]
    raw = 0.5 * onset_norm + 0.5 * hf_norm

    # 4) Agrège par seconde + boost si le pic est marqué (burst)
    laugh_score: List[float] = []
    for s in range(duration):
        a = int(s * frames_per_sec)
        b = int((s + 1) * frames_per_sec)
        win = raw[a:b]
        if len(win) == 0:
            laugh_score.append(0.0)
            continue
        avg = float(win.mean())
        mx = float(win.max())
        # Burst factor : un rire = pic court intense, pas un plateau
        burst = (mx - avg) / (mx + 1e-9)
        score = min(1.0, 0.6 * mx + 0.2 * avg + 0.2 * burst)
        laugh_score.append(score)

    laugh_seconds = [s for s, v in enumerate(laugh_score) if v >= threshold]
    print(f"   ✅ {len(laugh_seconds)} secondes de rire détectées "
          f"(seuil {threshold})")
    return LaughterTrack(
        duration=duration,
        laugh_score_per_second=laugh_score,
        laugh_seconds=laugh_seconds,
    )


# ---------------------------------------------------------------------------
# 2. PUNCHLINE PATTERN (texte + speakers + rires)
# ---------------------------------------------------------------------------

def detect_punchlines(
    words,
    laughter: Optional[LaughterTrack] = None,
    setup_min: float = 2.0,
    setup_max: float = 18.0,
    reaction_max: float = 4.0,
) -> List[PunchlineHit]:
    """Détecte les patterns setup→punch→reaction dans les mots speaker-aware.

    Algo :
      - On scanne les transitions de speakers
      - Quand SPK_A parle entre setup_min et setup_max, puis SPK_B parle court
        (≤ reaction_max), c'est un candidat
      - Bonus si la transition coïncide avec un pic de rire

    Utilise les `Word.speaker` (assigné via speaker_diarization). Si pas de
    speakers → retourne [].
    """
    if not words or not any(w.speaker for w in words):
        return []

    # Construit les "tours" de parole (groupes de mots consécutifs même speaker)
    turns: List[Tuple[float, float, str, List[int]]] = []
    cur_spk = None
    cur_start = 0.0
    cur_indices: List[int] = []
    for i, w in enumerate(words):
        if w.speaker != cur_spk:
            if cur_indices:
                turns.append((cur_start, words[cur_indices[-1]].end, cur_spk, cur_indices))
            cur_spk = w.speaker
            cur_start = w.start
            cur_indices = [i]
        else:
            cur_indices.append(i)
    if cur_indices:
        turns.append((cur_start, words[cur_indices[-1]].end, cur_spk, cur_indices))

    hits: List[PunchlineHit] = []
    for k in range(len(turns) - 1):
        a_start, a_end, a_spk, _ = turns[k]
        b_start, b_end, b_spk, _ = turns[k + 1]
        if a_spk is None or b_spk is None or a_spk == b_spk:
            continue
        a_dur = a_end - a_start
        b_dur = b_end - b_start
        if not (setup_min <= a_dur <= setup_max):
            continue
        if b_dur > reaction_max:
            continue

        # Score base : un échange court qui suit un setup propre = punchline
        score = 0.4
        # Bonus si rire pile au moment de la réaction
        if laughter and laughter.laugh_score_per_second:
            ls = laughter.laugh_score_per_second
            window = ls[int(b_start): min(int(b_end) + 2, len(ls))]
            if window:
                peak = max(window)
                score += min(0.5, peak * 0.6)

        # Bonus si retour au speaker A juste après (3 turns punchline classique)
        if k + 2 < len(turns):
            c_start, c_end, c_spk, _ = turns[k + 2]
            if c_spk == a_spk and (c_end - c_start) >= 1.0:
                score += 0.15

        speakers_seen = list(dict.fromkeys([a_spk, b_spk]))
        hits.append(PunchlineHit(
            setup_start=a_start,
            setup_end=a_end,
            punch_time=b_start,
            reaction_end=b_end,
            score=min(1.0, score),
            speakers=speakers_seen,
        ))

    print(f"   ✅ {len(hits)} patterns punchline détectés")
    return hits


# ---------------------------------------------------------------------------
# 3. DUO DYNAMIC (banter density)
# ---------------------------------------------------------------------------

def compute_duo_track(words, total_duration: int,
                      window_sec: int = 20) -> Optional[DuoTrack]:
    """Pour chaque seconde, calcule l'intensité du dialogue dans une fenêtre
    glissante de `window_sec` :

      intensity = turn_density * speaker_balance

    où :
      - turn_density = nb de changements de speaker dans la fenêtre / window_sec
      - speaker_balance = 1 - |t_A/t_total - 0.5|*2  (1 = parfaitement équilibré)

    → 0 = monologue, 1 = ping-pong serré entre 2 speakers.
    """
    if not words or not any(w.speaker for w in words):
        return None

    # Construit un tableau : pour chaque seconde, le speaker dominant
    dominant: List[str] = ["?"] * (total_duration + 1)
    for w in words:
        if not w.speaker:
            continue
        s = int(w.start)
        e = min(int(w.end) + 1, total_duration + 1)
        for sec in range(s, e):
            if 0 <= sec <= total_duration:
                # On garde le 1er speaker rencontré pour cette seconde
                if dominant[sec] == "?":
                    dominant[sec] = w.speaker

    intensity: List[float] = [0.0] * (total_duration + 1)
    half = window_sec // 2
    for s in range(total_duration + 1):
        a = max(0, s - half)
        b = min(total_duration + 1, s + half)
        window = [d for d in dominant[a:b] if d != "?"]
        if len(window) < 4:
            continue

        # Turn density
        turns = sum(1 for i in range(1, len(window)) if window[i] != window[i - 1])
        turn_density = turns / max(1, len(window))

        # Speaker balance (top 2 speakers)
        from collections import Counter
        cnt = Counter(window).most_common(2)
        if len(cnt) < 2:
            balance = 0.0
        else:
            t_a, t_b = cnt[0][1], cnt[1][1]
            tot = t_a + t_b
            balance = 1.0 - abs(t_a / tot - 0.5) * 2

        intensity[s] = min(1.0, turn_density * 2.5 * balance)

    print(f"   ✅ Duo dynamic calculé "
          f"(max intensity = {max(intensity):.2f})")
    return DuoTrack(
        duration=total_duration,
        duo_intensity_per_second=intensity,
        dominant_speaker_per_second=dominant,
    )


# ---------------------------------------------------------------------------
# Cache disque
# ---------------------------------------------------------------------------

def _cache_path(stem: str, cache_dir: Path) -> Path:
    return cache_dir / f"{stem}.patterns.json"


def load_patterns(stem: str, cache_dir: Path) -> Optional[ViralPatterns]:
    p = _cache_path(stem, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    out = ViralPatterns()
    if data.get("laughter"):
        out.laughter = LaughterTrack(**data["laughter"])
    if data.get("punchlines"):
        out.punchlines = [PunchlineHit(**h) for h in data["punchlines"]]
    if data.get("duo"):
        out.duo = DuoTrack(**data["duo"])
    return out


def save_patterns(stem: str, cache_dir: Path, patterns: ViralPatterns) -> None:
    p = _cache_path(stem, cache_dir)
    payload = {
        "laughter": asdict(patterns.laughter) if patterns.laughter else None,
        "punchlines": [asdict(h) for h in patterns.punchlines],
        "duo": asdict(patterns.duo) if patterns.duo else None,
    }
    p.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# API publique : compute_all_patterns
# ---------------------------------------------------------------------------

def compute_all_patterns(
    audio_wav: Path,
    words,
    total_duration: int,
    cache_dir: Path,
    stem: str,
    enable_laughter: bool = True,
    enable_punchlines: bool = True,
    enable_duo: bool = True,
) -> ViralPatterns:
    """Calcule (ou recharge depuis cache) tous les patterns viraux.

    Note : les patterns punchline et duo dépendent des `Word.speaker`.
    Si la diarization a échoué, ces deux signaux retourneront vides.
    """
    cached = load_patterns(stem, cache_dir)
    if cached:
        n_punch = len(cached.punchlines) if cached.punchlines else 0
        print(f"📂 Patterns viraux en cache "
              f"({n_punch} punchlines, "
              f"laughter={'✓' if cached.laughter else '✗'}, "
              f"duo={'✓' if cached.duo else '✗'})")
        return cached

    patterns = ViralPatterns()
    if enable_laughter:
        patterns.laughter = compute_laughter_track(audio_wav)

    if enable_punchlines:
        print("🎤 Détection des patterns punchline (setup→punch→reaction)…")
        patterns.punchlines = detect_punchlines(words, laughter=patterns.laughter)

    if enable_duo:
        print("🏓 Calcul de la dynamique 'duo' (banter density)…")
        patterns.duo = compute_duo_track(words, total_duration)

    save_patterns(stem, cache_dir, patterns)
    return patterns


# ---------------------------------------------------------------------------
# Scoring : bonus pour un Segment basé sur les patterns
# ---------------------------------------------------------------------------

def patterns_bonus_for_segment(
    seg_start: float,
    seg_end: float,
    patterns: ViralPatterns,
) -> Tuple[float, str]:
    """Calcule un bonus [0-20] basé sur les patterns viraux dans la fenêtre.

    Pondération :
      - laughter present (pic ≥ 0.6) : 0-5 pts
      - punchline pattern overlap   : 0-8 pts
      - duo dynamic intensity       : 0-5 pts
      - punchline en fin de clip    : 0-2 pts (= la clip se TERMINE sur la chute)
    """
    bonus = 0.0
    reasons: List[str] = []

    s_start = int(seg_start)
    s_end = max(s_start + 1, int(seg_end))

    # --- Laughter ---
    if patterns.laughter and patterns.laughter.laugh_score_per_second:
        track = patterns.laughter.laugh_score_per_second
        window = track[s_start: min(s_end, len(track))]
        if window:
            mx = max(window)
            n_high = sum(1 for v in window if v >= 0.55)
            score = min(5.0, mx * 3.5 + n_high * 0.3)
            if score > 0.5:
                bonus += score
                reasons.append(f"laugh(max={mx:.2f},n={n_high},+{score:.1f})")

    # --- Punchlines overlap ---
    if patterns.punchlines:
        seg_punchlines = [
            h for h in patterns.punchlines
            if h.setup_start >= seg_start - 1 and h.reaction_end <= seg_end + 1
        ]
        if seg_punchlines:
            best = max(seg_punchlines, key=lambda h: h.score)
            score = min(8.0, best.score * 8)
            bonus += score
            reasons.append(f"punch({len(seg_punchlines)},best={best.score:.2f},+{score:.1f})")

            # Bonus si la punchline est dans le dernier tiers du clip
            seg_dur = seg_end - seg_start
            third = seg_start + seg_dur * 2 / 3
            if any(h.punch_time >= third for h in seg_punchlines):
                bonus += 2.0
                reasons.append("punch_late+2")

    # --- Duo dynamic ---
    if patterns.duo and patterns.duo.duo_intensity_per_second:
        track = patterns.duo.duo_intensity_per_second
        window = track[s_start: min(s_end, len(track))]
        if window:
            avg = sum(window) / len(window)
            score = min(5.0, avg * 5.5)
            if score > 0.5:
                bonus += score
                reasons.append(f"duo(avg={avg:.2f},+{score:.1f})")

    return min(bonus, 20.0), " ".join(reasons)
