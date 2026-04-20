"""
Signaux audio avancés pour mieux scorer la viralité.

Trois familles de features, chacune OPTIONNELLE (si la lib n'est pas
installée → on retourne None et le pipeline continue avec ses signaux
existants).

  1. Émotion par seconde (speechbrain wav2vec2 IEMOCAP)
     → score d'intensité émotionnelle (excited / angry / happy)
  2. Diarisation (pyannote-audio)
     → "interaction density" : nb de tours de parole par seconde
       les passages où les speakers se coupent / s'enchaînent vite
       sont presque toujours des moments forts
  3. Prosodie (librosa)
     → variation de pitch (f0_std), spectral flux, énergie
       détecte les passages emphatiques (cri, montée de voix)

Cache les résultats sur disque (parsing audio = lent).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Conteneurs
# ---------------------------------------------------------------------------

@dataclass
class EmotionTrack:
    """Score d'émotion 'arousal' par seconde (0-1, 1 = max intensité)."""
    duration: int
    arousal_per_second: List[float]
    source: str = "speechbrain"


@dataclass
class DiarizationTrack:
    """Pour chaque seconde : nb de speakers actifs + indicateur de turn-taking."""
    duration: int
    speakers_per_second: List[int]   # 0, 1 ou 2 (rare 3+)
    turn_changes_per_second: List[int]  # 1 si changement de speaker à cette seconde
    source: str = "pyannote"


@dataclass
class ProsodyTrack:
    """Prosodie : pitch variability + energy variability par seconde."""
    duration: int
    pitch_std_per_second: List[float]      # std du F0 dans la fenêtre, normalisé 0-1
    energy_var_per_second: List[float]     # variance d'énergie, normalisé 0-1
    source: str = "librosa"


@dataclass
class ViralSignals:
    """Bundle de tout ce qu'on a réussi à calculer."""
    emotion: Optional[EmotionTrack] = None
    diarization: Optional[DiarizationTrack] = None
    prosody: Optional[ProsodyTrack] = None


# ---------------------------------------------------------------------------
# Émotion (speechbrain wav2vec2-IEMOCAP)
# ---------------------------------------------------------------------------

def _try_import_speechbrain():
    try:
        from speechbrain.inference.interfaces import foreign_class  # type: ignore
        return foreign_class
    except Exception:
        return None


def compute_emotion_track(
    audio_wav: Path,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
) -> Optional[EmotionTrack]:
    """Calcule un score d'arousal par seconde.

    Le modèle IEMOCAP renvoie 4 classes (angry / happy / neutral / sad).
    On considère arousal = max(angry, happy) — c'est le proxy d'un moment
    "viral" (intensité haute, peu importe la valence).

    Si speechbrain n'est pas dispo → fallback léger basé sur RMS uniquement
    (déjà calculé dans AudioAnalysis), pour ne pas planter.
    """
    foreign_class = _try_import_speechbrain()
    if foreign_class is None:
        print("   ⚠️  speechbrain non installé → émotion désactivée "
              "(pip install speechbrain torchaudio)")
        return None

    try:
        import torchaudio  # type: ignore
        import torch  # type: ignore
    except Exception:
        print("   ⚠️  torchaudio/torch absent → émotion désactivée")
        return None

    print(f"🎭 Analyse émotion (speechbrain wav2vec2 IEMOCAP)…")
    try:
        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Impossible de charger le modèle émotion: {exc}")
        return None

    # On charge l'audio en 16kHz mono
    try:
        waveform, sr = torchaudio.load(str(audio_wav))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Lecture audio impossible: {exc}")
        return None

    total_samples = waveform.shape[1]
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n_steps = max(1, (total_samples - win) // hop + 1)

    # Indices de classes IEMOCAP (vérif via classifier.hparams.label_encoder.lab2ind)
    labels = []
    try:
        labels = list(classifier.hparams.label_encoder.lab2ind.keys())
    except Exception:
        labels = []
    happy_idx = labels.index("hap") if "hap" in labels else -1
    angry_idx = labels.index("ang") if "ang" in labels else -1

    arousal: List[float] = []
    with torch.no_grad():
        for step in range(n_steps):
            start = step * hop
            end = min(start + win, total_samples)
            chunk = waveform[:, start:end]
            if chunk.shape[1] < sr // 4:  # < 250ms = skip
                arousal.append(0.0)
                continue
            try:
                # classify_batch retourne (out_prob, score, index, label)
                out_prob, _, _, _ = classifier.classify_batch(chunk)
                probs = torch.softmax(out_prob, dim=-1)[0].cpu().numpy()
                if happy_idx >= 0 or angry_idx >= 0:
                    a = 0.0
                    if happy_idx >= 0:
                        a = max(a, float(probs[happy_idx]))
                    if angry_idx >= 0:
                        a = max(a, float(probs[angry_idx]))
                    arousal.append(a)
                else:
                    arousal.append(float(probs.max()))
            except Exception:
                arousal.append(0.0)

            if step % 60 == 0 and step > 0:
                print(f"   … {step}s analysés", flush=True)

    print(f"   ✅ {len(arousal)}s d'arousal calculé")
    return EmotionTrack(duration=len(arousal), arousal_per_second=arousal)


# ---------------------------------------------------------------------------
# Diarisation (pyannote-audio)
# ---------------------------------------------------------------------------

def _try_import_pyannote():
    try:
        from pyannote.audio import Pipeline  # type: ignore
        return Pipeline
    except Exception:
        return None


def compute_diarization_track(
    audio_wav: Path,
    duration_seconds: int,
    hf_token: Optional[str] = None,
) -> Optional[DiarizationTrack]:
    """Lance pyannote/speaker-diarization-3.1 sur l'audio.

    Nécessite un token Hugging Face (env HF_TOKEN) car le modèle est gated.
    Sans token → return None.
    """
    Pipeline = _try_import_pyannote()
    if Pipeline is None:
        print("   ⚠️  pyannote-audio non installé → diarisation désactivée "
              "(pip install pyannote.audio)")
        return None

    import os
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("   ⚠️  HF_TOKEN absent → diarisation désactivée "
              "(récupère un token sur huggingface.co/settings/tokens et "
              "accepte la licence pyannote/speaker-diarization-3.1)")
        return None

    print(f"🗣️🗣️  Diarisation (pyannote-audio)…")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Modèle pyannote inaccessible: {exc}")
        return None

    try:
        diarization = pipeline(str(audio_wav))
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Diarisation échouée: {exc}")
        return None

    speakers_per_second = [0] * (duration_seconds + 1)
    turn_changes_per_second = [0] * (duration_seconds + 1)
    last_speaker_at_sec: Dict[int, str] = {}

    intervals: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        intervals.append((turn.start, turn.end, str(speaker)))

    # Compte de speakers actifs par seconde
    for s in range(duration_seconds + 1):
        active = {sp for (st, en, sp) in intervals if st <= s < en}
        speakers_per_second[s] = min(len(active), 3)
        if active:
            primary = sorted(active)[0]
            if s > 0 and last_speaker_at_sec.get(s - 1) and primary != last_speaker_at_sec[s - 1]:
                turn_changes_per_second[s] = 1
            last_speaker_at_sec[s] = primary

    print(f"   ✅ Diarisation OK ({len(set(sp for _,_,sp in intervals))} speakers)")
    return DiarizationTrack(
        duration=duration_seconds,
        speakers_per_second=speakers_per_second,
        turn_changes_per_second=turn_changes_per_second,
    )


# ---------------------------------------------------------------------------
# Prosodie (librosa)
# ---------------------------------------------------------------------------

def _try_import_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None


def compute_prosody_track(audio_wav: Path) -> Optional[ProsodyTrack]:
    """Calcule la variation de pitch (F0) et d'énergie par seconde.

    Un f0_std élevé = la voix monte/descend beaucoup = excitation.
    Une energy_var élevée = pic d'intensité (cri, rire, claque).
    """
    librosa = _try_import_librosa()
    if librosa is None:
        print("   ⚠️  librosa non installé → prosodie désactivée "
              "(pip install librosa)")
        return None

    print(f"🎼 Analyse prosodie (librosa pitch + energy)…")
    try:
        import numpy as np  # type: ignore
        y, sr = librosa.load(str(audio_wav), sr=16000, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Chargement audio prosodie échoué: {exc}")
        return None

    duration = int(len(y) / sr)

    # F0 via piptrack (rapide, suffisant pour macro-tendance)
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=80, fmax=400, sr=sr,
            frame_length=2048, hop_length=512,
        )
    except Exception:
        f0 = None

    pitch_std_per_second: List[float] = []
    if f0 is not None:
        frames_per_sec = sr / 512
        for s in range(duration):
            frame_start = int(s * frames_per_sec)
            frame_end = int((s + 1) * frames_per_sec)
            window = f0[frame_start:frame_end]
            if window is None or len(window) == 0:
                pitch_std_per_second.append(0.0)
                continue
            valid = [x for x in window if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if len(valid) < 3:
                pitch_std_per_second.append(0.0)
                continue
            mean = sum(valid) / len(valid)
            var = sum((v - mean) ** 2 for v in valid) / len(valid)
            pitch_std_per_second.append(math.sqrt(var))

    # Normalisation 0-1
    if pitch_std_per_second:
        mx = max(pitch_std_per_second) or 1.0
        pitch_std_per_second = [v / mx for v in pitch_std_per_second]
    else:
        pitch_std_per_second = [0.0] * duration

    # Energy variance par seconde
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    frames_per_sec = sr / 512
    energy_var_per_second: List[float] = []
    for s in range(duration):
        a = int(s * frames_per_sec)
        b = int((s + 1) * frames_per_sec)
        window = rms[a:b]
        if len(window) < 2:
            energy_var_per_second.append(0.0)
            continue
        mean = sum(window) / len(window)
        var = sum((v - mean) ** 2 for v in window) / len(window)
        energy_var_per_second.append(float(math.sqrt(var)))

    if energy_var_per_second:
        mx = max(energy_var_per_second) or 1.0
        energy_var_per_second = [v / mx for v in energy_var_per_second]

    print(f"   ✅ {duration}s de prosodie calculée")
    return ProsodyTrack(
        duration=duration,
        pitch_std_per_second=pitch_std_per_second,
        energy_var_per_second=energy_var_per_second,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _signals_cache_path(video_stem: str, cache_dir: Path) -> Path:
    return cache_dir / f"{video_stem}.signals.json"


def load_signals(video_stem: str, cache_dir: Path) -> Optional[ViralSignals]:
    p = _signals_cache_path(video_stem, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    out = ViralSignals()
    if data.get("emotion"):
        out.emotion = EmotionTrack(**data["emotion"])
    if data.get("diarization"):
        out.diarization = DiarizationTrack(**data["diarization"])
    if data.get("prosody"):
        out.prosody = ProsodyTrack(**data["prosody"])
    return out


def save_signals(video_stem: str, cache_dir: Path, signals: ViralSignals) -> None:
    p = _signals_cache_path(video_stem, cache_dir)
    payload = {
        "emotion": asdict(signals.emotion) if signals.emotion else None,
        "diarization": asdict(signals.diarization) if signals.diarization else None,
        "prosody": asdict(signals.prosody) if signals.prosody else None,
    }
    p.write_text(json.dumps(payload))


def compute_all_signals(
    audio_wav: Path,
    duration_seconds: int,
    cache_dir: Path,
    video_stem: str,
    enable_emotion: bool = True,
    enable_diarization: bool = True,
    enable_prosody: bool = True,
    hf_token: Optional[str] = None,
) -> ViralSignals:
    """Calcule (ou recharge depuis cache) tous les signaux activés."""
    cached = load_signals(video_stem, cache_dir)
    if cached:
        print(f"📂 Signaux viraux en cache: {video_stem}.signals.json")
        return cached

    signals = ViralSignals()
    if enable_emotion:
        signals.emotion = compute_emotion_track(audio_wav)
    if enable_diarization:
        signals.diarization = compute_diarization_track(
            audio_wav, duration_seconds, hf_token=hf_token,
        )
    if enable_prosody:
        signals.prosody = compute_prosody_track(audio_wav)

    save_signals(video_stem, cache_dir, signals)
    return signals


# ---------------------------------------------------------------------------
# Scoring : combiner les tracks pour donner un bonus à un Segment
# ---------------------------------------------------------------------------

def signals_bonus_for_segment(
    seg_start: float,
    seg_end: float,
    signals: ViralSignals,
) -> Tuple[float, str]:
    """Calcule un bonus score [0-15] basé sur les signaux audio avancés
    présents sur la fenêtre [seg_start, seg_end].

    Pondération :
      - emotion arousal moyen   : 0-6 points
      - turn-taking density     : 0-4 points
      - pitch / energy variance : 0-3 points
      - reaction late-clip      : 0-2 points (bonus si pic en fin)
    """
    bonus = 0.0
    reasons: List[str] = []

    s_start = int(seg_start)
    s_end = max(s_start + 1, int(seg_end))

    if signals.emotion and signals.emotion.arousal_per_second:
        track = signals.emotion.arousal_per_second
        window = track[s_start: min(s_end, len(track))]
        if window:
            avg = sum(window) / len(window)
            mx = max(window)
            score = avg * 4.0 + mx * 2.0
            bonus += score
            reasons.append(f"emo(avg={avg:.2f},max={mx:.2f},+{score:.1f})")

            # Bonus si le pic émotionnel est dans le dernier tiers (= punchline)
            third = s_start + (s_end - s_start) * 2 // 3
            late_window = track[third: min(s_end, len(track))]
            if late_window and max(late_window) > 0.6:
                bonus += 2.0
                reasons.append("emo_late+2")

    if signals.diarization and signals.diarization.turn_changes_per_second:
        track = signals.diarization.turn_changes_per_second
        window = track[s_start: min(s_end, len(track))]
        if window:
            changes = sum(window)
            density = changes / max(1, len(window))
            # Échange vif = 0.05+ changements/sec (1 toutes les 20s déjà beaucoup)
            score = min(4.0, density * 80)
            if score > 0.5:
                bonus += score
                reasons.append(f"turns({changes},+{score:.1f})")

    if signals.prosody:
        if signals.prosody.pitch_std_per_second:
            track = signals.prosody.pitch_std_per_second
            window = track[s_start: min(s_end, len(track))]
            if window:
                avg = sum(window) / len(window)
                score = min(2.0, avg * 4.0)
                if score > 0.3:
                    bonus += score
                    reasons.append(f"pitch({avg:.2f},+{score:.1f})")
        if signals.prosody.energy_var_per_second:
            track = signals.prosody.energy_var_per_second
            window = track[s_start: min(s_end, len(track))]
            if window:
                mx = max(window)
                score = min(1.0, mx * 1.5)
                if score > 0.3:
                    bonus += score
                    reasons.append(f"egyvar({mx:.2f},+{score:.1f})")

    return min(bonus, 15.0), " ".join(reasons)
