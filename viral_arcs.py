"""
Viral arc detector — construit des candidats de clip CENTRÉS sur les pics
émotionnels (rires, réactions fortes, montées d'énergie) plutôt que sur
les frontières topiques de ClipsAI.

Idée : un moment viral n'est PAS défini par son sujet, il est défini par
la RÉACTION qu'il provoque. Donc on part des pics audio et on construit
l'arc narratif autour :

    [─── setup (jusqu'à 60s avant) ───][ PIC ][ after-laugh (5-15s) ]
                                              ↑ centre = climax
    snap aux frontières de phrase            cap total = max_len * 1.15

On émet ensuite ces arcs comme candidats Segment, qui partent dans le
même pipeline (LLM judge, génération hook…) que ceux de ClipsAI.

Ça résout 2 problèmes :
  1. Les vrais moments viraux qui durent > max_len (storytelling +
     punchline + rire) sont enfin capturés en entier.
  2. La sélection se base sur un signal réel (réaction du public) au
     lieu d'un score textuel qui rate les bons gags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from make_reels import AudioAnalysis, Segment, Word


_STRONG_END_PUNCT = {".", "!", "?", "…"}


@dataclass
class PeakCluster:
    """Un cluster contigu de secondes où l'audience réagit fort."""
    start_sec: int          # première seconde du cluster (inclus)
    end_sec: int            # dernière seconde du cluster (inclus)
    intensity: float        # somme normalisée de l'énergie au-dessus du seuil
    laughter_sec: int       # nb de secondes "laughter" dedans
    peak_sec: int           # nb de secondes "peak" dedans

    @property
    def center(self) -> float:
        return (self.start_sec + self.end_sec) / 2

    @property
    def duration(self) -> int:
        return self.end_sec - self.start_sec + 1


def cluster_peaks(
    audio: AudioAnalysis,
    merge_gap: int = 4,
    min_intensity_seconds: int = 1,
) -> List[PeakCluster]:
    """Regroupe les `peak_seconds` et `laughter_seconds` en clusters contigus.

    Deux pics séparés par moins de `merge_gap` secondes sont fusionnés
    (= une réaction qui se prolonge en plusieurs vagues, typique des
    rires en cascade ou d'un échange chargé).
    """
    if not audio or not audio.energies:
        return []

    flagged = sorted(set(audio.peak_seconds) | set(audio.laughter_seconds))
    if not flagged:
        return []

    laughter_set = set(audio.laughter_seconds)
    peak_set = set(audio.peak_seconds)

    clusters: List[PeakCluster] = []
    cur_start = flagged[0]
    cur_end = flagged[0]

    for s in flagged[1:]:
        if s - cur_end <= merge_gap:
            cur_end = s
        else:
            clusters.append(_make_cluster(cur_start, cur_end, audio, laughter_set, peak_set))
            cur_start = s
            cur_end = s
    clusters.append(_make_cluster(cur_start, cur_end, audio, laughter_set, peak_set))

    # On vire les micro-clusters (1 seconde isolée = sûrement faux positif)
    return [c for c in clusters if c.laughter_sec + c.peak_sec >= min_intensity_seconds]


def _make_cluster(
    start: int, end: int, audio: AudioAnalysis,
    laughter_set: set, peak_set: set,
) -> PeakCluster:
    intensity = 0.0
    laughter_sec = 0
    peak_sec = 0
    threshold = audio.mean + audio.std  # seuil "peak"
    for s in range(start, end + 1):
        if 0 <= s < len(audio.energies):
            excess = max(0.0, audio.energies[s] - threshold)
            intensity += excess
            if s in laughter_set:
                laughter_sec += 1
            if s in peak_set:
                peak_sec += 1
    # Bonus d'intensité pour les rires (= signal le plus prédictif)
    intensity += laughter_sec * (audio.std * 2.0)
    return PeakCluster(
        start_sec=start, end_sec=end,
        intensity=intensity,
        laughter_sec=laughter_sec,
        peak_sec=peak_sec,
    )


# ---------------------------------------------------------------------------
# Construction d'arcs autour des clusters
# ---------------------------------------------------------------------------

def _word_index_at_time(words: List[Word], t: float) -> int:
    """Index du premier mot dont start >= t (binaire light)."""
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if words[mid].start < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _snap_start_back_to_sentence(words: List[Word], idx: int, look_back: int = 30) -> int:
    """Recule jusqu'à juste APRÈS la fin d'une phrase (point/?/!).

    `look_back` est en NOMBRE DE MOTS (pas de secondes) — ~30 mots ≈ 10s.
    Si on ne trouve rien, on retourne idx tel quel.
    """
    if idx <= 0:
        return 0
    for k in range(1, min(look_back, idx) + 1):
        i = idx - k
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1
    return idx


def _snap_end_forward_to_sentence(words: List[Word], idx: int, look_ahead: int = 25) -> int:
    """Avance jusqu'à inclure la fin de la phrase courante."""
    if idx >= len(words):
        return len(words)
    for k in range(0, min(look_ahead, len(words) - idx)):
        i = idx + k
        t = words[i].text.strip()
        if t and t[-1] in _STRONG_END_PUNCT:
            return i + 1
    return idx


def _is_clean_first_word(text: str) -> bool:
    bad = {
        "and", "or", "but", "so", "because", "yeah", "yes", "no", "okay", "ok",
        "well", "uh", "um", "like", "i", "you",
        "et", "ou", "mais", "donc", "alors", "car", "puis", "ouais", "non",
        "bah", "ben", "bon", "euh",
    }
    return text.strip().lower().rstrip(",.!?") not in bad


def build_arc_for_cluster(
    cluster: PeakCluster,
    words: List[Word],
    min_len: float,
    max_len: float,
    setup_max: float = 60.0,
    after_laugh: float = 8.0,
) -> Optional[Segment]:
    """Construit un Segment "arc viral" autour d'un cluster de pics.

    - On part du milieu du cluster.
    - On recule jusqu'à `setup_max` secondes pour récupérer le setup (mais
      on snap sur une frontière de phrase propre).
    - On avance de `after_laugh` secondes après le pic pour inclure la
      réaction (post-laugh / réplique).
    - On cap la durée totale à `max_len * 1.15`.
    - On garantit au minimum `min_len`.
    """
    if not words:
        return None

    pic_start_t = float(cluster.start_sec)
    pic_end_t = float(cluster.end_sec) + 1.0

    setup_target_t = max(0.0, pic_start_t - setup_max)
    end_target_t = pic_end_t + after_laugh

    start_idx = _word_index_at_time(words, setup_target_t)
    end_idx = _word_index_at_time(words, end_target_t)

    start_idx = _snap_start_back_to_sentence(words, start_idx, look_back=40)
    end_idx = _snap_end_forward_to_sentence(words, end_idx, look_ahead=30)

    if start_idx >= end_idx or start_idx >= len(words):
        return None

    while start_idx < end_idx and not _is_clean_first_word(words[start_idx].text):
        start_idx += 1
    if start_idx >= end_idx:
        return None

    duration = words[end_idx - 1].end - words[start_idx].start

    hard_cap = max_len * 1.15
    if duration > hard_cap:
        # Trop long : on resserre côté setup uniquement (on garde la chute)
        target_start_t = words[end_idx - 1].end - hard_cap
        start_idx = _word_index_at_time(words, target_start_t)
        start_idx = _snap_start_back_to_sentence(words, start_idx, look_back=20)
        while start_idx < end_idx and not _is_clean_first_word(words[start_idx].text):
            start_idx += 1
        duration = words[end_idx - 1].end - words[start_idx].start

    if duration < min_len:
        # Trop court : on étend côté setup d'abord
        needed = min_len - duration
        target_start_t = max(0.0, words[start_idx].start - needed - 5.0)
        start_idx = _word_index_at_time(words, target_start_t)
        start_idx = _snap_start_back_to_sentence(words, start_idx, look_back=40)
        while start_idx < end_idx and not _is_clean_first_word(words[start_idx].text):
            start_idx += 1

    if start_idx >= end_idx:
        return None
    duration = words[end_idx - 1].end - words[start_idx].start
    if duration < min_len * 0.7:
        return None

    seg_words = words[start_idx:end_idx]
    text = " ".join(w.text for w in seg_words).strip()
    if not text:
        return None

    seg = Segment(
        start=seg_words[0].start,
        end=seg_words[-1].end,
        text=text,
        words=seg_words,
        score=cluster.intensity,
        reason=(
            f"viral_arc(laugh={cluster.laughter_sec}s peak={cluster.peak_sec}s "
            f"intensity={cluster.intensity:.0f}, dur={duration:.0f}s)"
        ),
    )
    return seg


def build_viral_arc_candidates(
    words: List[Word],
    audio: Optional[AudioAnalysis],
    min_len: float,
    max_len: float,
    setup_max: float = 60.0,
    after_laugh: float = 8.0,
    max_arcs: int = 30,
) -> List[Segment]:
    """Pipeline complet : audio → clusters → arcs Segment.

    Renvoie les `max_arcs` arcs les plus intenses, triés par intensité décroissante
    (avant que le scoring viral / LLM judge ne reprenne la main).
    """
    if not audio or not audio.energies:
        return []

    clusters = cluster_peaks(audio)
    if not clusters:
        return []

    clusters.sort(key=lambda c: c.intensity, reverse=True)

    arcs: List[Segment] = []
    seen_centers: List[float] = []
    for c in clusters:
        # Anti-doublons : si on a déjà un arc dont le centre est < 30s → skip
        if any(abs(c.center - sc) < 30 for sc in seen_centers):
            continue
        arc = build_arc_for_cluster(
            c, words,
            min_len=min_len, max_len=max_len,
            setup_max=setup_max, after_laugh=after_laugh,
        )
        if arc is None:
            continue
        arcs.append(arc)
        seen_centers.append(c.center)
        if len(arcs) >= max_arcs:
            break
    return arcs


def merge_arcs_with_clipsai(
    arc_candidates: List[Segment],
    clipsai_candidates: List[Segment],
    overlap_tolerance: float = 5.0,
) -> List[Segment]:
    """Fusionne les 2 sources de candidats en évitant les doublons exacts.

    Stratégie : on garde TOUS les arcs (signal le plus fort), puis on ajoute
    les ClipsAI qui ne chevauchent pas un arc à plus de 50%.
    """
    out = list(arc_candidates)
    for cand in clipsai_candidates:
        cand_dur = cand.end - cand.start
        if cand_dur <= 0:
            continue
        overlapped = False
        for arc in arc_candidates:
            inter_start = max(cand.start, arc.start)
            inter_end = min(cand.end, arc.end)
            inter = max(0.0, inter_end - inter_start)
            if inter / cand_dur > 0.5:
                overlapped = True
                break
        if not overlapped:
            out.append(cand)
    return out
