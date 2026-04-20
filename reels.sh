#!/usr/bin/env bash
# Lanceur du générateur de reels (v3 par défaut — full stack viral).
#
# Usage:
#   ./reels.sh <video.mp4> [options]
#
# Astuce: glisse-dépose ton fichier vidéo dans le terminal pour récupérer son chemin.

set -e
cd "$(dirname "$0")"

# Choix de l'env Python : on préfère venv312 si dispo (compatible clipsai),
# sinon on tombe sur venv (legacy v1).
if [ -d "venv312" ]; then
  VENV_DIR="venv312"
elif [ -d "venv" ]; then
  VENV_DIR="venv"
else
  echo "⚠️  Setup initial nécessaire. Lance:"
  echo "   python3.12 -m venv venv312"
  echo "   source venv312/bin/activate"
  echo "   pip install -r requirements.txt"
  exit 1
fi

if [ -z "$1" ]; then
  cat <<EOF
Usage: $0 <video.mp4> [options]

⚠️  Pipeline verrouillé EN-only par défaut (--lang en, --model small.en).
    Pour traiter une autre langue : $0 vid.mp4 --lang fr --model small

Exemples:
  $0 podcast.mp4                                      # full stack viral EN (défaut)
  $0 podcast.mp4 --dry-run                            # juste l'analyse
  $0 podcast.mp4 --max-clips 8 --max-len 90
  $0 podcast.mp4 --all-signals                        # +emotion +diarization +prosody
  $0 podcast.mp4 --no-llm                             # sans LLM judge (plus rapide)
  $0 podcast.mp4 --model medium.en                    # +précision (~3x plus lent)
  $0 podcast.mp4 --legacy                             # ancien pipeline v1 (heuristique seul)

Options principales:
  --max-clips N      Nombre max de reels (défaut: 5)
  --shortlist N      Nb candidats au LLM (défaut: 20)
  --min-len N        Durée min en s (défaut: 15)
  --max-len N        Durée max en s (défaut: 90 — sweet spot Reels)
  --setup-max N      Durée max du setup avant un pic viral (défaut: 60s)
  --after-laugh N    Durée à garder après le pic (défaut: 8s)
  --no-viral-arcs    Désactive le viral arc detector
  --no-llm           Pas de LLM judge (heuristique seul)
  --emotion          Active speechbrain emotion-recognition par seconde
  --diarization      Active pyannote diarisation (HF_TOKEN requis)
  --prosody          Active librosa prosodie (pitch + energy)
  --all-signals      Raccourci: --emotion --diarization --prosody
  --lang X           ISO 639-1 (défaut: en). 'auto' pour détection (-30% vitesse)
  --model X          Whisper. Défaut: small.en (EN-only, +précis +rapide).
                     EN-only: tiny.en|base.en|small.en|medium.en
                     Multilingue: tiny|base|small|medium|large-v3
  --llm-model X      Modèle Ollama (défaut: qwen2.5:7b)
  --horizontal       Pas de recadrage 9:16
  --dry-run          Affiche les moments choisis sans découper la vidéo

  --legacy           Utilise l'ancien make_reels.py (pas de viral arcs / LLM)
EOF
  exit 1
fi

source "$VENV_DIR/bin/activate"

if [ "$1" = "--legacy" ]; then
  shift
  python3 make_reels.py "$@"
else
  python3 make_reels_v3.py "$@"
fi
