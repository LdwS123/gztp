#!/bin/bash
# Lance la webapp Flask (drag-and-drop vidéo). Appelé par les .command ou : ./start_webapp.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

clear
cat <<'BANNER'
================================================================
  REELS GENERATOR - WEBAPP LOCALE
================================================================
  Navigateur : http://localhost:5151
  Arrêt : Ctrl+C dans ce terminal
================================================================
BANNER

if [ ! -d "venv312" ]; then
  echo "ERREUR : venv312 introuvable dans $SCRIPT_DIR"
  echo "Crée l'environnement :"
  echo "  cd \"$SCRIPT_DIR\""
  echo "  python3.12 -m venv venv312 && source venv312/bin/activate && pip install -r requirements.txt && pip install flask"
  read -n 1 -s -r -p "Entrée pour fermer..." _
  exit 1
fi

source venv312/bin/activate

if ! python3 -c "import flask" 2>/dev/null; then
  echo "Installation de Flask..."
  pip install flask
fi

if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ""
  echo "⚠️  Ollama ne semble pas démarré (http://localhost:11434)."
  echo "   Lance : ollama serve"
  echo ""
  sleep 2
fi

exec python3 webapp.py
