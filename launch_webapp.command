#!/bin/bash
# Launcher Reels Generator — double-clic pour démarrer la webapp.
# Fonctionne quel que soit l’emplacement du dossier (~/gztp, etc.).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

clear
cat <<'BANNER'
================================================================
  REELS GENERATOR - WEBAPP LOCALE
================================================================
  Une fenêtre de browser va s'ouvrir automatiquement.
  Pour arrêter le serveur : Ctrl+C ou ferme cette fenêtre.
================================================================
BANNER

if [ ! -d "venv312" ]; then
  echo "ERREUR : venv312 introuvable dans $SCRIPT_DIR"
  echo "Crée d'abord l'environnement avec :"
  echo "  cd \"$SCRIPT_DIR\""
  echo "  python3.12 -m venv venv312 && source venv312/bin/activate && pip install -r requirements.txt && pip install flask"
  read -n 1 -s -r -p "Appuie sur une touche pour fermer..."
  exit 1
fi

source venv312/bin/activate

# Vérifie que Flask est installé
if ! python3 -c "import flask" 2>/dev/null; then
  echo "Installation de Flask..."
  pip install flask
fi

# Vérifie qu'Ollama tourne (sinon le LLM judge va planter)
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ""
  echo "⚠️  Ollama ne semble pas démarré (http://localhost:11434)."
  echo "   Lance-le dans un autre terminal : ollama serve"
  echo "   (ou désactive le LLM judge dans webapp.py)"
  echo ""
  sleep 2
fi

exec python3 webapp.py
