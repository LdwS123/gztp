#!/bin/bash
# Même chose que launch_webapp.command (nom plus visible dans le Finder)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/start_webapp.sh"
