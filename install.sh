#!/bin/bash

# --- NEU: Not-Aus-Schalter ---
# Beendet das Skript sofort, wenn ein Befehl fehlschlägt.
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================================="
echo "    JaegerBot Installations-Skript (Version mit LFS)"
echo "=======================================================${NC}"

# --- System-Abhängigkeiten installieren ---
echo -e "\n${YELLOW}1/5: Aktualisiere Paketlisten und installiere System-Abhängigkeiten...${NC}"
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git curl

# --- Git LFS (Large File Storage) installieren ---
echo -e "\n${YELLOW}2/5: Installiere Git Large File Storage (LFS) für große Dateien...${NC}"
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git lfs install
echo -e "${GREEN}✔ Git LFS erfolgreich installiert und konfiguriert.${NC}"

# --- Python Virtuelle Umgebung einrichten ---
echo -e "\n${YELLOW}3/5: Erstelle eine isolierte Python-Umgebung (.venv)...${NC}"
python3 -m venv .venv
echo -e "${GREEN}✔ Virtuelle Umgebung wurde erstellt.${NC}"

# --- Python-Bibliotheken installieren ---
echo -e "\n${YELLOW}4/5: Aktiviere die virtuelle Umgebung und installiere die notwendigen Python-Bibliotheken...${NC}"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✔ Alle Python-Bibliotheken wurden erfolgreich installiert.${NC}"
deactivate

# --- Symlink für /home/ubuntu/jaegerbot erstellen (falls als root installiert) ---
CURRENT_DIR=$(pwd)
UBUNTU_HOME="/home/ubuntu"
BOT_NAME="jaegerbot"

if [ "$(id -u)" -eq 0 ] && [ -d "$UBUNTU_HOME" ]; then
    TARGET_LINK="$UBUNTU_HOME/$BOT_NAME"
    
    if [ "$CURRENT_DIR" != "$TARGET_LINK" ]; then
        if [ -L "$TARGET_LINK" ]; then
            echo -e "${YELLOW}  → Symlink $TARGET_LINK existiert bereits${NC}"
        elif [ -d "$TARGET_LINK" ]; then
            echo -e "${YELLOW}  → Lösche existierendes Verzeichnis $TARGET_LINK und erstelle Symlink...${NC}"
            rm -rf "$TARGET_LINK"
            ln -s "$CURRENT_DIR" "$TARGET_LINK"
            echo -e "${GREEN}✔ Symlink erstellt: $TARGET_LINK -> $CURRENT_DIR${NC}"
        else
            ln -s "$CURRENT_DIR" "$TARGET_LINK"
            echo -e "${GREEN}✔ Symlink erstellt: $TARGET_LINK -> $CURRENT_DIR${NC}"
        fi
    fi
fi

# --- Abschluss ---
echo -e "\n${YELLOW}5/5: Setze Ausführungsrechte für alle .sh-Skripte...${NC}"
chmod +x *.sh

echo -e "\n${GREEN}======================================================="
echo "✅  Installation erfolgreich abgeschlossen!"
echo ""
echo "WICHTIG: Wenn du Modelle von GitHub verwendest, lade sie jetzt herunter:"
echo "     ( git lfs pull )"
echo ""
echo "Nächste Schritte:"
echo "  1. Erstelle die 'secret.json' Datei mit deinen API-Keys."
echo "     ( nano secret.json )"
echo "  2. Führe die Pipeline aus, um Modelle direkt auf dem Server zu erstellen:"
echo "     ( bash ./run_pipeline.sh )"
echo "  3. Starte den Live-Bot mit:"
echo "     ( python3 master_runner.py )"
echo -e "=======================================================${NC}"
