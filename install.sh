#!/bin/bash

# --- NEU: Not-Aus-Schalter ---
# Beendet das Skript sofort, wenn ein Befehl fehlschlägt.
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================================="
echo "    JaegerBot v5 Installations-Skript"
echo "=======================================================${NC}"

# --- System-Abhängigkeiten installieren ---
echo -e "\n${YELLOW}1/4: Aktualisiere Paketlisten und installiere System-Abhängigkeiten...${NC}"
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git

# --- Python Virtuelle Umgebung einrichten ---
echo -e "\n${YELLOW}2/4: Erstelle eine isolierte Python-Umgebung (.venv)...${NC}"
python3 -m venv .venv
echo -e "${GREEN}✔ Virtuelle Umgebung wurde erstellt.${NC}"

# --- Python-Bibliotheken installieren ---
echo -e "\n${YELLOW}3/4: Aktiviere die virtuelle Umgebung und installiere die notwendigen Python-Bibliotheken...${NC}"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✔ Alle Python-Bibliotheken wurden erfolgreich installiert.${NC}"
deactivate

# --- Abschluss ---
echo -e "\n${YELLOW}4/4: Setze Ausführungsrechte für alle .sh-Skripte...${NC}"
chmod +x *.sh

echo -e "\n${GREEN}======================================================="
echo "✅  Installation erfolgreich abgeschlossen!"
echo ""
echo "Nächste Schritte:"
echo "  1. secret.json anlegen (Schlüssel 'jaegerbot' + 'telegram')"
echo "  2. Kerzen-Cache einmalig laden (dauert einige Stunden, im Hintergrund):"
echo "     nohup .venv/bin/python3 download_data.py > logs/download.log 2>&1 &"
echo "  3. Cronjob alle 15 Minuten (siehe README), danach läuft alles automatisch"
echo -e "=======================================================${NC}"
