#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
VENV_PATH=".venv/bin/activate"
ANALYSIS_SCRIPT="analyze_features.py"

# Überprüfen, ob das Analyse-Skript existiert
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo -e "${RED}Fehler: Das Skript '$ANALYSIS_SCRIPT' wurde nicht gefunden.${NC}"
    echo "Bitte stelle sicher, dass du die Datei 'analyze_features.py' im Hauptverzeichnis gespeichert hast."
    exit 1
fi

source "$VENV_PATH"
echo -e "\n${BLUE}--- JaegerBot Feature-Wichtigkeits-Analyse ---${NC}"

# Interaktive Abfrage
read -p "Handelspaar(e) eingeben (ohne /USDT, z.B. BTC ETH): " SYMBOLS
read -p "Zeitfenster eingeben (z.B. 1h 4h): " TIMEFRAMES
read -p "Startdatum (JJJJ-MM-TT) [Standard: 2023-01-01]: " START_DATE
START_DATE=${START_DATE:-"2023-01-01"}
read -p "Enddatum (JJJJ-MM-TT) [Standard: Heute]: " END_DATE
END_DATE=${END_DATE:-$(date +%F)}

if [ -z "$SYMBOLS" ] || [ -z "$TIMEFRAMES" ]; then
    echo -e "${RED}Fehler: Symbole und Timeframes dürfen nicht leer sein.${NC}"
    deactivate
    exit 1
fi

for symbol in $SYMBOLS; do
    for timeframe in $TIMEFRAMES; do
        echo -e "\n${YELLOW}=======================================================${NC}"
        echo -e "${YELLOW}Analysiere Features für: $symbol ($timeframe)${NC}"
        echo -e "${YELLOW}=======================================================${NC}"
        
        python3 "$ANALYSIS_SCRIPT" \
            --symbol "$symbol" \
            --timeframe "$timeframe" \
            --start_date "$START_DATE" \
            --end_date "$END_DATE"
        
        # Das Diagramm wird bei jeder Analyse überschrieben
        echo -e "${GREEN}✔ Analyse für $symbol ($timeframe) abgeschlossen.${NC}"
        echo "Das Diagramm 'feature_importance.png' wurde aktualisiert."
    done
done

deactivate
echo -e "\n${BLUE}--- Alle Analysen abgeschlossen ---${NC}"
