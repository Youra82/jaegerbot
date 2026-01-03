#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
VENV_PATH=".venv/bin/activate"
RESULTS_SCRIPT="src/jaegerbot/analysis/show_results.py"
OPTIMAL_CONFIGS_FILE=".optimal_configs.tmp"
UPDATE_SCRIPT="update_settings_from_optimizer.py"

source "$VENV_PATH"

# --- ERWEITERTES MODUS-MENÜ ---
echo -e "\n${YELLOW}Wähle einen Analyse-Modus:${NC}"
echo "  1) Einzel-Analyse (jede Strategie wird isoliert getestet)"
echo "  2) Manuelle Portfolio-Simulation (du wählst das Team)"
echo "  3) Automatische Portfolio-Optimierung (der Bot wählt das beste Team)"
echo "  4) Interaktive Charts (mit EMA, MACD, RSI, Bollinger Bands)"
read -p "Auswahl (1-4) [Standard: 1]: " MODE
MODE=${MODE:-1}

python3 "$RESULTS_SCRIPT" --mode "$MODE"

# --- OPTION 4: INTERAKTIVE CHARTS ---
if [ "$MODE" == "4" ]; then
    echo -e "\n${YELLOW}========== INTERAKTIVE CHARTS ===========${NC}"
    echo ""
    read -p "Symbol (z.B. DOGE/USDT): " SYMBOL
    read -p "Timeframe (z.B. 4h, 1h) [Standard: 4h]: " TIMEFRAME
    TIMEFRAME=${TIMEFRAME:-4h}
    read -p "Start-Kapital [Standard: 1000]: " START_CAPITAL
    START_CAPITAL=${START_CAPITAL:-1000}
    read -p "Letzte N Tage anzeigen (oder leer für alle): " WINDOW
    read -p "Telegram versenden? (j/n) [Standard: n]: " SEND_TELEGRAM
    
    TELEGRAM_FLAG=""
    if [[ "$SEND_TELEGRAM" =~ ^[jJyY]$ ]]; then
        TELEGRAM_FLAG="--send-telegram"
    fi
    
    WINDOW_FLAG=""
    if [ ! -z "$WINDOW" ]; then
        WINDOW_FLAG="--window $WINDOW"
    fi
    
    echo -e "\n${BLUE}Generiere Chart...${NC}"
    python3 src/jaegerbot/analysis/interactive_status.py \
        --symbol "$SYMBOL" \
        --timeframe "$TIMEFRAME" \
        --start-capital "$START_CAPITAL" \
        $WINDOW_FLAG \
        $TELEGRAM_FLAG
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Chart wurde generiert!${NC}"
    else
        echo -e "${RED}❌ Fehler beim Generieren des Charts.${NC}"
    fi
    
    deactivate
    exit 0
fi
if [ "$MODE" == "3" ] && [ -f "$OPTIMAL_CONFIGS_FILE" ]; then
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  SETTINGS AUTOMATISCH AKTUALISIEREN?${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "Die optimierten Strategien können jetzt automatisch"
    echo "in die settings.json übernommen werden."
    echo ""
    echo -e "${RED}ACHTUNG:${NC} Dies ersetzt alle aktuellen Strategien!"
    echo "Es wird automatisch ein Backup erstellt (settings.json.backup)."
    echo ""
    read -p "Sollen die optimierten Strategien übernommen werden? (j/n): " APPLY_SETTINGS
    
    if [[ "$APPLY_SETTINGS" =~ ^[jJyY]$ ]]; then
        echo ""
        echo -e "${BLUE}Aktualisiere settings.json...${NC}"
        
        # Lese Config-Dateien aus Temp-Datei
        CONFIGS=$(cat "$OPTIMAL_CONFIGS_FILE")
        
        # Rufe Python-Script auf mit allen Config-Namen als Argumente
        python3 "$UPDATE_SCRIPT" $CONFIGS
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ Settings wurden erfolgreich aktualisiert!${NC}"
            echo -e "${GREEN}   Backup wurde erstellt: settings.json.backup${NC}"
        else
            echo ""
            echo -e "${RED}❌ Fehler beim Aktualisieren der Settings.${NC}"
        fi
        
        # Lösche Temp-Datei
        rm -f "$OPTIMAL_CONFIGS_FILE"
    else
        echo ""
        echo -e "${YELLOW}ℹ  Settings wurden NICHT aktualisiert.${NC}"
        echo "Du kannst die Strategien später manuell in settings.json eintragen."
        
        # Lösche Temp-Datei
        rm -f "$OPTIMAL_CONFIGS_FILE"
    fi
fi

deactivate
