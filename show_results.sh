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
read -p "Auswahl (1-3) [Standard: 1]: " MODE
MODE=${MODE:-1}

python3 "$RESULTS_SCRIPT" --mode "$MODE"

# --- INTERAKTIVE SETTINGS-ÜBERNAHME FÜR OPTION 3 ---
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
