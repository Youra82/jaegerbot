import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

def main():
    """
    Liest die aktiven Strategien aus settings.json (manuell) oder 
    optimization_results.json (Autopilot) und zeigt die konfigurierten 
    Hebel- und Risikoeinstellungen aus den config_*.json Dateien an.
    """
    settings_path = os.path.join(PROJECT_ROOT, 'settings.json')
    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
    results_path = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'optimization_results.json')
    
    # --- Ausgabe-Header ---
    print(f"\n{'STRATEGIE / DATEI':<40} | {'HEBEL':<5} | {'RISIKO %':<8}")
    print("-" * 65)

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        live_settings = settings.get('live_trading_settings', {})
        use_auto = live_settings.get('use_auto_optimizer_results', False)
        
        active_files = []

        if use_auto:
            # Autopilot Modus (Logik unverändert beibehalten)
            print(f"(Modus: Autopilot - Lese aus optimization_results.json)\n")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    res = json.load(f)
                    active_files = res.get('optimal_portfolio', []) 
            else:
                print("Fehler: optimization_results.json nicht gefunden.")
                return

        else:
            # Manueller Modus
            print(f"(Modus: Manuell - Lese aus settings.json)\n")
            strats = live_settings.get('active_strategies', [])
            for s in strats:
                if isinstance(s, dict) and s.get('active'):
                    
                    # >>> HIER IST DIE KORREKTUR: Robuste Reinigung des Symbols <<<
                    symbol_raw = s['symbol'] # z.B. 'SOL/USDT:USDT'
                    symbol_clean = symbol_raw.replace('/', '').replace(':', '') # Liefert 'SOLUSDTUSDT'
                    tf = s['timeframe']
                    
                    # Versuche Dateinamen zu finden (ohne und mit _macd Filter-Suffix)
                    candidates = [
                        f"config_{symbol_clean}_{tf}.json",
                        f"config_{symbol_clean}_{tf}_macd.json"
                    ]
                    found = False
                    for c in candidates:
                        full_path_candidate = os.path.join(configs_dir, c)
                        if os.path.exists(full_path_candidate):
                            active_files.append(c)
                            found = True
                            break
                    if not found:
                        print(f"WARNUNG: Config für {s['symbol']} {tf} nicht gefunden.")

        # Werte auslesen
        if not active_files:
            print("Keine aktiven Konfigurationen gefunden.")
        
        for filename in active_files:
            # Sicherstellen, dass Dateiname vollständig ist (für Autopilot-Fallback)
            if not filename.startswith('config_'): filename = f"config_{filename}"
            if not filename.endswith('.json'): filename = f"{filename}.json"

            full_path = os.path.join(configs_dir, filename)
            try:
                with open(full_path, 'r') as f:
                    config_data = json.load(f)
                    risk_data = config_data.get('risk', {})
                    
                    leverage = risk_data.get('leverage', 'N/A')
                    risk_pct = risk_data.get('risk_per_trade_pct', 'N/A')
                    
                    # Strategie-Name schön formatieren (entfernt config_ und .json)
                    display_name = filename.replace('config_', '').replace('.json', '')
                    
                    print(f"{display_name:<40} | {str(leverage):<5} | {str(risk_pct):<8}")
            except Exception as e:
                print(f"Fehler beim Lesen der Config '{filename}': {e}")

    except FileNotFoundError:
        print(f"Kritischer Fehler: settings.json oder eine andere Datei nicht gefunden.")
    except Exception as e:
        print(f"Kritischer Fehler: {e}")
    
    print("-" * 65)

if __name__ == "__main__":
    main()
