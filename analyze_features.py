# /root/jaegerbot/analyze_features.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
import warnings # Import für Warnungen hinzugefügt

# --- Pfad-Setup ---
# Stelle sicher, dass der Bot seine eigenen Module importieren kann
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Deaktiviere laute TensorFlow-Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
# Ignoriere Keras-spezifische UserWarnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

try:
    from jaegerbot.analysis.backtester import load_data
    from jaegerbot.utils.ann_model import prepare_data_for_ann
except ImportError as e:
    print(f"Fehler: Konnte Bot-Module nicht importieren. Stelle sicher, dass du im .venv bist. Fehler: {e}")
    sys.exit(1)
# --- Ende Pfad-Setup ---


def analyze_feature_importance(symbol, timeframe, start_date, end_date):
    """
    Lädt Daten, trainiert ein RandomForest-Modell und bewertet die Wichtigkeit der Features.
    """
    print(f"\n--- Starte Feature-Wichtigkeits-Analyse für {symbol} ({timeframe}) ---")
    print(f"Zeitraum: {start_date} bis {end_date}\n")

    # 1. Daten laden und vorbereiten (exakt wie im Trainer)
    data = load_data(symbol, timeframe, start_date, end_date)
    if data.empty:
        print("Fehler: Konnte keine Daten laden.")
        return

    # prepare_data_for_ann gibt uns die Features (X) und das Ziel (y)
    # Es filtert bereits nur die "klaren Signale" heraus, was perfekt ist.
    X, y = prepare_data_for_ann(data, timeframe, verbose=False)

    if X.empty:
        print("Fehler: Keine Trainingsdaten (X) nach der Vorbereitung gefunden.")
        return

    print(f"Daten geladen. {len(X)} klare Signale gefunden, die analysiert werden.")

    # Daten normalisieren (Standard-Praxis)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. RandomForest-Modell trainieren
    # RandomForests sind "Ensembles" von Entscheidungsbäumen und eignen sich
    # hervorragend zur Bewertung der Wichtigkeit von Features.
    print("Trainiere RandomForest-Modell, um Feature-Wichtigkeit zu bewerten...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    # 3. Wichtigkeit extrahieren und anzeigen
    importances = model.feature_importances_
    feature_names = X.columns # Holt die Namen der Indikatoren

    # Erstelle ein DataFrame zur schönen Anzeige
    importance_df = pd.DataFrame({
        'Indikator': feature_names,
        'Wichtigkeit': importances
    })
    importance_df = importance_df.sort_values(by='Wichtigkeit', ascending=False)

    print("\n--- Ergebnisse der Feature-Wichtigkeit ---")
    print(importance_df.to_string(index=False))

    # 4. Ergebnisse als Diagramm speichern
    print("\nSpeichere Diagramm als 'feature_importance.png'...")
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature-Wichtigkeit für {symbol} ({timeframe})')
    plt.barh(importance_df['Indikator'], importance_df['Wichtigkeit'])
    plt.xlabel('Wichtigkeit (Score)')
    plt.gca().invert_yaxis() # Wichtigstes Feature oben
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    print("\n--- Analyse abgeschlossen ---")
    
    # Dynamische Empfehlung basierend auf Feature-Anzahl
    n_features = len(feature_names)
    avg_importance = 1.0 / n_features
    threshold_low = avg_importance * 0.5  # 50% vom Durchschnitt
    threshold_high = avg_importance * 2.0  # 200% vom Durchschnitt
    
    print(f"\n📊 INTERPRETATIONSHILFE (bei {n_features} Features):")
    print(f"   Durchschnittliche Wichtigkeit: {avg_importance*100:.2f}%")
    print(f"   • Sehr unwichtig: < {threshold_low*100:.2f}% → Entfernen erwägen")
    print(f"   • Unterdurchschnittlich: {threshold_low*100:.2f}% - {avg_importance*100:.2f}%")
    print(f"   • Durchschnittlich: {avg_importance*100:.2f}% - {threshold_high*100:.2f}%")
    print(f"   • Sehr wichtig: > {threshold_high*100:.2f}% → Definitiv behalten!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JaegerBot Feature-Wichtigkeits-Analyse")
    parser.add_argument('--symbol', required=True, type=str, help="Symbol (z.B. BTC)")
    parser.add_argument('--timeframe', required=True, type=str, help="Timeframe (z.B. 4h)")
    parser.add_argument('--start_date', required=True, type=str, help="Startdatum (JJJJ-MM-TT)")
    parser.add_argument('--end_date', required=True, type=str, help="Enddatum (JJJJ-MM-TT)")
    args = parser.parse_args()

    # Prüfe, ob matplotlib installiert ist
    try:
        import matplotlib
    except ImportError:
        print("\nWARNUNG: 'matplotlib' fehlt. Bitte füge 'matplotlib' zu deiner requirements.txt hinzu und installiere es.")
        print("Führe aus: echo 'matplotlib' >> requirements.txt && .venv/bin/pip install -r requirements.txt")
        sys.exit(1)

    analyze_feature_importance(
        symbol=f"{args.symbol.upper()}/USDT:USDT",
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
