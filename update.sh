#!/bin/bash
# Sicheres Update: holt den Repo-Stand, behaelt secret.json, das live laufende Team und -- falls der
# VPS seit dem letzten Push selbst nachtrainiert hat (Quartals-Pipeline) -- dessen neuere Modelle/Configs.
set -e
cd "$(dirname "$0")"

echo "1. Sichere secret.json, settings.json, Configs und Modelle ..."
TMP=$(mktemp -d)
cp secret.json "$TMP/secret.json"
[ -f settings.json ] && cp settings.json "$TMP/settings.json"
for d in configs configs_mensch; do [ -d src/jaegerbot/strategy/$d ] && cp -r src/jaegerbot/strategy/$d "$TMP/$d"; done
for d in models models_mensch; do [ -d artifacts/$d ] && cp -r artifacts/$d "$TMP/$d"; done

echo "2. Hole den neuesten Stand von GitHub ..."
git fetch origin
git reset --hard origin/main
cp "$TMP/secret.json" secret.json

echo "3. Python-Pakete in .venv aktualisieren ..."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

echo "4. Team und neueren VPS-Stand uebernehmen ..."
.venv/bin/python3 - "$TMP" <<'PY'
import json, os, sys, glob, shutil
tmp = sys.argv[1]

def newest_created(folder):
    stamps = []
    for f in glob.glob(os.path.join(folder, 'config_*.json')):
        try:
            stamps.append(json.load(open(f, encoding='utf-8')).get('_backtest', {}).get('created', ''))
        except Exception:
            pass
    return max(stamps) if stamps else ''

# je Profil (Long: configs/models, Short: configs_mensch/models_mensch)
for cdir, mdir in (('configs', 'models'), ('configs_mensch', 'models_mensch')):
    repo_c = os.path.join('src', 'jaegerbot', 'strategy', cdir)
    vps, repo = newest_created(os.path.join(tmp, cdir)), newest_created(repo_c)
    keep_vps = bool(vps) and vps > repo
    if keep_vps:
        shutil.rmtree(repo_c, ignore_errors=True)
        shutil.copytree(os.path.join(tmp, cdir), repo_c)
        print(f'   {cdir}: vom VPS behalten (VPS {vps} neuer als Repo {repo or "-"})')
    if os.path.isdir(os.path.join(tmp, mdir)):
        # fehlende/neuere Quartalsmodelle vom VPS zurueck (selbst nachtrainiert)
        for f in glob.glob(os.path.join(tmp, mdir, '*', '*.joblib')):
            dst = os.path.join('artifacts', mdir, os.path.relpath(f, os.path.join(tmp, mdir)))
            if not os.path.exists(dst) or keep_vps:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)

old_path = os.path.join(tmp, 'settings.json')
if os.path.exists(old_path):
    old, new = json.load(open(old_path, encoding='utf-8')), json.load(open('settings.json', encoding='utf-8'))
    o, n = old.get('live_trading_settings', {}), new.get('live_trading_settings', {})
    # nur v5-Teams uebernehmen (v4-Team hatte use_macd_filter)
    if o.get('active_strategies') and not any('use_macd_filter' in s for s in o['active_strategies']):
        for k in ('active_strategies', 'risk_per_trade_pct'):
            if k in o:
                n[k] = o[k]
        json.dump(new, open('settings.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"   Team beibehalten: {len(n['active_strategies'])} Strategien")
    else:
        print(f"   Team aus dem Repo: {len(n.get('active_strategies', []))} Strategien")
PY
rm -rf "$TMP"

echo "5. Python-Cache leeren, Rechte setzen ..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
chmod +x *.sh

echo "✅ Update fertig. Erster Start nach v5-Umstieg: ./update.sh ein zweites Mal (das erste Mal lief noch das alte Skript)."
