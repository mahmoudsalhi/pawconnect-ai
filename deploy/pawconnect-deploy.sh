#!/usr/bin/env bash
# PawConnect deploy. Runs as the forced command of the GitHub Actions deploy key
# (~ubuntu/.ssh/authorized_keys), so that key can run this script and nothing else.
#   ssh ubuntu@vps          -> deploy the latest origin/main
#   ssh ubuntu@vps check    -> only report what is deployed
set -euo pipefail

APP=/opt/pawconnect
exec 9>/tmp/pawconnect-deploy.lock
flock -n 9 || { echo "Another deploy is already running."; exit 1; }
cd "$APP"

if [ "${SSH_ORIGINAL_COMMAND:-}" = "check" ]; then
    echo "Deploy key OK. Deployed: $(git log --oneline -1)"
    exit 0
fi

echo "==> Pulling main"
git fetch -q origin main
git merge -q --ff-only origin/main
echo "    now at $(git log --oneline -1)"

echo "==> Backend: dependencies and migrations"
cd "$APP/backend"
.venv/bin/pip install -q -r requirements.txt
.venv/bin/alembic upgrade head

echo "==> Frontend: build"
cd "$APP/frontend"
npm ci --no-audit --no-fund --loglevel=error
rm -rf dist-new
npm run build -- --outDir dist-new >/dev/null
rm -rf dist-old && { [ -d dist ] && mv dist dist-old || true; } && mv dist-new dist && rm -rf dist-old

echo "==> Restarting API"
sudo systemctl restart pawconnect-api
for i in $(seq 1 30); do
    if curl -fs http://127.0.0.1:8100/api/health >/dev/null; then
        echo "==> Deployed $(git -C "$APP" log --oneline -1)"
        exit 0
    fi
    sleep 2
done
echo "API did not come back after restart:"; sudo journalctl -u pawconnect-api -n 30 --no-pager
exit 1
