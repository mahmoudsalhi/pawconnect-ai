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
git lfs pull   # model files stored with Git LFS (M2 checkpoint)
echo "    now at $(git log --oneline -1)"

# PyTorch comes from the CPU-only index: the default Linux wheels bundle ~2 GB of CUDA the VPS can't use.
TORCH_CPU=(--extra-index-url https://download.pytorch.org/whl/cpu)

echo "==> Backend: dependencies and migrations"
cd "$APP/backend"
# requirements-ml.txt = requirements.txt + the libraries M7 (lost-pet location) needs at runtime
.venv/bin/pip install -q "${TORCH_CPU[@]}" -r requirements-ml.txt
.venv/bin/alembic upgrade head
# Switch from the placeholders to the trained models once the code has them (one-time change to .env).
if [ -f app/ai/models.py ] && grep -q '^AI_PROVIDER=demo$' .env; then
    sed -i 's/^AI_PROVIDER=demo$/AI_PROVIDER=app.ai.models:PawModels/' .env
    echo "    AI_PROVIDER switched to app.ai.models:PawModels"
fi

M2="$APP/ml/photo_matching"
if [ -f "$M2/requirements.txt" ] && [ -f "$M2/checkpoints/m2.pt" ]; then
    echo "==> M2 photo matching: dependencies and restart"
    "$M2/.venv/bin/pip" install -q "${TORCH_CPU[@]}" -r "$M2/requirements.txt"
    sudo systemctl restart pawconnect-m2
fi

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
        if systemctl is-active -q pawconnect-m2; then
            curl -fs --retry 10 --retry-delay 3 --retry-all-errors http://127.0.0.1:8002/health >/dev/null \
                && echo "    M2 service healthy" || echo "    WARNING: M2 service not answering (photo matching returns 503)"
        fi
        echo "==> Deployed $(git -C "$APP" log --oneline -1)"
        exit 0
    fi
    sleep 2
done
echo "API did not come back after restart:"; sudo journalctl -u pawconnect-api -n 30 --no-pager
exit 1
