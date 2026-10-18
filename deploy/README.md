# Deployment

Every push to `main` deploys to the VPS through `.github/workflows/deploy.yml`
(live at https://pawconnect.minolingo.online). It can also be started by hand from the
**Actions** tab ("Deploy to VPS" > "Run workflow").

What runs on the server is `/usr/local/bin/pawconnect-deploy`; `pawconnect-deploy.sh` here is a
copy for reference. It pulls `main` into `/opt/pawconnect`, installs backend requirements, runs
`alembic upgrade head`, builds the frontend, restarts the `pawconnect-api` service and waits for
`/api/health`. If you change the script here, copy it to the server as well.

The deploy key (GitHub secret `VPS_SSH_KEY`) is restricted on the server to that one script.
Other secrets: `VPS_HOST`, `VPS_USER`, `VPS_KNOWN_HOSTS`.

Not handled by the deploy: files kept out of git (M2 `checkpoints/m2.pt`, face login models)
and the server's `backend/.env`.
