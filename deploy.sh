#!/usr/bin/env bash
# DS4 Flash production deploy for gx10. Two subcommands; no mode matrix, no intent
# file, no image-hash verification — see docs/DEPLOY.md for the rationale.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  production)
    # GB10 unified-memory leak: only a reboot reclaims a leaked slice. Refuse to
    # start unless the host is clean (freshly booted, nothing holding the slice).
    avail_gib=$(free -g | awk '/^Mem:/{print $7}')
    if (( avail_gib < 100 )); then
      echo "ERROR: only ${avail_gib} GiB available — host is not clean (leaked?)." >&2
      echo "Run '$0 clean' first (it reboots), reconnect, then rerun '$0 production'." >&2
      exit 1
    fi
    echo ">> deploying $(git describe --tags --always) ($(git rev-parse --short HEAD)) from current checkout..."
    echo ">> building + starting DS4..."
    docker compose up -d --build
    echo ">> polling /v1/models (warm-up can take ~1-2 min)..."
    for _ in $(seq 1 120); do
      if curl -sf localhost:8000/v1/models 2>/dev/null | grep -q deepseek-v4-flash; then
        echo ">> DS4 ready."; exit 0
      fi
      sleep 5
    done
    echo "ERROR: DS4 not ready in time; check 'docker logs ds4'." >&2
    exit 1
    ;;
  clean)
    echo ">> stopping DS4 and rebooting (only way to reclaim the GB10 unified-memory leak)..."
    docker compose down || true
    sudo systemctl reboot
    ;;
  *)
    echo "Usage: $0 {production|clean}" >&2
    echo "  production   build+up from current checkout; poll readiness (host must be clean)" >&2
    echo "  clean        compose down + reboot (reclaims unified memory; takes the host offline)" >&2
    exit 2
    ;;
esac
