#!/bin/sh
set -eu

# Bind-mounted service-account files commonly retain restrictive host
# permissions. Copy the file while running as root, then hand a private copy
# to the unprivileged application user.
if [ -n "${GOOGLE_SERVICE_ACCOUNT_FILE:-}" ] && [ -f "$GOOGLE_SERVICE_ACCOUNT_FILE" ]; then
    runtime_credentials=/tmp/google-service-account.json
    umask 077
    cp "$GOOGLE_SERVICE_ACCOUNT_FILE" "$runtime_credentials"
    chown appuser:appuser "$runtime_credentials"
    export GOOGLE_SERVICE_ACCOUNT_FILE="$runtime_credentials"
fi

exec su -s /bin/sh appuser -c \
    'GOOGLE_SERVICE_ACCOUNT_FILE=/tmp/google-service-account.json exec python -m uvicorn server:app --host 0.0.0.0 --port 9004'
