#!/bin/bash
# Compatibility wrapper. Keep the canonical implementation under scripts/.

set -uo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "$PROJECT/scripts/run_server.sh" "$@"
