#!/usr/bin/env bash
# Run a shell command on the Colab runtime through the kernel channel (colab exec),
# so it does not depend on (or collide with) an open `colab ssh` connection.
# Usage: scripts/cx.sh 'shell command' [timeout_seconds]
set -euo pipefail
CMD="$1"; TMO="${2:-120}"
TMP="$(mktemp -t cx).py"
python3 - "$CMD" > "$TMP" <<'EOF'
import sys
print("import subprocess")
print("r = subprocess.run(%r, shell=True, capture_output=True, executable='/bin/bash')" % sys.argv[1])
print("print(r.stdout.decode('utf-8', 'replace')[-7000:]); print(r.stderr.decode('utf-8', 'replace')[-3000:]); print('exit=', r.returncode)")
EOF
"$HOME/.local/bin/colab" exec -s stack -f "$TMP" --timeout "$TMO"
rm -f "$TMP"
