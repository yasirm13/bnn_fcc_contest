#!/usr/bin/bash
set -euo pipefail

COV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${COV_DIR}/.." && pwd)"

source "${ROOT_DIR}/activate.sh"

cd "${COV_DIR}"

make clean
make sim

REPORT="${1:-coverage_report.txt}"

if [[ ! -f "$REPORT" ]]; then
    echo "Coverage report not found: $REPORT"
    echo "Run 'make sim' in coverage/ first."
    exit 2
fi

covergroup_pct="$(
    awk '/TOTAL COVERGROUP COVERAGE:/ {print $4; exit}' "$REPORT"
)"

if [[ -z "${covergroup_pct}" ]]; then
    echo "Could not parse coverage percentages from $REPORT"
    exit 2
fi

echo "Covergroup coverage: $covergroup_pct"
echo "Covergroups:"
awk '
    /^ TYPE / {
        name = $2
        pct = $3
        sub("^.*/", "", name)
        printf "  %s %s\n", name, pct
    }
' "$REPORT"

if [[ "$covergroup_pct" == "100.00%" ]]; then
    echo "All functional covergroup coverage is 100%."
    exit 0
fi

echo "Functional covergroup coverage is NOT 100%."
exit 1
