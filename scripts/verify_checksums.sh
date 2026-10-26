#!/usr/bin/env sh
# Verify (default) or regenerate (--update) data/SHA256SUMS.
#
#   sh scripts/verify_checksums.sh            # check every file listed in data/SHA256SUMS
#   sh scripts/verify_checksums.sh --update   # rewrite data/SHA256SUMS from data/ as checked out
#
# Hashes are over the LF-terminated bytes stored in git (see .gitattributes). Regenerate
# only from a clean checkout, never from a copy whose line endings an editor changed.
# Needs GNU coreutils `sha256sum` (macOS: `brew install coreutils`, or use `shasum -a 256`).
set -eu
cd "$(dirname "$0")/../data"
if command -v sha256sum >/dev/null 2>&1; then
    SHA="sha256sum"
else
    SHA="shasum -a 256"
fi
if [ "${1:-}" = "--update" ]; then
    find . -type f ! -name SHA256SUMS ! -name README.md | LC_ALL=C sort | xargs $SHA > SHA256SUMS
    echo "Rewrote data/SHA256SUMS ($(wc -l < SHA256SUMS | tr -d ' ') files)"
else
    $SHA -c --quiet SHA256SUMS && echo "data/SHA256SUMS: all $(wc -l < SHA256SUMS | tr -d ' ') files OK"
fi
