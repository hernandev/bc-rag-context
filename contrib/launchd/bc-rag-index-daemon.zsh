#!/bin/zsh
set -eo pipefail

# launchd does not load ~/.zshenv. Pull JINA_API_KEY and PATH from there.
# zshenv may reference unset vars; do not use nounset while sourcing it.
if [[ -f "${HOME}/.zshenv" ]]; then
  set +u
  source "${HOME}/.zshenv"
  set -u
fi
export PATH="${HOME}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:${PATH}"

ROOT="${BC_RAG_INDEX_ROOT:-/Users/hernandev/code/pleinair/bigcolony-workspaces}"
INTERVAL="${BC_RAG_INDEX_EVERY:-5m}"
BIN="${BC_RAG_BIN:-/Users/hernandev/code/pleinair/bc-rag-context/bin/bc-rag}"

cd "${ROOT}"
exec "${BIN}" index --every "${INTERVAL}" "${ROOT}"
