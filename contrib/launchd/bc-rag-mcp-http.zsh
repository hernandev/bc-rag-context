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

BIN="${BC_RAG_BIN:-/Users/hernandev/code/pleinair/bc-rag-context/bin/bc-rag}"
HOST="${BC_RAG_MCP_HOST:-127.0.0.1}"
PORT="${BC_RAG_MCP_PORT:-32323}"

exec "${BIN}" mcp --http --host "${HOST}" --port "${PORT}"
