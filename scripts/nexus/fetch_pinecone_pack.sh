#!/usr/bin/env bash
# Fetch Pinecone's published v2 instruction pack (pinecone-io/tau2-bench, branch `nexus`) into
# configs/packs/matched/. Kept out of git: it is their content and it is tuned on benchmark tasks.
set -euo pipefail
cd "$(dirname "$0")/../.."
base="https://raw.githubusercontent.com/pinecone-io/tau2-bench/nexus/configs/nexus_setups/v2"
mkdir -p configs/packs/matched
for f in policy.md query_directive.txt hard_rules.txt; do
  curl -fsSL "$base/$f" -o "configs/packs/matched/$f"
done
cat > configs/packs/matched/setup.env <<'ENV'
export NEXUS_CONTEXT="${NEXUS_CONTEXT:-banking}"
export NEXUS_URL="${NEXUS_URL:-inproc}"
export NEXUS_QUERY_DEPTH="${NEXUS_QUERY_DEPTH:-standard}"
export PACK_RETRIEVAL="nexus_multiturn"
export PACK_POLICY_FILE="configs/packs/matched/policy.md"
export PACK_USE_HARD_RULES=1
ENV
echo "matched pack written to configs/packs/matched/"
