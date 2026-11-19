#!/usr/bin/env bash
# Run banking_knowledge with a frozen instruction pack + the nexus-lite knowledge layer.
#   PACK=clean|matched AGENT_LLM=<litellm model> [AGENT_LLM_ARGS=json] [USER_LLM=gpt-5.2]
#   [USER_LLM_ARGS='{"reasoning_effort":"low"}'] [NUM_TRIALS=4] [SEED=300] [CONCURRENCY=8]
#   [NUM_TASKS=] [TASK_IDS="task_001 task_002"] [SAVE_TO=name]
set -euo pipefail
cd "$(dirname "$0")/../.."
PACK="${PACK:-clean}"
source "configs/packs/$PACK/setup.env"
export NEXUS_QUERY_DIRECTIVE="$(cat "configs/packs/$PACK/query_directive.txt")"
[ -n "${PACK_POLICY_FILE}" ] && export TAU2_NEXUS_POLICY_FILE="$PWD/$PACK_POLICY_FILE"
[ "${PACK_USE_HARD_RULES}" = 1 ] && export TAU2_AGENT_HARD_RULES_FILE="$PWD/configs/packs/$PACK/hard_rules.txt"
: "${AGENT_LLM:?set AGENT_LLM}"
SAVE_TO="${SAVE_TO:-nexus_${PACK}_$(echo "$AGENT_LLM" | tr '/:' '__')_$(date +%Y%m%d_%H%M%S)}"
mkdir -p data/simulations
export NEXUS_USAGE_LOG="$PWD/data/simulations/$SAVE_TO.nexus_usage.jsonl"
export PYTHONUTF8=1
args=(run --domain banking_knowledge --retrieval-config "$PACK_RETRIEVAL"
  --agent-llm "$AGENT_LLM" --agent-llm-args "${AGENT_LLM_ARGS:-{\}}"
  --user-llm "${USER_LLM:-gpt-5.2}" --user-llm-args "${USER_LLM_ARGS:-{\"reasoning_effort\":\"low\"\}}"
  --num-trials "${NUM_TRIALS:-4}" --seed "${SEED:-300}" --max-concurrency "${CONCURRENCY:-8}"
  --save-to "$SAVE_TO")
[ -n "${NUM_TASKS:-}" ] && args+=(--num-tasks "$NUM_TASKS")
[ -n "${TASK_IDS:-}" ] && args+=(--task-ids $TASK_IDS)
echo "pack=$PACK retrieval=$PACK_RETRIEVAL context=$NEXUS_CONTEXT url=$NEXUS_URL save=$SAVE_TO"
exec ${TAU2:-tau2} "${args[@]}" ${EXTRA_ARGS:-}
