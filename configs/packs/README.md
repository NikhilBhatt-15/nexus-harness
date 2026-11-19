# Instruction packs

A pack freezes everything on the agent side except the model: retrieval variant, policy template,
query directive (appended to every `KB_query`) and hard rules (injected before `<policy>`).

| Pack | Contents | Purpose |
|---|---|---|
| `clean` | Generic process rules written for this repo. No task-specific content. | The honest number. |
| `matched` | Pinecone's published `v2` pack (policy, directive, hard rules 1–12), fetched by `scripts/nexus/fetch_pinecone_pack.sh`. Not committed. | Reproduce Pinecone's setup. Contains rules written against specific benchmark tasks, so scores with it are test-set-tuned. |

Run: `PACK=clean AGENT_LLM=<model> bash scripts/nexus/run_pack.sh`
