# Deploying nexus-agent

The app is one Python process (`nxa serve`) that serves the web UI and the API. Everything it keeps
(projects, uploads, chat sessions, knowledge bases) lives in `NXA_HOME`. Model calls go out to your LLM
provider with your keys.

**Before any deployment that others can reach:**
- Set `NXA_SERVER_TOKEN`. Without it, anyone with the URL can query your data and spend your API keys.
  The UI asks for the token once and remembers it in that browser.
- Give it read-only database credentials. `nxa connect` attaches databases read-only and the generated
  agent only gets read tools, but a read-only DB user is the real guarantee.
- Keep API keys in environment variables or the host's secret store, never in the image or the repo.

## Option 1: your own machine + a free tunnel (free, data never leaves your PC)

Best for personal use and demos, and for large local data like the 3.3 GB Home Credit database.

```powershell
$env:NXA_SERVER_TOKEN = "pick-a-long-random-string"
nexus-harness\.venv\Scripts\nxa.exe serve --port 8787
```

In a second terminal, start a free Cloudflare quick tunnel (no account needed):

```powershell
cloudflared tunnel --url http://localhost:8787
```

It prints a `https://<random>.trycloudflare.com` URL that reaches your local server. It stops when you
close the tunnel or your PC sleeps. `cloudflared` is a free download from Cloudflare.

## Option 2: Hugging Face Spaces (free, always-on URL)

Free CPU Spaces (2 vCPU, 16 GB RAM) run Docker images. The disk is **not persistent** on the free tier:
projects and chats created in the UI are lost when the Space restarts. Commit small datasets and their
`nxa connect` projects into the Space, or connect to an external database.

1. Create a Space: SDK **Docker**, hardware **CPU basic (free)**.
2. Push this repo's `Dockerfile`, `nexus_lite/` and `nexus_agent/` to the Space repo, and add this line to
   the Space README's YAML header: `app_port: 8787`.
3. In the Space's **Settings → Variables and secrets** add the secrets `GEMINI_API_KEYS` (or `GEMINI_API_KEY`)
   and `NXA_SERVER_TOKEN`.
4. Open the Space, enter the token, use **Connect data** with a database URL or uploaded files.

Large files (over a few hundred MB) are better kept in an external database than uploaded to a free Space.

## Option 3: any Docker host (VPS, Render, Railway, Fly.io, Cloud Run)

```bash
docker build -t nexus-agent .
docker run -d -p 8787:8787 \
  -e GEMINI_API_KEYS=key1,key2 -e NXA_SERVER_TOKEN=change-me \
  -v nxa-data:/data nexus-agent
```

- Mount a volume at `/data` so projects and chats survive restarts.
- Hosted platforms set `PORT`; the container respects it.
- Free tiers with 512 MB RAM (Render free) work for small databases only. Large analysis needs 2–4 GB.
- Put HTTPS in front (most platforms do this for you) and keep the token set.

## Configuration reference

| Variable | Purpose |
|---|---|
| `NXA_SERVER_TOKEN` | Required for public deployments. Clients send `Authorization: Bearer <token>`; the UI handles it. |
| `GEMINI_API_KEYS` / `OPENAI_API_KEYS` / … | Comma-separated keys, rotated per call with failover. A single `*_API_KEY` also works. |
| `NXA_HOME` | Where projects, uploads, sessions and knowledge bases live (`/data` in Docker). |
| `NXA_EMBEDDER` | Knowledge-base embeddings: `ollama:<model>` locally, `none` (BM25 only) in Docker, or `litellm:<model>`. |
| `NXA_APPROVAL_TIMEOUT` | Seconds a risky tool call waits for Allow/Deny in the UI (default 300). |

The model for new projects is set in each project's `agent.yaml` (`model:`); change it there.
