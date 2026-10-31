# Codebase Rover

Local-first question answering and pull request review for source code you already have on disk. The app runs on your machine, stores its index in PostgreSQL, and uses Ollama for embeddings and answers. Hosted models are off unless you turn them on for a single request.

Codebase Rover does not require a cloud service. It does not ship a separate search engine. It does not investigate production incidents, pipelines, or telemetry.

## What is implemented

- A default tenant named Local, plus separate **FastCrib** and **POS** workspaces. Every chunk and search is stored and queried with a workspace id.
- Environments `development`, `test`, `stage`, and `production` under each workspace. They are metadata. A commit is recorded only when you set one and a connected repository can resolve it.
- A local Git source connector. It identifies the repository, reads an exact revision, and lists added, modified, deleted, and renamed files. Native id and URL are the absolute path and `file://` URL. `origin`, when present, is stored separately.
- Provider interfaces for a future source-control connector that can fetch a pull request and publish review comments. The local Git connector returns a clear error for both. No GitHub, GitLab, or Azure DevOps connector is included.
- Full and incremental indexing. Tree-sitter (via `tree-sitter-language-pack`) splits recognized languages into symbols with line ranges. Other text uses line windows. Generated files, build outputs, binaries, vendored directories, `.gitignore` matches, and extra globs are skipped.
- Index jobs are safe to retry. Unchanged blobs are not rechunked unless you force a rebuild. Deleted files leave the index on a completed run. The status endpoint shows the live HEAD, the latest indexed commit, and whether the repository is `never`, `running`, `indexed`, `partial`, `stale`, or `error`.
- Workspace questions. Retrieval combines exact text (PostgreSQL full text and substring match) with pgvector similarity. Before an excerpt is shown or sent to a model, the app reads that file from Git at the indexed commit. If the revision cannot be read, the chunk is not cited. If the excerpts do not support an answer, the API says so.
- Diff review. Submit a unified diff and, when a repository is selected, the head revision. The review reads changed files from that commit, retrieves related code from the same workspace (including other repositories), and keeps only findings that quote the diff or head file and sit on changed lines. Speculative notes are returned separately and are not labeled as defects.
- A small web UI and a CLI. The HTTP API works without the UI.
- Docker Compose for the API, PostgreSQL with pgvector, and Ollama.

## Source provider decision

The first connector is local Git. Review and indexing call the connector interfaces instead of a vendor API. A later connector can implement `fetch_pull_request` and `publish_review` without changing the index or the review validator. No hosted Git vendor was selected.

## Limits

- One process runs index jobs. A restart marks an in-progress job failed; start it again.
- There is no authentication. The default server binds to `127.0.0.1`.
- Embeddings are 768 dimensions, matching `nomic-embed-text`. Changing dimension means rebuilding the database.
- Full-text search uses the PostgreSQL `english` configuration.
- Submodules and symlinks are not indexed. Default ignore rules also skip directories named `build`, `dist`, `vendor`, `node_modules`, `target`, `bin`, `obj`, and similar paths, plus lockfiles and minified assets. Tracked files that match those rules or `.gitignore` are still skipped. After you change ignore globs, run a full index.
- The model sees the question or the diff and a bounded set of excerpts, not the whole repository. If Ollama is down, questions return the retrieved excerpts without a synthesized answer, and reviews return the parsed diff without findings.
- Hosted chat, when enabled, receives that same bounded prompt. Embeddings stay on Ollama. Hosted embeddings are not implemented.
- Review comments are not published anywhere.
- `pytest` was run against local PostgreSQL. The web UI was also exercised in headless Chrome: switching between FastCrib and POS, asking a question, and submitting a diff. Docker Compose was not started in that run, and Ollama was not running, so those checks used text search and the model-unavailable path.

## Setup with Docker Compose

Docker is not required to read the code, but it is the intended way to run the stack.

```bash
docker compose up --build
```

The API and UI listen on `http://127.0.0.1:8080`. Postgres listens on `127.0.0.1:5432`. Ollama listens on `127.0.0.1:11434`.

The Compose database user and password are the local defaults `rover` / `rover`. They exist only for this local database. Do not reuse them anywhere else. Do not put a hosted API key in the compose file; pass it through the environment if you enable a hosted model.

Mount the parent directory of your repositories and register the container path:

```bash
ROVER_REPOS_DIR=/path/to/checkouts docker compose up --build
```

Then connect `/repos/<name>` in the UI. Paths are the paths the app process can see.

Pull models once:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

To use Ollama already running on the host, start only the database and app and set `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

## Setup without Docker

Use Python 3.11+ and PostgreSQL with the `vector` and `pg_trgm` extensions. The database user must be allowed to create those extensions, which the Compose Postgres user can do.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export DATABASE_URL=postgresql+psycopg://rover:rover@localhost:5432/rover
export OLLAMA_BASE_URL=http://127.0.0.1:11434
codebase-rover serve
```

Open `http://127.0.0.1:8080`.

## Configuration

Environment variables are listed in `.env.example`.

| Variable | Role |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL for Postgres |
| `OLLAMA_BASE_URL` | Local Ollama daemon |
| `OLLAMA_CHAT_MODEL` | Chat model name, default `llama3.2` |
| `OLLAMA_EMBED_MODEL` | Embedding model name, default `nomic-embed-text` |
| `EMBEDDING_DIM` | Must stay `768` for this schema |
| `HOSTED_MODEL_ENABLED` | Default `false` |
| `HOSTED_MODEL_BASE_URL` | OpenAI-compatible base URL, only if hosted chat is enabled |
| `HOSTED_MODEL_API_KEY` | Sent as a bearer token. Never written to logs on purpose |
| `HOSTED_MODEL_NAME` | Hosted chat model name |
| `ROVER_ALLOWED_ROOTS` | Optional `os.pathsep`-separated list of directories that may be connected |
| `ROVER_CONFIG` | Optional YAML file. `./rover.yaml` is loaded when it exists |

Copy `config/workspaces.example.yaml` to `rover.yaml` to attach repository paths at startup. Missing paths are skipped. The file does not start an index by itself.

Workspaces can also be created from `POST /api/workspaces`. FastCrib and POS are always seeded.

## Model selection

The default provider is Ollama. A request can set `model_preference` to `ollama` or `hosted`.

Hosted chat is used only when all of these are true:

1. `HOSTED_MODEL_ENABLED=true`
2. The base URL and model name are set
3. The request sets `confirm_send_context` to true

The hosted request body is the chat message list already reduced to the question or diff and the retrieved excerpts. The provider refuses a prompt above the configured size. The API key is not returned by `GET /api/models`.

## Data location

- Source code stays in the Git working trees you connect. The app reads blobs with `git` and does not copy the repository.
- Metadata, chunks, embeddings, questions, and reviews are rows in Postgres. In Compose that data is the `rover_pgdata` volume.
- Ollama models are in the `ollama_data` volume, or in your host Ollama data directory if you point at the host.
- There is no other search index.

## Reset

Reset the index but keep workspaces and repository connections:

```bash
codebase-rover reset-index --yes
```

Or `POST /api/maintenance/reset-index` with `{"confirm":"reset-index"}`. The UI has the same action.

Delete all local metadata, including workspaces, questions, and reviews:

```bash
docker compose down -v
```

That removes the Postgres volume. It does not delete your Git checkouts. Starting Compose again recreates the database and reseeds FastCrib and POS.

## Using the app

1. Choose FastCrib or POS.
2. Connect the top-level path of a local Git repository.
3. Index it and read the status. `partial` means files are indexed and embeddings are still missing, usually because the Ollama embedding model is not pulled. `stale` means HEAD has moved since the indexed commit.
4. Ask a question in that workspace. Citations include repository, path, line range, and commit SHA.
5. Paste a unified diff and a head SHA. Select the repository when you want the review to read that exact commit.

CLI equivalents:

```bash
codebase-rover index --workspace fastcrib --path /path/to/repo
codebase-rover ask --workspace fastcrib "Where is tax calculated?"
codebase-rover review --workspace fastcrib --diff change.diff --head <sha> --repository <uuid>
```

Useful API routes, all under `/api`:

- `GET /health`, `GET /models`, `GET /workspaces`
- `POST /workspaces/{slug}/repositories`
- `POST /workspaces/{slug}/repositories/{id}/index`
- `GET /workspaces/{slug}/repositories/{id}/status`
- `POST /workspaces/{slug}/questions`
- `POST /workspaces/{slug}/reviews`
- `PATCH /workspaces/{slug}/environments/{env}`
- `POST /maintenance/reset-index`

Interactive API docs are at `/docs`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The integration tests expect `postgresql+psycopg://rover:rover@localhost:5432/rover_test` with `vector` and `pg_trgm`. They skip when that database is unreachable. They do not call a live Ollama model; synthesis is covered with a fake chat provider, and the HTTP tests cover the model-unavailable path.
