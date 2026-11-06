# sampleagent

## Setup with Docker

Requires Docker Desktop, an OpenAI API key, and a LangGraph license key.
Run these commands in PowerShell:

```powershell
git clone https://github.com/objn/simpleagent.git
cd simpleagent
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, and
`LANGGRAPH_CLOUD_LICENSE_KEY`. Use letters, digits, hyphens, or underscores
for the PostgreSQL password because Docker Compose puts it in a database URL.
`APP_PORT` defaults to `2025`; change it if that port is occupied.

```powershell
docker compose up --build -d
docker compose ps
curl.exe http://127.0.0.1:2025/ok
```

The Agent Server is at `http://127.0.0.1:2025` (or your `APP_PORT`).
API documentation is at `/docs`. Docker Compose starts the app, RAG worker,
PostgreSQL with pgvector, and Redis. To stop them, run `docker compose down`.

## Manual setup

Requires Python 3.12, `uv`, PostgreSQL with
the pgvector extension installed, and an OpenAI API key. Run these commands
in PowerShell:

```powershell
git clone https://github.com/objn/simpleagent.git
cd simpleagent
Copy-Item .env.example .env
psql -U postgres -d postgres
```

In `psql`, create the application database and enable pgvector. Replace the
example password with your own URL-safe password:

```sql
CREATE ROLE sampleagent LOGIN PASSWORD 'your-password';
CREATE DATABASE sampleagent OWNER sampleagent;
\connect sampleagent
CREATE EXTENSION IF NOT EXISTS vector;
```

Edit `.env` and set `OPENAI_API_KEY` and the database URL:

```dotenv
OPENAI_API_KEY=your-openai-key
APP_DATABASE_URI=postgresql://sampleagent:your-password@127.0.0.1:5432/sampleagent
```

Install dependencies and start the development server from the project root:

```powershell
uv sync
uv run --env-file .env langgraph dev
```

The development server uses `http://127.0.0.1:2024` by default. To process
uploaded RAG documents, open a second PowerShell window in the same project
directory and start the worker:

```powershell
uv run --env-file .env python -m RAG.worker
```

The server and worker use the same local `uploads/` directory. Redis is
optional for manual development; set `REDIS_URL` in `.env` only if you run
Redis for tool-result caching.
