# sampleagent

## Support ticket graph

`Agents/sample.py:agent` accepts a ticket as a `messages` list. The graph
stores every user message in PostgreSQL, then runs `guardrail` on the latest
message. Prompt injection attempts and explicit NSFW requests go directly to
`reject_request`, which returns a reason and saves the reply to `chat_history`.
Safe messages continue to urgency classification and extraction of product,
issue type, and sentiment
(`Positive`, `Neutral`, or `Negative`). Planning optionally calls either of
two tools before deciding whether to answer, route to a specialist, or
escalate for human review. The two tools appear as separate graph nodes:
`knowledge_base` searches uploaded RAG documents and `customer_history` reads
the user's last 10 earlier messages across sessions from PostgreSQL. It
excludes the current incoming ticket. After a history lookup,
`assess_history_relevance` compares each earlier message with the latest user
message and passes only related messages to planning, decision, and the final
answer. If none relate to the latest question, those steps receive an empty
history result. Neither tool is mandatory for every ticket.

In LangGraph Studio, a message-only run also works. The graph uses Studio's
thread ID as `session_id` and stores messages under a session-scoped guest ID:

```json
{"messages": [{"role": "user", "content": "ทำอะไรได้บ้าง"}]}
```

```python
from Agents.sample import agent

result = agent.invoke({
    "messages": [{
        "role": "user", "content": "My subscription was charged twice.",
        "id": "incoming-message-001",
    }],
    "user_id": "customer-123",
    "session_id": "session-456",
})
print({key: result.get(key) for key in (
    "urgency", "product", "issue_type", "sentiment", "action",
    "specialist_team", "decision_reason", "response_text", "status",
)})
print(result["messages"][-1].content)  # customer-facing reply
```

Set `OPENAI_API_KEY` and optionally `TICKET_MODEL` to override the default
chat model. For a knowledge-base lookup, the existing RAG PostgreSQL service
must be available and relevant documents must have been uploaded. The graph
uses `APP_DATABASE_URI` when set. Otherwise it uses a PostgreSQL `DATABASE_URI`
or connects to local PostgreSQL with `POSTGRES_PASSWORD`. LangGraph Studio's
`DATABASE_URI=memory:` is ignored for application data. In local development,
the application also reads `POSTGRES_PASSWORD` from the project `.env` if the
dev server does not expose it as a process variable. The graph
returns a customer-facing reply in `response_text` and as the last AI message.
The final reply is stored in `chat_history` as an agent message when identifiers
are supplied.

- `auto_respond` uses the LLM to answer the customer.
- `route_specialist` returns a mock chat session containing the selected team
  and a mock assigned agent ID.
- `escalate_human` returns a mock email record for the relevant department.

Mock records and replies are marked `[Mockup]`. No real email is sent and no
external specialist chat is opened. If a separate chat UI sends a message
outside this graph, record that actual event with the same IDs:

```python
from Models.chat_history import save_chat_message

save_chat_message(
    user_id="customer-123", session_id="session-456", role="agent",
    content="The message that was actually sent",
    message_id="sent-message-789",
)
```

`chat_history` is created when a new PostgreSQL volume initializes, or on
first use for an existing volume. It stores `id`, `user_id`,
`session_id`, `message_id`, `role` (`user` or `agent`), `content`, and
`created_at`. Repeating the same `message_id` in a session does not duplicate a
row. If your chat UI sends messages outside this graph, call
`save_chat_message` for those real user or agent messages as well. Without
explicit IDs, the graph uses `configurable.thread_id` for the session and
`guest:<session_id>` for the user. If there is no thread ID, it derives a
session from the first message ID. Guest history stays within that session.
To connect a customer's history across sessions, pass a stable `user_id` from
your trusted application session; `session_id` can also be supplied in state.
Give incoming messages stable IDs so retries do not create duplicate rows.

## Docker Compose

The stack exposes the Agent Server at `http://127.0.0.1:2025` by default,
leaving port 2024 available for `langgraph dev`. Set `APP_PORT` to another free
host port if needed. PostgreSQL with pgvector is available at
`127.0.0.1:2125`, and Redis at `127.0.0.1:2126`.

1. Add `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, and
   `LANGGRAPH_CLOUD_LICENSE_KEY` to `.env`. See `.env.example` for the names.
   Use a URL-safe PostgreSQL password (letters, digits, hyphens, or underscores),
   because the same value is inserted into `DATABASE_URI`.
2. Start Docker Desktop, then run `docker compose up --build -d`.
3. Check `http://127.0.0.1:2025/ok` or
   `http://127.0.0.1:2025/docs`.

`langgraph.json` uses Python 3.12. If it changes, regenerate the Dockerfile
with `uv run langgraph dockerfile Dockerfile` before rebuilding.

PostgreSQL stores LangGraph threads, run history, checkpoints, and long-term
memory. LangGraph Agent Server creates its own tables, including `assistant`,
`assistant_versions`, `thread`, `run`, `checkpoint_*`, `store`, `cron`, and
`schema_migrations`. The application creates these tables:

| Table | Purpose |
| --- | --- |
| `chat_history` | User and agent messages for customer history lookup |
| `rag_files` | Upload status and original file metadata |
| `rag_documents` | Indexed document metadata and deduplication |
| `rag_chunks` | Searchable passages and vector embeddings |

The `vector` extension is enabled when the database volume is first created.
File and document tables have UUID primary keys; `rag_files.document_id`
references `rag_documents.id`. Both store a `category`. An older
`rag_uploads` table may remain in an existing database if `rag_files` was
already created; the current application reads and writes `rag_files` only.

## Upload documents for RAG

The Agent Server exposes `POST /rag/upload` for PDF, DOCX, UTF-8 TXT, and Markdown
files (up to 5 MB). The request saves the original file under a random name in
the shared `rag_uploads` volume, records its original name, type, size, hash,
and status in PostgreSQL, then returns `201 Created` with an `upload_id` and
`status_url`. The separate `worker` service extracts text, creates up to 200
overlapping chunks, generates `text-embedding-3-small` embeddings, and stores
them in pgvector. Re-uploading the same bytes in the same category uses the
existing RAG document.
Scanned PDFs need OCR and will be marked `failed`.
DOCX paragraphs and table rows are indexed in document order. DOCX has no
page-number metadata in search results.

From PowerShell, upload a file with:

```powershell
$base = "http://127.0.0.1:2025" # Docker; use 2024 for langgraph dev
curl.exe -F "file=@notes.pdf" -F "category=manuals" "$base/rag/upload"
# DOCX works too:
curl.exe -F "file=@ooca_faq.docx" -F "category=ooca" "$base/rag/upload"
```

Check processing status using the returned `upload_id`:

```powershell
curl.exe "$base/rag/uploads/<upload_id>"
```

Statuses are `pending`, `processing`, `completed`, or `failed`. A failed job
includes an `error` field. The original file stays in the upload volume.

Search the stored passages directly:

```powershell
$body = @{ query = "What does the document say?"; limit = 4; category = "manuals" } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/rag/search" -Method Post -ContentType application/json -Body $body
```

Omit `category` to search all categories. Uploads default to `general`.
Category names use letters, digits, underscores, or hyphens, up to 63
characters. The Agent's `search_uploaded_knowledge` tool accepts an optional
`category` argument. Uploading a new document clears that tool's Redis cache, so searches
can see the new content immediately. With local `langgraph dev`, start
PostgreSQL using `docker compose up -d postgres`, then run the worker in a
second terminal with `uv run --env-file .env python -m RAG.worker`. The app and worker use
`POSTGRES_PASSWORD` in `.env` to connect to port 2125 and save originals in
the local `uploads/` directory. The worker creates embeddings through OpenAI,
so `OPENAI_API_KEY` must be set.

Agent Server uses Redis database 0 for streaming. Database 1 holds cached
read-only tool results under `tool-result:v1:*` keys. Add comma-separated tool
names to `CACHEABLE_TOOLS` in `.env` to opt them in. The default is
`search_uploaded_knowledge`. Each successful text result
is cached for 300 seconds using its tool name and arguments as the cache key.
`send_email` is always excluded because repeating a side-effecting tool call
must execute the tool. Conversation checkpoints are stored in PostgreSQL by
Agent Server.

The standalone Agent Server image requires a LangGraph license key. The
Docker Compose deployment is for local development and testing.
