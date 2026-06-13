# fraud-detection-platform — Instructions for Claude Code

Real-time fraud detection platform (Python). Key components: `ml_pipeline/`
(feature engineering, data prep, training), `scoring/` (inference service),
`streaming/`, `feature_store/`, `alert_service/`, `dashboard/`, `simulator/`,
`sdk/` + `sdks/`, `cli/`, plus plugins (telegram, registry/loader).

## Persistent Memory (Obsidian Vault)

Long-term project memory lives in the Obsidian vault at `~/vault`:

- `~/vault/fraud-detection-platform/architecture/` — decisions, conventions
- `~/vault/fraud-detection-platform/pipeline/` — data flows, APIs
- `~/vault/fraud-detection-platform/data/` — schema, data model
- `~/vault/fraud-detection-platform/features/` — planned/implemented features
- `~/vault/fraud-detection-platform/logs/` — session logs
- `~/vault/graphify/fraud-detection-platform/` — codebase knowledge graph (symlinked)

Use `/resume` at the start of a session to load context, and `/save` at the
end to persist a session log. See `~/vault/CLAUDE.md` for the full Zettelkasten
rules.

## Context Navigation (Graphify)

### 3-Layer Query Rule

1. **First:** query `graphify-out/graph.json` to understand code structure
   and connections (functions, modules, imports, call graph).
2. **Second:** query the Obsidian vault (`~/vault/fraud-detection-platform/`)
   for decisions, progress, and project context.
3. **Third:** only read raw source files when editing, or when the first two
   layers don't have the answer.

Query the graph instead of re-reading the codebase:

```bash
graphify query "how does scoring call the feature store"
graphify explain "score_transaction"
graphify path "ingest_event" "send_alert"
```

### When to rebuild the graph

- After structural changes (new modules, major refactors).
- `graphify update .` — only re-processes modified files (no LLM needed).
- The graph is persistent — NO need to rebuild every session.
- A post-commit git hook (if installed via `graphify hook install`) rebuilds it
  automatically.

### Do NOT

- Don't manually modify files inside `graphify-out/`.
- Don't re-read the entire codebase if the graph already has the information.

> The graph is **code-only** (a `.graphifyignore` excludes docs/data/images),
> so it builds with **0 LLM tokens** in pure AST mode.
