# alphapoly

Polymarket alpha detection: finds conditional probability arbitrage across related prediction markets.

## Setup

```bash
uv sync
cp .env.example .env  # Add OPENROUTER_API_KEY
```

## Usage

```bash
# Run pipeline (fetches markets, extracts entities, builds graph, detects alpha)
uv run poly run

# Start API + dashboard
uv run poly serve
# Open http://localhost:8000

# Commands
uv run poly run          # Incremental (new events only)
uv run poly run --full   # Full reprocess
uv run poly run state    # Check pipeline status
```

## Output

Results in `data/_live/`:
- `opportunities.json` — detected alpha (price discrepancies between related markets)
- `events.json` — processed market data
- `graph.json` — entity relationship graph
