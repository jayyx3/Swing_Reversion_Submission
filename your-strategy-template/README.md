# Swing Reversion Grid Strategy Template

Contest-ready swing trading bot that accumulates into pullbacks and scales out
on recoveries around a long-term anchor price. The grid widens automatically when
volatility rises, keeping risk controlled while staying active on Coinbase hourly
data (January–June 2024).

## Key Mechanics

- **Anchor SMA** – a long moving average defines a fair-value reference.
- **ATR lattice** – Average True Range sets ladder spacing so choppy markets
  loosen the grid while calmer regimes tighten it.
- **Inventory tracking** – each level records fills, enabling partial exits at
  the paired take-profit rung.
- **Optional RSI filter** – prevents over-trading when momentum sits in the
  middle of the range.
- **Cooldown** – throttles re-entries immediately after a fill.

Implementation lives in `your_strategy.py` and registers itself as `swing_grid`.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anchor_period` | 288 | SMA window anchoring the grid |
| `atr_period` | 56 | ATR window for ladder spacing |
| `grid_steps` | 4 | Levels per side of the anchor |
| `atr_multiplier` | 1.1 | Scales ATR to stretch spacing |
| `base_allocation` | 0.12 | Fraction of equity committed per rung |
| `max_position_fraction` | 0.35 | Cap on deployed capital |
| `min_order_notional` | 150 | Skip dust-sized orders |
| `cooldown_minutes` | 30 | Rest period after a fill |
| `use_rsi_filter` | false | Gate trades on RSI extremes |
| `min_history_bars` | 288 | Minimum bars required before trading |
| `preload_history_bars` | 420 | Bars fetched during `prepare()` to bypass warm-up |

Override via environment variable `BOT_STRATEGY_PARAMS`, for example (PowerShell):

```powershell
setx BOT_STRATEGY_PARAMS '{"grid_steps":5,"base_allocation":0.10}'
```

## Running Locally

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r base-bot-template/requirements.txt

$env:BOT_STRATEGY="swing_grid"
$env:BOT_SYMBOL="BTC-USD"
$env:BOT_STARTING_CASH="10000"
python swing-grid-submission/your-strategy-template/startup.py
```

## Docker Image

```powershell
docker build -t swing-grid-bot -f swing-grid-submission/your-strategy-template/Dockerfile .
docker run --rm -p 8080:8080 -p 3010:3010 `
  -e BOT_STRATEGY=swing_grid `
  -e BOT_SYMBOL=BTC-USD `
  -e BOT_STARTING_CASH=10000 `
  swing-grid-bot
```

## Deliverables Checklist

- `your_strategy.py` – core trading logic ✔️
- `startup.py` – local launch helper ✔️
- `Dockerfile` – container recipe ✔️
- `requirements.txt` – dependency notes ✔️
- `README.md` – documentation (this file) ✔️
