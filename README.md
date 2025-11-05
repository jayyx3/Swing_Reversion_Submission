# Swing Reversion Grid Submission (ENTRY by jayyx03)

## Repository Layout

```
swing-grid-submission/
├─ README.md
├─ analysis/
│  ├─ backtest_runner.py
│  ├─ configs/swing_grid_best.json
│  └─ cache/*.pkl          # optional Yahoo Finance fetch cache
├─ reports/
│  ├─ backtest-report.md
│  ├─ backtest_summary.json
│  ├─ btc_usd_trades.csv
│  ├─ btc_usd_equity_curve.csv
│  ├─ eth_usd_trades.csv
│  └─ eth_usd_equity_curve.csv
└─ your-strategy-template/
   ├─ Dockerfile
   ├─ README.md
   ├─ requirements.txt
   ├─ startup.py
   └─ your_strategy.py
```

Remove `analysis/cache/` before packaging if you prefer a smaller archive; the
backtester will regenerate those files automatically.

## Strategy Snapshot

- **Style**: Mean-reversion grid that scales into weakness and trims into
  strength around a long-term anchor SMA.
- **Key parameters**: anchor 288, ATR 56, grid steps 4, ATR multiplier 1.1,
  base allocation 12%, max position 35%, $150 minimum order.
- **Safeguards**: optional RSI filter, cooldown between fills, persistent rung
  bookkeeping for orderly exits, and eager history preloading so indicators are
  ready on the first live tick.

Raw summaries live in `reports/backtest-report.md` and
`reports/backtest_summary.json`; trade-by-trade and equity curves are stored in
companion CSVs.

## Local Development

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r base-bot-template/requirements.txt

$env:BOT_STRATEGY="swing_grid"
$env:BOT_SYMBOL="BTC-USD"
$env:BOT_STARTING_CASH="10000"
python swing-grid-submission/your-strategy-template/startup.py
```

Override defaults with environment variable `BOT_STRATEGY_PARAMS`, for example:

```powershell
$env:BOT_STRATEGY_PARAMS='{"grid_steps":5,"base_allocation":0.10}'
```

## Backtesting & Report Generation

```powershell
python swing-grid-submission/analysis/backtest_runner.py `
  --symbols BTC-USD,ETH-USD `
  --config-file swing-grid-submission/analysis/configs/swing_grid_best.json `
  --output swing-grid-submission/reports/backtest-report.md
```

The command refreshes every artifact inside `swing-grid-submission/reports/`:
`backtest-report.md`, `backtest_summary.json`, and both symbol CSV pairs.

## Container Build

```powershell
docker build -t swing-grid-bot -f swing-grid-submission/your-strategy-template/Dockerfile .
docker run --rm -p 8080:8080 -p 3010:3010 `
  -e BOT_STRATEGY=swing_grid `
  -e BOT_SYMBOL=BTC-USD `
  -e BOT_STARTING_CASH=10000 `
  swing-grid-bot
```


