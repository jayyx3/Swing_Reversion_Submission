Swing Reversion Grid Strategy
Overview
This strategy implements a Swing Reversion Grid approach. It assumes that prices oscillate around a long-term mean (Moving Average).

Logic
Anchor: Calculates a 50-period Simple Moving Average (SMA).
Grid: Defines buy levels below the SMA at fixed percentage steps (e.g., -2%, -4%, -6%).
Entry: Buys when price dips to these levels, scaling into weakness.
Exit: Sells all positions when price reverts to the SMA (Mean Reversion) or exceeds it.
Risk Management:
Max Position Size: 55% of portfolio (Contest Rule).
Stop Loss: 15% drop from entry price.
Configuration
ma_period: Period for the SMA (default: 50).
grid_step_pct: Percentage step between grid levels (default: 0.02).
max_grid_levels: Maximum number of grid levels (default: 5).
position_size_pct: Size of each trade relative to equity (default: 0.05).
How to Run
Use the provided startup.py or Docker container.


https://www.freelancer.com/contest/trading-strategy-contest-build-the-most-profitable-bot-2649140 - $1000
