FROM python:3.11-slim

WORKDIR /app

COPY ../../base-bot-template /app/base
COPY . /app/strategy

WORKDIR /app

RUN pip install --no-cache-dir -r base/requirements.txt

ENV BOT_STRATEGY=swing_grid \
    BOT_SYMBOL=BTC-USD \
    BOT_STARTING_CASH=10000

CMD ["python", "strategy/startup.py"]
