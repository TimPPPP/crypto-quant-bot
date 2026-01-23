#!/bin/bash
# Check Coinbase ingestion progress

echo "=== Data Ingestion Progress ==="
echo ""

# Total count
TOTAL=$(curl -s "http://localhost:9000/exec?query=SELECT%20count(*)%20FROM%20candles_1m" | grep -o '"dataset":\[\[[0-9]*' | grep -o '[0-9]*')
echo "Total candles: $(printf "%'d" $TOTAL)"

# Coins count
COINS=$(curl -s "http://localhost:9000/exec?query=SELECT%20count(DISTINCT%20symbol)%20FROM%20candles_1m" | grep -o '"dataset":\[\[[0-9]*' | grep -o '[0-9]*')
echo "Coins ingested: $COINS / 40"

# Date range
echo ""
echo "=== Date Range ==="
curl -s "http://localhost:9000/exec?query=SELECT%20min(timestamp),%20max(timestamp)%20FROM%20candles_1m"

# Per coin
echo ""
echo ""
echo "=== Per Coin ==="
curl -s "http://localhost:9000/exec?query=SELECT%20symbol,%20count(*)%20as%20candles%20FROM%20candles_1m%20GROUP%20BY%20symbol%20ORDER%20BY%20candles%20DESC"

echo ""
echo ""
echo "=== Latest Log ==="
tail -3 /tmp/claude/tasks/bf7a8f8.output 2>/dev/null || echo "Log file not found"
