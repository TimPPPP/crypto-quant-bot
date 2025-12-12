import time
import socket
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants
from src.utils.universe import get_liquid_universe

# --- CONFIGURATION ---
QUESTDB_HOST = 'localhost'
QUESTDB_PORT = 9009  # Influx Line Protocol (ILP) Port
TIMEFRAME = '1h'     # 1 Hour candles
LOOKBACK_CANDLES = 5000 # Max allowed by Hyperliquid API

class QuestDBSender:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_candle(self, symbol, timestamp_ms, o, h, l, c, v):
        # ILP Format: candles,symbol=BTC open=90000.0,high=... 1630000000000000000
        msg = (f"candles,symbol={symbol} "
               f"open={o},high={h},low={l},close={c},volume={v} "
               f"{timestamp_ms}000000\n")
        self.sock.sendall(msg.encode())

    def send_funding(self, symbol, timestamp_ms, rate, premium):
        msg = (f"funding,symbol={symbol} "
               f"rate={rate},premium={premium} "
               f"{timestamp_ms}000000\n")
        self.sock.sendall(msg.encode())

    def close(self):
        self.sock.close()

def main():
    # 1. Setup Connections
    print("Connecting to Hyperliquid & QuestDB...")
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    db = QuestDBSender(QUESTDB_HOST, QUESTDB_PORT)
    
    # 2. Get Universe (The list you just saw)
    coins = get_liquid_universe(top_n=30) 
    print(f"Target Universe: {coins}")

    # 3. Loop and Download
    for coin in coins:
        print(f"Processing {coin}...")
        
        # --- A. Fetch Candles ---
        # Note: endTime is Optional. If omitted, fetches most recent.
        try:
            candles = info.candles_snapshot(coin, TIMEFRAME, 0, int(time.time()*1000))
        except Exception as e:
            print(f"Failed to fetch candles for {coin}: {e}")
            continue

        print(f"  > Found {len(candles)} candles. Ingesting...")
        for c in candles:
            # c = {'t': timestamp, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume, ...}
            db.send_candle(
                symbol=coin,
                timestamp_ms=c['t'],
                o=c['o'], h=c['h'], l=c['l'], c=c['c'], v=c['v']
            )

        # --- B. Fetch Funding History ---
        # Note: Funding usually comes every hour on Hyperliquid
        try:
            # startTime is required. Let's ask for last ~6 months (approx 1.5e10 ms)
            start_time = int((time.time() - 15552000) * 1000) 
            funding_data = info.funding_history(coin, start_time)
        except Exception as e:
            print(f"  > Warning: No funding history for {coin} ({e})")
            funding_data = []

        print(f"  > Found {len(funding_data)} funding records. Ingesting...")
        for f in funding_data:
            # f = {'coin': 'BTC', 'fundingRate': '0.0001', 'premium': '...', 'time': 163...}
            db.send_funding(
                symbol=coin,
                timestamp_ms=f['time'],
                rate=f['fundingRate'],
                premium=f['premium']
            )
            
        # Respect Rate Limits (Hyperliquid is generous, but let's be safe)
        time.sleep(0.5)

    db.close()
    print("Done! Data lake hydrated.")

if __name__ == "__main__":
    main()