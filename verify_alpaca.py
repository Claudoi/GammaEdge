import requests

KEY_ID = "AKSF5FVOKXYKG4ATGCJQBA"
SECRET_KEY = "MFRGGZDFMZTWQYLCMNSGKZTHNBQWEY3EMVTGO2DBMJRWIZLGM5UGCYTDMRSWMZ3I"
BASE_URL = "https://data.alpaca.markets"

headers = {"APCA-API-KEY-ID": KEY_ID, "APCA-API-SECRET-KEY": SECRET_KEY}

# Try to fetch 1 day of bars for SPY
url = f"{BASE_URL}/v2/stocks/bars"
params = {
    "symbols": "SPY",
    "timeframe": "1Day",
    "start": "2023-01-03T00:00:00Z",
    "end": "2023-01-04T00:00:00Z",
    "limit": 1,
}

print(f"Testing credentials against {BASE_URL}...")
try:
    resp = requests.get(url, headers=headers, params=params)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Success! Data received:")
        print(resp.json())
    else:
        print("Failed:")
        print(resp.text)
except Exception as e:
    print(f"Error: {e}")
