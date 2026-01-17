import requests

# User's provided "OAuth" keys
CLIENT_ID = "AKSF5FVOKXYKG4ATGCJQBA"
CLIENT_SECRET = "MFRGGZDFMZTWQYLCMNSGKZTHNBQWEY3EMVTGO2DBMJRWIZLGM5UGCYTDMRSWMZ3I"

AUTH_URL = "https://authx.alpaca.markets/v1/oauth2/token"
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"

print("1. Requesting Token...")
try:
    auth_resp = requests.post(
        AUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )

    if auth_resp.status_code != 200:
        print(f"Auth Failed: {auth_resp.text}")
        exit(1)

    token = auth_resp.json().get("access_token")
    print(f"Token received: {token[:10]}...")

    print("\n2. Requesting Data (SPY)...")
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "symbols": "SPY",
        "timeframe": "1Day",
        "start": "2023-01-03T00:00:00Z",
        "end": "2023-01-04T00:00:00Z",
        "limit": 1,
    }

    data_resp = requests.get(DATA_URL, headers=headers, params=params)
    print(f"Status: {data_resp.status_code}")
    if data_resp.status_code == 200:
        print("Success! Data:")
        print(data_resp.json())
    else:
        print("Data Access Failed:")
        print(data_resp.text)

except Exception as e:
    print(f"Error: {e}")
