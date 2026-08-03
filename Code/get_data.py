
import requests
import csv
from datetime import datetime

API_KEY = "LAKMMKPezDEBOvhY1HYPagFUPOrJbuTuR2uW3aHG" 

def fetch_and_save(region_code="CISO", start_date="2024-01-01", end_date="2025-12-31", filename="real_data.csv"):
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_records = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region_code,
            "facets[type][]": "D",
            "start": start_date + "T00",
            "end": end_date + "T00",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": page_size,
            "offset": offset,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        records = response.json()["response"]["data"]
        print(f"  fetched {len(records)} rows at offset {offset}")
        if not records:
            break
        all_records.extend(records)
        offset += page_size
        if len(records) < page_size:
            break

    print(f"\nSaved {len(all_records)} total rows to {filename}")
    return filename