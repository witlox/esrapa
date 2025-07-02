import os
import pandas as pd
import requests


def fetch_esg_data(query):
    """
    Fetch ESG data from Climate TRACE API.

    Relevant queries based on divergence.py, dynamics.py, and efficiency.py:
    - "emissions_by_sector" for sector-level emissions data
    - "emissions_by_asset" for asset-level emissions data
    - "emissions_by_country" for country-level emissions data
    - "emissions_trends" for historical emissions trends
    - "emissions_forecast" for projected emissions data
    - "rating_divergence" for ESG rating divergence analysis
    - "insurance_market_dynamics" for insurance market dynamics
    - "greenwashing_effects" for greenwashing and its impact on ESG ratings
    """
    """
    Fetch ESG data from Climate TRACE API.

    Relevant queries based on divergence.py, dynamics.py, and efficiency.py:
    - "emissions_by_sector" for sector-level emissions data
    - "emissions_by_asset" for asset-level emissions data
    - "emissions_by_country" for country-level emissions data
    - "emissions_trends" for historical emissions trends
    - "emissions_forecast" for projected emissions data
    """
    """
    Fetch ESG data from Climate TRACE API.
    """
    url = "https://api.climatetrace.org/v6/assets"
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(
        url,
        headers=headers,
        params={
            "query": query,
            "start_year": 2022,
            "end_year": datetime.now().year,
            "detail_level": "high"
        }
    )
    response.raise_for_status()
    if response.headers.get("Content-Type") == "application/json":
        return response.json()
    else:
        raise ValueError(
            f"Unexpected content type: {response.headers.get('Content-Type')}"
        )


def save_to_parquet(data, file_path):
    """
    Save data to a parquet file.
    """
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)


def main():
    """
    Main function to fetch ESG data and save it to a parquet file.
    """
    parquet_file = "esg_data.parquet"
    if not os.path.exists(parquet_file):
        print("Fetching ESG data from Climate TRACE...")
        queries = [
            "emissions_by_sector",
            "emissions_by_asset",
            "emissions_by_country",
            "emissions_trends",
            "emissions_forecast",
            "rating_divergence",
            "insurance_market_dynamics",
            "greenwashing_effects"
        ]
    
        all_data = {}
        for query in queries:
            print(f"Fetching ESG data for query: {query}")
            esg_data = fetch_esg_data(query)
            all_data[query] = esg_data
    
        save_to_parquet(all_data, parquet_file)
        print(f"ESG data saved to {parquet_file}")
    else:
        print(f"{parquet_file} already exists. Skipping data fetch.")


if __name__ == "__main__":
    main()
