import os
import pandas as pd
import requests


def fetch_esg_data(query):
    """
    Fetch ESG data from Climate TRACE API.
    """
    url = "https://api.climatetrace.org/v6/assets"
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers, params={"query": query})
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
        query = "AAPL"  # Example query (ticker or company name)
        esg_data = fetch_esg_data(query)
        save_to_parquet(esg_data, parquet_file)
        print(f"ESG data saved to {parquet_file}")
    else:
        print(f"{parquet_file} already exists. Skipping data fetch.")


if __name__ == "__main__":
    main()
