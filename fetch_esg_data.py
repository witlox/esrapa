import os
import pandas as pd
import requests

def fetch_msci_esg_data(api_url, api_key):
    """
    Fetch ESG data from MSCI's API.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    return response.json()

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
    parquet_file = "msci_esg_data.parquet"
    if not os.path.exists(parquet_file):
        print("Fetching ESG data from MSCI...")
        api_url = "https://api.msci.com/esg-ratings"  # Replace with actual API URL
        api_key = "your_api_key_here"  # Replace with your actual API key
        esg_data = fetch_msci_esg_data(api_url, api_key)
        save_to_parquet(esg_data, parquet_file)
        print(f"ESG data saved to {parquet_file}")
    else:
        print(f"{parquet_file} already exists. Skipping data fetch.")

if __name__ == "__main__":
    main()
