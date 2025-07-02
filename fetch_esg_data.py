import os
import pandas as pd
import requests


def fetch_esg_data(source, query):
    """
    Fetch ESG data from CDP or Climate TRACE.
    """
    sources = {
        "CDP": "https://www.cdp.net/en/data",
        "Climate TRACE": "https://climatetrace.org/data"
    }
    
    if source not in sources:
        raise ValueError(f"Source '{source}' is not supported.")
    
    url = sources[source]
    response = requests.get(url, params={"query": query})
    response.raise_for_status()
    if response.headers.get("Content-Type") == "application/json":
        return response.json()
    elif response.headers.get("Content-Type").startswith("text/html"):
        print(f"Received HTML content from {url}. Saving for debugging.")
        with open(f"debug_response_{source}.html", "w") as f:
            f.write(response.text)
        raise ValueError(
            f"Unexpected HTML content received from {url}. Check debug_response_{source}.html for details."
        )
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
        print("Fetching ESG data from CDP and Climate TRACE...")
        query = "AAPL"  # Example query (ticker or company name)
        esg_data_cdp = fetch_esg_data("CDP", query)
        esg_data_climate_trace = fetch_esg_data("Climate TRACE", query)
        
        # Combine data from both sources
        combined_data = {
            "CDP": esg_data_cdp,
            "Climate TRACE": esg_data_climate_trace
        }
        
        save_to_parquet(combined_data, parquet_file)
        print(f"ESG data saved to {parquet_file}")
    else:
        print(f"{parquet_file} already exists. Skipping data fetch.")


if __name__ == "__main__":
    main()
