import os
import pandas as pd
import requests

def fetch_esg_data(source_name, query):
    """
    Fetch ESG data from various free sources.
    """
    sources = {
        "WWF Risk Filter Suite": "https://riskfilter.org/",
        "MSCI ESG Fund Ratings": "https://www.msci.com/our-solutions/esg-investing/esg-fund-ratings-climate-search-tool",
        "MSCI ESG Ratings": "https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool",
        "Sustainalytics ESG Risk Ratings": "https://www.sustainalytics.com/esg-ratings",
        "Refinitiv ESG Scores": "https://www.refinitiv.com/en/sustainable-finance/esg-scores",
        "SASB Materiality Finder": "https://www.sasb.org/standards/materiality-finder/find/",
        "S&P Global ESG Scores": "https://www.spglobal.com/esg/solutions/data-intelligence-esg-scores",
        "UN Data": "https://data.un.org/default.aspx",
        "World Bank Open Data": "https://data.worldbank.org/",
        "Sustainable Development Report": "https://dashboards.sdgindex.org/profiles"
    }
    
    if source_name not in sources:
        raise ValueError(f"Source '{source_name}' is not supported.")
    
    url = sources[source_name]
    response = requests.get(url, params={"query": query})
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
    parquet_file = "esg_data.parquet"
    if not os.path.exists(parquet_file):
        print("Fetching ESG data from free sources...")
        source_name = "Sustainalytics ESG Risk Ratings"  # Example source
        query = "AAPL"  # Example query (ticker or company name)
        esg_data = fetch_esg_data(source_name, query)
        save_to_parquet(esg_data, parquet_file)
        print(f"ESG data saved to {parquet_file}")
    else:
        print(f"{parquet_file} already exists. Skipping data fetch.")

if __name__ == "__main__":
    main()
