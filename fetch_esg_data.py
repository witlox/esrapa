import os
import pandas as pd
import requests
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fetch_esg_data.log"),
        logging.StreamHandler()
    ]
)


@dataclass
class Asset:
    id: str
    name: str
    sector: str
    country: str
    emissions: float
    year: int
    details: Optional[dict] = None


@dataclass
class EmissionsBySector:
    sector: str
    total_emissions: float
    year: int


@dataclass
class EmissionsByAsset:
    asset_id: str
    asset_name: str
    emissions: float
    year: int


@dataclass
class EmissionsByCountry:
    country: str
    total_emissions: float
    year: int


@dataclass
class EmissionsTrends:
    year: int
    emissions: float


@dataclass
class EmissionsForecast:
    year: int
    projected_emissions: float


@dataclass
class RatingDivergence:
    rater_id: str
    asset_id: str
    divergence_score: float


@dataclass
class InsuranceMarketDynamics:
    insurer_id: str
    asset_id: str
    premium: float
    claims: float


@dataclass
class GreenwashingEffects:
    asset_id: str
    greenwashing_score: float
    impact_on_ratings: float


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
    if "json" in response.headers.get("Content-Type"):
        response_data = response.json()
        logging.debug(f"response for {query}: {response_data}")
        if isinstance(response_data, list):
            # Map response data to appropriate dataclass instances
            if query == "emissions_by_sector":
                return [EmissionsBySector(**item) for item in response_data]
            elif query == "emissions_by_asset":
                return [
                    EmissionsByAsset(
                        asset_id=item["asset"]["id"],
                        asset_name=item["asset"]["name"],
                        emissions=item["emissions"],
                        year=item["year"]
                    ) for item in response_data
                ]
            elif query == "emissions_by_country":
                return [EmissionsByCountry(**item) for item in response_data]
            elif query == "emissions_trends":
                return [EmissionsTrends(**item) for item in response_data]
            elif query == "emissions_forecast":
                return [EmissionsForecast(**item) for item in response_data]
            elif query == "rating_divergence":
                return [
                    RatingDivergence(
                        rater_id=item["rater"]["id"],
                        asset_id=item["asset"]["id"],
                        divergence_score=item["divergence_score"]
                    ) for item in response_data
                ]
            elif query == "insurance_market_dynamics":
                return [
                    InsuranceMarketDynamics(
                        insurer_id=item["insurer"]["id"],
                        asset_id=item["asset"]["id"],
                        premium=item["premium"],
                        claims=item["claims"]
                    ) for item in response_data
                ]
            elif query == "greenwashing_effects":
                return [
                    GreenwashingEffects(
                        asset_id=item["asset"]["id"],
                        greenwashing_score=item["greenwashing_score"],
                        impact_on_ratings=item["impact_on_ratings"]
                    ) for item in response_data
                ]
            else:
                raise ValueError(f"Unknown query type: {query}")
        elif isinstance(response_data, dict):
            # Map single dictionary response to appropriate dataclass instance
            if query == "emissions_by_sector":
                return EmissionsBySector(**response_data)
            elif query == "emissions_by_asset":
                return EmissionsByAsset(**response_data)
            elif query == "emissions_by_country":
                return EmissionsByCountry(**response_data)
            elif query == "emissions_trends":
                return EmissionsTrends(**response_data)
            elif query == "emissions_forecast":
                return EmissionsForecast(**response_data)
            elif query == "rating_divergence":
                return RatingDivergence(**response_data)
            elif query == "insurance_market_dynamics":
                return InsuranceMarketDynamics(**response_data)
            elif query == "greenwashing_effects":
                return GreenwashingEffects(**response_data)
            else:
                raise ValueError(f"Unknown query type: {query}")
        else:
            raise ValueError("Unexpected response format")
    else:
        raise ValueError(
            f"Unexpected content type: {response.headers.get('Content-Type')}"
        )


def save_to_parquet(data, file_path):
    """
    Save data to a parquet file.
    """
    # Ensure data is properly unpacked into columns
    if isinstance(data, pd.DataFrame):
        data.to_parquet(file_path, index=False)
    else:
        raise ValueError("Data must be a pandas DataFrame to save as Parquet.")


def main():
    """
    Main function to fetch ESG data and save it to a parquet file.
    """
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

    for query in queries:
        parquet_file = f"{query}.parquet"
        if not os.path.exists(parquet_file):
            logging.info(f"Fetching ESG data for query: {query}")
            esg_data = fetch_esg_data(query)
            save_to_parquet(esg_data, parquet_file)
            logging.info(f"ESG data for {query} saved to {parquet_file}")
        else:
            logging.info(f"{parquet_file} already exists. Skipping data fetch.")


if __name__ == "__main__":
    main()
