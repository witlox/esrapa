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
    base_url = "https://api.climatetrace.org/v6"
    endpoints = {
        "emissions_by_sector": "/country/emissions",
        "emissions_by_asset": "/assets",
        "emissions_by_country": "/country/emissions",
        "emissions_trends": "/country/emissions",
        "emissions_forecast": "/country/emissions",
        "rating_divergence": "/assets",
        "insurance_market_dynamics": "/assets/emissions",
        "greenwashing_effects": "/assets/emissions"
    }
    if query not in endpoints:
        raise ValueError(f"Unknown query type: {query}")
    
    url = f"{base_url}{endpoints[query]}"
    headers = {
        "Accept": "application/json"
    }
    params = {}
    if query in ["emissions_by_sector", "emissions_by_country", "emissions_trends", "emissions_forecast"]:
        params = {
            "since": 2022,
            "to": datetime.now().year,
            "sector_optional": ["power", "transportation", "buildings"]
        }
    elif query in ["emissions_by_asset", "insurance_market_dynamics", "greenwashing_effects"]:
        params = {
            "year": 2022,
            "countries_csv_optional": ["USA", "CAN", "MEX"]
        }
    
    if query == "rating_divergence":
        source_id = params.get("sourceId")
        if not source_id:
            logging.error("Missing sourceId for rating_divergence query. Skipping request.")
            return []
        url = f"{base_url}/assets/{source_id}"
        if "sourceId" in params:
            del params["sourceId"]  # Remove sourceId from params as it's part of the URL

    response = requests.get(
        url,
        headers=headers,
        params=params
    )
    response.raise_for_status()
    if "json" in response.headers.get("Content-Type"):
        response_data = response.json()
        logging.info(f"Raw response for {query}: {response_data}")
        results = []
        if isinstance(response_data, list):
            # Map response data to appropriate dataclass instances
            if query == "emissions_by_sector":
                results.extend([
                    EmissionsBySector(
                        sector=item.get("Sector"),
                        total_emissions=item.get("Emissions"),
                        year=item.get("Year")
                    ) for item in response_data
                ])
            elif query == "emissions_by_asset":
                results.extend([
                    EmissionsByAsset(
                        asset_id=item.get("id"),
                        asset_name=item.get("description"),
                        emissions=item.get("emissions")[0].get("quantity"),
                        year=item.get("emissions")[0].get("year")
                    ) for item in response_data
                ])
            elif query == "emissions_by_country":
                results.extend([
                    EmissionsByCountry(
                        country=item.get("Country"),
                        total_emissions=item.get("Emissions"),
                        year=item.get("Year")
                    ) for item in response_data
                ])
            elif query == "emissions_trends":
                results.extend([
                    EmissionsTrends(
                        year=item.get("year"),
                        emissions=item.get("emissions")[0].get("quantity")
                    ) for item in response_data
                ])
            elif query == "emissions_forecast":
                results.extend([
                    EmissionsForecast(
                        year=item.get("year"),
                        projected_emissions=item.get("emissions")[0].get("quantity")
                    ) for item in response_data
                ])
            elif query == "rating_divergence":
                results.extend([
                    RatingDivergence(
                        rater_id=item.get("rater_id"),
                        asset_id=item.get("id"),
                        divergence_score=item.get("divergence_score")
                    ) for item in response_data
                ])
            elif query == "insurance_market_dynamics":
                results.extend([
                    InsuranceMarketDynamics(
                        insurer_id=item.get("insurer_id"),
                        asset_id=item.get("id"),
                        premium=item.get("premium"),
                        claims=item.get("claims")
                    ) for item in response_data
                ])
            elif query == "greenwashing_effects":
                results.extend([
                    GreenwashingEffects(
                        asset_id=item.get("id"),
                        greenwashing_score=item.get("greenwashing_score"),
                        impact_on_ratings=item.get("impact_on_ratings")
                    ) for item in response_data
                ])
            else:
                raise ValueError(f"Unknown query type: {query}")
        elif isinstance(response_data, dict):
            # Map single dictionary response to appropriate dataclass instance
            if query == "emissions_by_sector":
                results.append(EmissionsBySector(
                    sector=response_data.get("sector"),
                    total_emissions=response_data.get("total_emissions"),
                    year=response_data.get("year")
                ))
            elif query == "emissions_by_asset":
                results.append(EmissionsByAsset(
                    asset_id=response_data.get("asset", {}).get("id"),
                    asset_name=response_data.get("asset", {}).get("name"),
                    emissions=response_data.get("emissions"),
                    year=response_data.get("year")
                ))
            elif query == "emissions_by_country":
                results.append(EmissionsByCountry(
                    country=response_data.get("country"),
                    total_emissions=response_data.get("total_emissions"),
                    year=response_data.get("year")
                ))
            elif query == "emissions_trends":
                results.append(EmissionsTrends(
                    year=response_data.get("year"),
                    emissions=response_data.get("emissions")
                ))
            elif query == "emissions_forecast":
                results.append(EmissionsForecast(
                    year=response_data.get("year"),
                    projected_emissions=response_data.get("projected_emissions")
                ))
            elif query == "rating_divergence":
                results.append(RatingDivergence(
                    rater_id=response_data.get("rater", {}).get("id"),
                    asset_id=response_data.get("asset", {}).get("id"),
                    divergence_score=response_data.get("divergence_score")
                ))
            elif query == "insurance_market_dynamics":
                results.append(InsuranceMarketDynamics(
                    insurer_id=response_data.get("insurer", {}).get("id"),
                    asset_id=response_data.get("asset", {}).get("id"),
                    premium=response_data.get("premium"),
                    claims=response_data.get("claims")
                ))
            elif query == "greenwashing_effects":
                results.append(GreenwashingEffects(
                    asset_id=response_data.get("asset", {}).get("id"),
                    greenwashing_score=response_data.get("greenwashing_score"),
                    impact_on_ratings=response_data.get("impact_on_ratings")
                ))
            else:
                raise ValueError(f"Unknown query type: {query}")
        else:
            raise ValueError("Unexpected response format")
        return results
    else:
        raise ValueError(
            f"Unexpected content type: {response.headers.get('Content-Type')}. Raw response: {response.text}"
        )


def save_to_parquet(data, file_path):
    """
    Save data to a parquet file. Convert dataclass instances to DataFrame if necessary.
    """
    # Ensure data is properly unpacked into columns
    if isinstance(data, list) and all(isinstance(item, (EmissionsBySector, EmissionsByAsset, EmissionsByCountry, EmissionsTrends, EmissionsForecast, RatingDivergence, InsuranceMarketDynamics, GreenwashingEffects)) for item in data):
        # Convert list of dataclass instances to DataFrame
        data_dicts = []
        for item in data:
            item_dict = item.__dict__.copy()
            # Flatten nested dictionaries if present
            for key, value in list(item_dict.items()):
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        item_dict[f"{key}_{sub_key}"] = sub_value
                    del item_dict[key]
            data_dicts.append(item_dict)
        logging.info(f"data dicts: {data_dicts}")
        df = pd.DataFrame(data_dicts)
        if not df.empty:
            df.to_parquet(file_path, index=False)
            logging.info(f"Data successfully written to {file_path}")
        else:
            logging.warning(f"No data to write to {file_path}")
    elif isinstance(data, pd.DataFrame):
        data.to_parquet(file_path, index=False)
    else:
        raise ValueError("Data must be a pandas DataFrame or a list of dataclass instances to save as Parquet.")


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
            if esg_data:
                logging.info(f"ESG data fetched successfully for query: {query}")
            else:
                logging.warning(f"No data fetched for query: {query}")
            save_to_parquet(esg_data, parquet_file)
            logging.info(f"ESG data saved to {parquet_file}")
        else:
            logging.info(f"{parquet_file} already exists. Skipping data fetch.")


if __name__ == "__main__":
    main()
