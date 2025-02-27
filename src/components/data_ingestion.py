import os
import sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pathlib import Path  # Fixed import
from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.data_access.phising_data import PhisingData
from src.utils.main_utlis import MainUtlis
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder, "data_ingestion")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtlis()

    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        """Export data from MongoDB and save as CSV files in the artifact directory."""
        try:
            logging.info("Exporting data from MongoDB...")
            raw_batch_files_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_batch_files_path, exist_ok=True)

            incoming_data = PhisingData(database_name=MONGO_DATABASE_NAME)

            logging.info(f"Saving exported data to: {raw_batch_files_path}")
            for collection_name, dataset in incoming_data.export_collection_as_dataframe():  # ✅ FIXED HERE
                logging.info(f"Saving {collection_name}, Shape: {dataset.shape}")
                feature_store_file_path = os.path.join(raw_batch_files_path, f"{collection_name}.csv")
                dataset.to_csv(feature_store_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> str:  # Changed return type to str
        """Initiate data ingestion and return the directory containing raw data."""
        logging.info("Starting data ingestion process...")
        try:
            self.export_data_into_raw_data_dir()
            logging.info("Data successfully exported from MongoDB.")
            return self.data_ingestion_config.data_ingestion_dir  # Returns a string path

        except Exception as e:
            raise CustomException(e, sys) from e
