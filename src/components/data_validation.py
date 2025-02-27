import os
import numpy as np
import pandas as pd
import sys
import json
import shutil
import re
from pathlib import Path
from dataclasses import dataclass

from src.constants import artifact_folder
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utlis import MainUtlis

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(artifact_folder, 'data_validation')
    valid_data_dir: str = os.path.join(data_validation_dir, 'validated')
    invalid_data_dir: str = os.path.join(data_validation_dir, 'invalid')
    schema_config_file_path: str = os.path.join('config', 'training_schema.json')


class DataValidation:
    def __init__(self, raw_data_store_dir: Path):
        self.raw_data_store_dir = raw_data_store_dir
        self.data_validation_config = DataValidationConfig()
        self.utils = MainUtlis()

    def ValueFromSchema(self):
        """Extract relevant information from schema file."""
        try:
            logging.info("Loading schema configuration...")
            if not os.path.exists(self.data_validation_config.schema_config_file_path):
                raise FileNotFoundError(f"Schema file not found: {self.data_validation_config.schema_config_file_path}")

            with open(self.data_validation_config.schema_config_file_path, 'r') as f:
                schema = json.load(f)

            return (
                schema.get('LengthOfDateStampInFile', 8),
                schema.get('LengthOfTimeStampInFile', 6),
                schema.get('ColName', {}),
                schema.get('NumberOfColumns', 31)
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_file_name(self, file_path: str, length_of_date_stamp: int, length_of_time_stamp: int) -> bool:
        """Validate raw file names using regex."""
        try:
            file_name = os.path.basename(file_path)
            regex = rf"^phishing_\d{{{length_of_date_stamp}}}_\d{{{length_of_time_stamp}}}\.csv$"

            if re.match(regex, file_name):
                parts = file_name.replace('.csv', '').split('_')
                return len(parts[1]) == length_of_date_stamp and len(parts[2]) == length_of_time_stamp
            return False

        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_no_of_columns(self, file_path: str, schema_no_of_columns: int) -> bool:
        """Validate the number of columns in the raw file."""
        try:
            dataframe = pd.read_csv(file_path)
            return len(dataframe.columns) == schema_no_of_columns

        except Exception as e:
            raise CustomException(e, sys) from e

    def validating_missing_values_in_whole_column(self, file_path: str) -> bool:
        """Check for missing values in entire columns."""
        try:
            dataframe = pd.read_csv(file_path)
            return not any(dataframe[col].isnull().all() for col in dataframe.columns)

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_raw_batch_files_path(self) -> list:
        """Return list of raw file paths."""
        try:
            if not os.path.exists(self.raw_data_store_dir):
                logging.error(f"Raw data directory not found: {self.raw_data_store_dir}")
                return []

            raw_batch_files_names = os.listdir(self.raw_data_store_dir)
            if not raw_batch_files_names:
                logging.warning("No files found in the raw data directory.")

            return [os.path.join(self.raw_data_store_dir, file_name) for file_name in raw_batch_files_names]

        except Exception as e:
            raise CustomException(e, sys) from e

    def move_raw_files_to_validation_dir(self, src_path: str, dest_path: str):
        """Move validated files to validation directory."""
        try:
            os.makedirs(dest_path, exist_ok=True)
            file_name = os.path.basename(src_path)

            if file_name not in os.listdir(dest_path):
                shutil.move(src_path, os.path.join(dest_path, file_name))
                logging.info(f"Moved {file_name} to {dest_path}")

        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_raw_files(self) -> bool:
        """Validate raw files based on file name, column count, and missing values."""
        try:
            raw_batch_files_paths = self.get_raw_batch_files_path()

            if not raw_batch_files_paths:
                logging.error("No raw batch files found for validation.")
                return False

            length_of_date_stamp, length_of_time_stamp, column_name, no_of_columns = self.ValueFromSchema()
            validated_files = 0

            for raw_file_path in raw_batch_files_paths:
                file_valid = (
                    self.validate_file_name(raw_file_path, length_of_date_stamp, length_of_time_stamp) and
                    self.validate_no_of_columns(raw_file_path, no_of_columns) and
                    self.validating_missing_values_in_whole_column(raw_file_path)
                )

                dest_dir = (
                    self.data_validation_config.valid_data_dir if file_valid else self.data_validation_config.invalid_data_dir
                )

                self.move_raw_files_to_validation_dir(raw_file_path, dest_dir)

                if file_valid:
                    validated_files += 1

            return validated_files > 0  # Returns True if at least one file is validated

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self):
        """Initiate data validation process."""
        logging.info("Starting data validation process.")

        try:
            if self.validate_raw_files():
                valid_data_dir = self.data_validation_config.valid_data_dir
                logging.info(f"Validation successful. Valid data stored at: {valid_data_dir}")
                return valid_data_dir
            else:
                logging.error("No valid data files found. Stopping pipeline.")
                raise Exception("No valid data files found. Stopping pipeline.")

        except Exception as e:
            raise CustomException(e, sys) from e
