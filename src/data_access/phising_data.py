import sys 
import os 
import numpy as np
import pandas as pd
from typing import List
from pymongo import MongoClient
from database_connect import mongo_operation as monogo
from src.exception import CustomException


class PhisingData:
    def __init__(self, database_name: str):
        try:
            self.database_name = database_name
            self.mongo_url = os.getenv("client_url")  # ✅ Correct attribute name

            if not self.mongo_url:
                raise ValueError("MongoDB client URL is not set in environment variables.")
        
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_collection_names(self) -> List[str]:  # ✅ Renamed function
        """Retrieve collection names from MongoDB."""
        try:
            mongo_db_client = MongoClient(self.mongo_url)  # ✅ Fixed attribute name
            collection_names = mongo_db_client[self.database_name].list_collection_names()
            return collection_names
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_collection_data(self, collection_name: str) -> pd.DataFrame:
        """Fetch data from a specific collection and return it as a Pandas DataFrame."""
        try:
            mongo_connection = monogo(  # ✅ Fixed function call
                client_url=self.mongo_url,
                database_name=self.database_name,
                collection_name=collection_name
            )
            df = pd.DataFrame(list(mongo_connection.find()))  # ✅ Ensure data is converted to DataFrame

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def export_collection_as_dataframe(self):
        """Yield collection names and their data as Pandas DataFrames."""
        try:
            collections = self.get_collection_names()  # ✅ Fixed method call

            for collection_name in collections:
                df = self.get_collection_data(collection_name)
                yield collection_name, df  # ✅ Yielding as expected
            
        except Exception as e:
            raise CustomException(e, sys) from e
