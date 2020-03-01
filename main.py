"""
Main run script.
"""
import pandas as pd
import pickle
import logging
import yaml
from pathlib import Path
from g1_distance_calculation.DistanceMatrixCalculator import DistanceMatrixCalculator
from g2_data_processing.DataProcessor import DataProcessor

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("MainExecution")


with open(Path("config.yml")) as file:
    CONFIG = yaml.full_load(file)

EXECUTION_PIPELINE = CONFIG["execution_pipeline"]


if __name__ == "__main__":

    if EXECUTION_PIPELINE["fetch_distances"]:

        input_data = pd.read_excel("data_input.xlsx")

        dmc = DistanceMatrixCalculator(input_data)

        DISTANCES = dmc.execute_pipeline(True)

    else:
        LOGGER.info("Loading saved distances")
        with open(
            Path("00_saved_data/saved_distances") / "latest_distances.pickle", "rb"
        ) as handle:
            DISTANCES = pickle.load(handle)

    data_processor = DataProcessor(DISTANCES)

    PROC_DISTANCES = data_processor.execute_pipeline()

    print(PROC_DISTANCES)
