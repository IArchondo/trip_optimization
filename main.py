"""
Main run script.
"""
import pandas as pd
import pickle
import logging
import yaml
from pathlib import Path
from g0_utils.utils import load_problem_definition
from g1_distance_calculation.DistanceMatrixCalculator import DistanceMatrixCalculator
from g2_data_processing.DataProcessor import DataProcessor
from g3_schedule_optimization.ScheduleOptimizer import ScheduleOptimizer

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger("MainExecution")


with open(Path("config.yml")) as file:
    CONFIG = yaml.full_load(file)

EXECUTION_PIPELINE = CONFIG["execution_pipeline"]


if __name__ == "__main__":

    if EXECUTION_PIPELINE["fetch_distances"]:

        input_data = pd.read_excel("data_input.xlsx")

        dmc = DistanceMatrixCalculator(input_data)

        DISTANCES = dmc.execute_pipeline(True)

        with open(
            Path("00_saved_data/saved_distances") / "latest_distances_2.pickle", "wb"
        ) as handle:
            pickle.dump(DISTANCES, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        LOGGER.info("Loading saved distances")
        with open(
            Path("00_saved_data/saved_distances") / "latest_distances_2.pickle", "rb"
        ) as handle:
            DISTANCES = pickle.load(handle)

    data_processor = DataProcessor(DISTANCES)

    PROC_DISTANCES = data_processor.execute_pipeline()

    PROBLEM_DEFINITION = load_problem_definition()

    SCHED_OPT = ScheduleOptimizer(PROC_DISTANCES, PROBLEM_DEFINITION)

    SOLUTION_DICT = SCHED_OPT.execute_optimizer_pipeline()

