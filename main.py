"""
Main run script.
"""
import os
import pandas as pd
import pickle
import logging
import yaml
from shutil import copy as copy_file
from pathlib import Path
from datetime import datetime

from sched_opt.g0_utils.utils import load_problem_definition
from sched_opt.g1_distance_calculation.DistanceMatrixCalculator import (
    DistanceMatrixCalculator,
)
from sched_opt.g2_data_processing.DataProcessor import DataProcessor
from sched_opt.g3_schedule_optimization.ScheduleOptimizer import ScheduleOptimizer
from sched_opt.g4_schedule_visualization.ScheduleVisualizer import ScheduleVisualizer

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("MainExecution")


with open(Path("config.yml")) as file:
    CONFIG = yaml.full_load(file)

EXECUTION_PIPELINE = CONFIG["execution_pipeline"]


if __name__ == "__main__":

    CURRENT_RUN = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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

    SCHED_VIS = ScheduleVisualizer(PROC_DISTANCES, SOLUTION_DICT)
    os.makedirs(f"02_reports/{CURRENT_RUN}")

    SCHED_VIS.execute_visualizer_pipeline(CURRENT_RUN)

    copy_file(
        "03_notebook_templates/trip_report_template.ipynb",
        f"02_reports/{CURRENT_RUN}/trip_report_{CURRENT_RUN}.ipynb",
    )

    os.system(
        (
            "jupyter nbconvert --execute --no-input --no-prompt --to html "
            + f"02_reports/{CURRENT_RUN}/trip_report_{CURRENT_RUN}.ipynb"
        )
    )

    os.remove(f"02_reports/{CURRENT_RUN}/trip_report_{CURRENT_RUN}.ipynb")

