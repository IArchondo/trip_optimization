"""Main run script."""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from sched_opt.data_processing.data_processor import DataProcessor
from sched_opt.distance_calculation.distance_matrix_calculator import (
    DistanceMatrixCalculator,
)
from sched_opt.g0_utils.utils import load_problem_definition
from sched_opt.schedule_optimization.gurobi_optimizer import run_new_solver
from sched_opt.schedule_visualization.new_schedule_visualizer import visualize_output

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("MainExecution")


with open(Path("config.yml")) as file:
    config = yaml.full_load(file)

execution_pipeline = config["execution_pipeline"]


if __name__ == "__main__":
    current_run = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if execution_pipeline["fetch_distances"]:
        input_data = pd.read_excel("data_input copy.xlsx")

        dmc = DistanceMatrixCalculator(input_data)

        distances = dmc.execute_pipeline()

        with open(Path("00_saved_data/saved_distances") / "latest_distances_test.pickle", "wb") as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        LOGGER.info("Loading saved distances")
        with open(Path("00_saved_data/saved_distances") / "latest_distances_test.pickle", "rb") as handle:
            distances = pickle.load(handle)

    data_processor = DataProcessor()

    proc_distances = data_processor.execute_pipeline(distances)

    problem_definition = load_problem_definition()

    model_inputs, solution_dict = run_new_solver(proc_distances, problem_definition)

    os.makedirs(f"02_reports/{current_run}")

    visualize_output(model_inputs, solution_dict, current_run)

    # copy_file(
    #     "03_notebook_templates/trip_report_template.ipynb",
    #     f"02_reports/{current_run}/trip_report_{current_run}.ipynb",
    # )

    # os.system(
    #     "jupyter nbconvert --execute --no-input --no-prompt --to html "
    #     + f"02_reports/{current_run}/trip_report_{current_run}.ipynb"
    # )

    # os.remove(f"02_reports/{current_run}/trip_report_{current_run}.ipynb")
