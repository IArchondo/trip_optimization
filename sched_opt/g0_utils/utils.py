"""Hold utils for package."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ProblemDefinition:
    """Hold problem definition."""

    num_days: int
    hotel_index: int


def load_standard_date() -> datetime:
    """Load standard date from config file."""
    with open(Path("config.yml")) as file:
        config = yaml.full_load(file)

    standard_date_config = config["standard_time"]

    standard_date = datetime(
        year=standard_date_config["year"],
        month=standard_date_config["month"],
        day=standard_date_config["day"],
        hour=standard_date_config["hour"],
        minute=standard_date_config["minute"],
    )

    return standard_date


def load_problem_definition() -> ProblemDefinition:
    """Load problem definition from config file."""
    with open(Path("config.yml")) as file:
        config = yaml.full_load(file)

    problem_definition_config = config["problem_definition"]

    num_days = problem_definition_config["num_days"]
    hotel_index = problem_definition_config["hotel_index"]

    return ProblemDefinition(num_days=num_days, hotel_index=hotel_index)
