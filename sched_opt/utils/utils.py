"""Hold utils for package."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from sched_opt.elements import Trip


@dataclass
class ProblemDefinition:
    """Hold problem definition."""

    num_days: int
    hotel_index: int
    total_hours_in_day: int
    solve_time_in_minutes: int
    use_gurobi: bool
    reduce_no_of_possible_trips: float


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
    total_hours_in_day = problem_definition_config["total_hours_in_day"]
    solve_time_in_minutes = problem_definition_config["solve_time_in_minutes"]
    use_gurobi = problem_definition_config["use_gurobi"]
    reduce_no_of_possible_trips = problem_definition_config["reduce_no_of_possible_trips"]

    return ProblemDefinition(
        num_days=num_days,
        hotel_index=hotel_index,
        total_hours_in_day=total_hours_in_day,
        solve_time_in_minutes=solve_time_in_minutes,
        use_gurobi=use_gurobi,
        reduce_no_of_possible_trips=reduce_no_of_possible_trips,
    )


def order_route_from_start(unordered_route: list[tuple[str, str]], hotel: str) -> list[tuple[str, str]]:
    """Order a route of tuples."""
    # Build adjacency list: from_node -> list of (to_node, index)
    adj = defaultdict(list)
    for i, (frm, to) in enumerate(unordered_route):
        adj[frm].append((to, i))

    def backtrack(path: list[tuple[str, str]], used: set[int], current: str) -> list[tuple[str, str]]:
        """Backtrack route."""
        if len(used) == len(unordered_route):
            return path

        for next_dest, idx in adj[current]:
            if idx in used:
                continue
            used.add(idx)
            path.append((current, next_dest))
            result = backtrack(path, used, next_dest)
            if result:
                return result
            path.pop()
            used.remove(idx)

        raise ValueError("Route does not make sense")

    return backtrack([], set(), hotel)


def transform_route_to_list_of_destinations(route: list[tuple[str, str]], hotel: str) -> list[str]:
    """Transform a route to a list of destinations."""
    return [origin for origin, _ in route] + [hotel]


def create_duration_matrix(trips: list[Trip]) -> pd.DataFrame:
    """Create duration matrix for trips."""
    # Get all unique places
    places = sorted({trip.origin for trip in trips} | {trip.destination for trip in trips})

    # Initialize a DataFrame with NaNs
    matrix = pd.DataFrame(index=places, columns=places, dtype=float)

    # Fill in the matrix
    for trip in trips:
        matrix.loc[trip.origin, trip.destination] = trip.duration

    return matrix
