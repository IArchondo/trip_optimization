"""Hold utils for package."""

from collections import defaultdict
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
