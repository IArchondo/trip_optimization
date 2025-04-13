"""Optimize schedule."""

import logging
from dataclasses import dataclass
from typing import Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore

from sched_opt.distance_calculation.distance_matrix_calculator import LocalizationData
from sched_opt.g0_utils.utils import ProblemDefinition

LOGGER = logging.getLogger("ScheduleOptimizer")


@dataclass
class DataModel:
    """Hold data model."""

    distance_matrix: list[list[float]]
    num_days: int
    hotel_index: int


class ScheduleOptimizer:
    """Define schedule optimizer."""

    def __init__(self, localization_data: LocalizationData, problem_definition: ProblemDefinition) -> None:
        """Initiate class."""
        self.localization_data = localization_data
        self.num_days = problem_definition.num_days
        self.hotel_index = problem_definition.hotel_index

        # Create data model
        self.data_model = self.__create_data_model(self.localization_data, self.num_days, self.hotel_index)

    def __create_data_model(self, localization_data: LocalizationData, num_days: int, hotel_index: int) -> DataModel:
        """Create data model for model."""
        distance_total = []
        for place in localization_data.places:
            distance: list[float] = []
            for destination in localization_data.places:
                if place == destination:
                    distance.append(10000000.0)
                else:
                    distance.append(localization_data.combination_distance_stay_dict[(place, destination)])

            distance_total.append(distance)

        return DataModel(distance_matrix=distance_total, num_days=num_days, hotel_index=hotel_index)

    def define_model(self, data_model: DataModel) -> dict[str, Any]:
        """Define model for solving."""
        manager = pywrapcp.RoutingIndexManager(
            len(data_model.distance_matrix),
            data_model.num_days,
            [data_model.hotel_index] * data_model.num_days,
            [data_model.hotel_index] * data_model.num_days,
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index: int, to_index: int) -> Any:
            """Return the distance between the two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data_model.distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Distance constraint
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            10,  # realistic max distance per day (change as needed)
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)

        # This tries to balance travel length across days
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # OPTIONAL: Enforce upper bounds more strictly:
        for day in range(data_model.num_days):
            distance_dimension.SetSpanUpperBoundForVehicle(20000, day)

        # Avoid skipping locations (disjunctions)
        penalty = 100000  # Big penalty to discourage skipping
        for node in range(len(self.localization_data.places)):
            if node == data_model.hotel_index:
                continue
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Search parameters: Add metaheuristics to improve solution
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.FromSeconds(10)

        return {
            "manager": manager,
            "routing": routing,
            "search_parameters": search_parameters,
        }

    def solve_problem(self, problem_dict: dict[str, Any]) -> pywrapcp.Assignment:
        """Solve problem."""
        solution = problem_dict["routing"].SolveWithParameters(problem_dict["search_parameters"])

        return solution

    def extract_solution(
        self, data_model: DataModel, problem_dict: dict[str, Any], solution: pywrapcp.Assignment
    ) -> dict[int, list[str]]:
        """Extract solution."""
        solution_dict = {}
        # distance_dimension = problem_dict["routing"].GetDimensionOrDie("Distance")

        for day in range(data_model.num_days):
            route = []
            index = problem_dict["routing"].Start(day)

            while not problem_dict["routing"].IsEnd(index):
                node_index = problem_dict["manager"].IndexToNode(index)
                route.append(self.localization_data.places[node_index])
                index = solution.Value(problem_dict["routing"].NextVar(index))

            # Add final stop (which will be hotel, because we forced it)
            node_index = problem_dict["manager"].IndexToNode(index)
            route.append(self.localization_data.places[node_index])

            # m = distance_dimension.CumulVar(index).Value()

            # LOGGER.info(f"Day {index}, Distance: {m}")

            solution_dict[day] = route
            LOGGER.info(f"Day {day + 1}: {route}")

        return solution_dict

    def execute_optimizer_pipeline(self) -> dict[int, list[str]]:
        """Execute whole optimizer pipeline."""
        problem_dict = self.define_model(self.data_model)
        solution = self.solve_problem(problem_dict)

        return self.extract_solution(self.data_model, problem_dict, solution)
