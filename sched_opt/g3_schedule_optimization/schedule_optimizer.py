"""Hold schedule optimizer."""

import logging
from dataclasses import dataclass
from typing import Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore

from sched_opt.g0_utils.utils import ProblemDefinition
from sched_opt.g1_distance_calculation.distance_matrix_calculator import LocalizationData

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

        ## create data model
        self.data_model = self.__create_data_model(self.localization_data, self.num_days, self.hotel_index)

    def __create_data_model(self, localization_data: LocalizationData, num_days: int, hotel_index: int) -> DataModel:
        """Create data model for model."""
        distance_total = []
        for place in localization_data.places:
            distance: list[float] = []
            for destination in localization_data.places:
                if place == destination:
                    distance.append(0.0)
                else:
                    distance.append(localization_data.combination_distance_stay_dict[(place, destination)])

            distance_total.append(distance)

        return DataModel(distance_matrix=distance_total, num_days=num_days, hotel_index=hotel_index)

    def define_model(self, data_model: DataModel) -> dict[str, Any]:
        """Define model for solving."""
        manager = pywrapcp.RoutingIndexManager(
            len(data_model.distance_matrix),
            data_model.num_days,
            data_model.hotel_index,
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index: int, to_index: int) -> Any:
            """Return the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data_model.distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            30000000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        output_dict = {
            "manager": manager,
            "routing": routing,
            "search_parameters": search_parameters,
        }

        return output_dict

    def solve_problem(self, problem_dict: dict[str, Any]) -> pywrapcp.Assignment:
        """Solve problem."""
        solution = problem_dict["routing"].SolveWithParameters(problem_dict["search_parameters"])

        return solution

    def extract_solution(
        self, data_model: DataModel, problem_dict: dict[str, Any], solution: pywrapcp.Assignment
    ) -> dict[int, list[str]]:
        """Extract solution."""
        solution_dict = {}

        for day in range(data_model.num_days):
            stops = []
            index = problem_dict["routing"].Start(day)
            hotel_index = index

            while not problem_dict["routing"].IsEnd(index):
                stops.append(self.localization_data.places[problem_dict["manager"].IndexToNode(index)])
                index = solution.Value(problem_dict["routing"].NextVar(index))

            stops.append(self.localization_data.places[problem_dict["manager"].IndexToNode(hotel_index)])
            solution_dict[day] = stops

        return solution_dict

    def execute_optimizer_pipeline(self) -> dict[int, list[str]]:
        """Execute whole optimizer pipeline."""
        problem_dict = self.define_model(self.data_model)

        solution = self.solve_problem(problem_dict)

        return self.extract_solution(self.data_model, problem_dict, solution)
