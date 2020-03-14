from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging

LOGGER = logging.getLogger("ScheduleOptimizer")


class ScheduleOptimizer:
    def __init__(self, proc_data_dict, problem_definition_dict):
        self.proc_data_dict = proc_data_dict

        self.num_days = problem_definition_dict["num_days"]
        self.hotel_index = problem_definition_dict["hotel_index"]

        ## create data model
        self.data_model = self.__create_data_model(
            self.proc_data_dict, self.num_days, self.hotel_index
        )

    def __create_data_model(self, proc_data_input, num_days, hotel_index):
        data_google = {}

        distance_total = []
        for place in proc_data_input["places"]:
            distance = []
            for destination in proc_data_input["places"]:
                if place == destination:
                    distance.append(0)
                else:
                    distance.append(
                        proc_data_input["combination_distance_stay_dict"][
                            (place, destination)
                        ]
                    )

            distance_total.append(distance)

        data_google["distance_matrix"] = distance_total

        data_google["num_days"] = num_days

        data_google["hotel"] = hotel_index

        return data_google

    def define_model(self, data_model):
        manager = pywrapcp.RoutingIndexManager(
            len(data_model["distance_matrix"]),
            data_model["num_days"],
            data_model["hotel"],
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data_model["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        output_dict = {
            "manager": manager,
            "routing": routing,
            "search_parameters": search_parameters,
        }

        return output_dict

    def solve_problem(self, problem_dict):
        solution = problem_dict["routing"].SolveWithParameters(
            problem_dict["search_parameters"]
        )

        return solution

    def extract_solution(self, data_model, problem_dict, solution):
        """Extract solution
        
        Args:
            data_model (dict): input data model
            problem_dict (dict): problem dict
            solution (solution): solution to problem
        
        Returns:
            dict: dict with all routes per day
        """

        solution_dict = {}

        for day in range(data_model["num_days"]):
            stops = []
            index = problem_dict["routing"].Start(day)
            hotel_index = index

            while not problem_dict["routing"].IsEnd(index):

                stops.append(
                    self.proc_data_dict["places"][
                        problem_dict["manager"].IndexToNode(index)
                    ]
                )
                index = solution.Value(problem_dict["routing"].NextVar(index))

            stops.append(
                self.proc_data_dict["places"][
                    problem_dict["manager"].IndexToNode(hotel_index)
                ]
            )
            solution_dict[day] = stops

        return solution_dict

    def execute_optimizer_pipeline(self):
        """Execute whole optimizer pipeline
        
        Returns:
            dict: solution dictionary
        """

        problem_dict = self.define_model(self.data_model)

        solution = self.solve_problem(problem_dict)

        solution_dict = self.extract_solution(self.data_model, problem_dict, solution)

        return solution_dict

