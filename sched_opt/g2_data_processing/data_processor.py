"""Process data."""

import logging

from sched_opt.g1_distance_calculation.distance_matrix_calculator import LocalizationData

LOGGER = logging.getLogger("DataProcessor")


class DataProcessor:
    """Hold data processor."""

    def __init__(self) -> None:
        """Initiate class."""
        self.LARGE_NUMBER = 1000000

        LOGGER.info("DataProcessor initiated")

    def add_inverse_relationships(
        self, combination_distance: dict[tuple[str, str], float]
    ) -> dict[tuple[str, str], float]:
        """Add inverse relationships to complete matrix."""
        combinations = list(combination_distance.keys())

        for comb in combinations:
            combination_distance[(comb[1], comb[0])] = combination_distance[comb]

        return combination_distance

    def add_own_combination(
        self, combination_distance: dict[tuple[str, str], float], places: list[str]
    ) -> dict[tuple[str, str], float]:
        """Add combinations with of places with themselves."""
        LOGGER.info("Balancing matrix")

        for place in list(places):
            combination_distance[(place, place)] = self.LARGE_NUMBER

        return combination_distance

    def add_schedule_dimension(
        self, combination_distance: dict[tuple[str, str], float], places: list[str]
    ) -> dict[tuple[str, str, int], float]:
        """Add schedule dimension to distance matrix."""
        LOGGER.info("Adding schedule dimension")

        schedule_length = len(places)
        comb_dist_sch = {}
        for combination in combination_distance:
            for i in range(schedule_length):
                comb_dist_sch[(combination[0], combination[1], i)] = combination_distance[
                    (combination[0], combination[1])
                ]

        return comb_dist_sch

    def execute_pipeline(self, distance_output: LocalizationData) -> LocalizationData:
        """Execute data processor pipeline."""
        combination_distance_stay = self.add_inverse_relationships(distance_output.combination_distance_stay_dict)

        combination_distance_stay = self.add_own_combination(combination_distance_stay, distance_output.places)

        comb_dist_sch = self.add_schedule_dimension(combination_distance_stay, distance_output.places)

        return LocalizationData(
            combination_distance_dict=distance_output.combination_distance_dict,
            combination_distance_stay_dict=combination_distance_stay,
            places=distance_output.places,
            duration_dict=distance_output.duration_dict,
            places_geocoding_dict=distance_output.places_geocoding_dict,
            comb_dist_sch=comb_dist_sch,
        )
