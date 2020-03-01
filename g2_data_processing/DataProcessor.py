import pandas as pd
import logging

LOGGER = logging.getLogger("DataProcessor")


class DataProcessor:
    def __init__(self, distances_dict):

        self.LARGE_NUMBER = 1000000

        self.distances_dict = distances_dict

        self.combination_distance = distances_dict["combination_distance_dict"]
        self.combination_distance_stay = distances_dict[
            "combination_distance_stay_dict"
        ]
        self.places = distances_dict["places"]

        LOGGER.info("DataProcessor initiated")

    def add_own_combination(self, combination_distance, places):
        """Add combinations with a place with its own to get an equilibrated matrix
        
        Args:
            combination_distance (dict): combination-travel time dict
            places (list): list with all the places
        
        Returns:
            dict: dict with added combinations
        """

        for place in places:
            combination_distance[(place, place)] = self.LARGE_NUMBER

        return combination_distance

    def add_schedule_dimension(self, combination_distance, places):
        """Add schedule dimension to distance matrix
        
        Args:
            combination_distance (dict): combination-travel time dict
            places (list): list with all the places
        
        Returns:
            dict: dict with added dimension
        """

        schedule_length = len(places)
        comb_dist_sch = {}
        for combination in combination_distance.keys():
            for i in range(schedule_length):
                comb_dist_sch[
                    (combination[0], combination[1], i)
                ] = combination_distance[(combination[0], combination[1])]

        return comb_dist_sch

    def execute_pipeline(self):
        """Execute data processor pipeline
        
        Returns:
            dict: modified input dict
        """

        self.combination_distance_stay = self.add_schedule_dimension(
            self.combination_distance_stay, self.places
        )

        self.comb_dist_sch = self.add_schedule_dimension(
            self.combination_distance_stay, self.places
        )

        self.distances_dict["comb_dist_sch"] = self.comb_dist_sch

        return self.distances_dict

