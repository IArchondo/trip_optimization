import googlemaps
import pandas as pd
import logging
import yaml
from pathlib import Path
from itertools import combinations
from g0_utils.utils import load_standard_date

LOGGER = logging.getLogger("DistanceMatrixCalculator")


class DistanceMatrixCalculator:
    def __init__(self, input_table):

        self.standard_date = load_standard_date()

        with open(Path("g0_utils/keys.yml")) as file:
            KEYS = yaml.full_load(file)

        self.gmaps = googlemaps.Client(key=KEYS["google_api_key"])
        LOGGER.debug("Google Maps Client initiated")
        # TODO input checker

        self.places = input_table["Place"]
        self.durations = input_table["Duration"]
        self.duration_dict = {x: [self.durations[i]] for i, x in enumerate(self.places)}

        LOGGER.info("DistanceMatrixCalculator initiated")

    def get_combination_list(self, places_list):
        """Generate a list of all combination tuples in a given places list
        
        Args:
            places_list (list): list of all places
        
        Returns:
            list: list with all combination tuples
        """
        LOGGER.info("Getting all place combinations")

        combination_list = list(combinations(places_list, 2))

        return combination_list

    def get_travel_time(self, place_tuple):
        """Calculate travel time in minutes for a give place pair in tuple format
        
        Args:
            place_tuple (tuple): tuple including two places
        
        Returns:
            float: travel time in minutes
        """

        directions_result = self.gmaps.directions(
            place_tuple[0],
            place_tuple[1],
            mode="transit",
            departure_time=self.standard_date,
        )

        travel_time = directions_result[0]["legs"][0]["duration"]["value"]

        travel_time = round(travel_time / 60, 3)

        return travel_time

    def execute_pipeline(self):
        self.combination_list = self.get_combination_list(self.places)

        self.distance_result_dict = {
            combination: self.get_travel_time(combination)
            for combination in self.combination_list
        }
