import googlemaps
import pandas as pd
import logging
import yaml
import pickle
import time
from pathlib import Path
from itertools import combinations
from g0_utils.utils import load_standard_date

LOGGER = logging.getLogger("DistanceMatrixCalculator")


class DistanceMatrixCalculator:
    def __init__(self, input_data):

        input_data = self.__check_input_data(input_data)

        self.standard_date = load_standard_date()

        with open(Path("g0_utils/keys.yml")) as file:
            KEYS = yaml.full_load(file)

        self.gmaps = googlemaps.Client(key=KEYS["google_api_key"])
        LOGGER.debug("Google Maps Client initiated")
        # TODO input checker

        self.places = input_data["Place"]
        self.durations = input_data["Duration"]
        self.duration_dict = {x: [self.durations[i]] for i, x in enumerate(self.places)}

        LOGGER.info("DistanceMatrixCalculator initiated")

    def __check_input_data(self, input_data):
        """Check if input data fulfills required criteria
        
        Args:
            input_data (pd.DataFrame): input data with places and durations
        
        Returns:
            pd.DataFrame: input data with places and durations
        """
        assert (
            pd.Series(["Place", "Duration"]).isin(input_data.columns).all()
        ), "Place or Duration are not present in input data"
        return input_data

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

    def get_geocoding(self, place):
        """Get geocoding of a given input
        
        Args:
            place (str): Place as string
        
        Returns:
            xxx: geocoding result
        """
        LOGGER.debug(f"Gathering geocoding for {place}")
        time.sleep(5)
        geocoding_result = self.gmaps.geocode(place)

        return geocoding_result

    def get_travel_time(self, place_tuple):
        """Calculate travel time in minutes for a give place pair in tuple format
        
        Args:
            place_tuple (tuple): tuple including two places
        
        Returns:
            float: travel time in minutes
        """
        LOGGER.debug(f"Fetching travel time from {place_tuple[0]} to {place_tuple[1]}")

        time.sleep(5)
        directions_result = self.gmaps.directions(
            place_tuple[0],
            place_tuple[1],
            mode="transit",
            departure_time=self.standard_date,
        )

        travel_time = directions_result[0]["legs"][0]["duration"]["value"]

        travel_time = round(travel_time / 60, 3)

        return travel_time

    def execute_pipeline(self, save_output=False):
        """Execute complete pipeline
        
        Returns:
            dict: Dictionary with:
                - dict with combination and their travel time
                - dict with combination and travel time + stay time
        """

        LOGGER.info("Executing pipeline")

        self.places_geocoding = {
            place: self.get_geocoding(place) for place in self.places
        }

        self.combination_list = self.get_combination_list(self.places)

        LOGGER.info("Gathering travel times")
        self.combination_distance_dict = {
            combination: self.get_travel_time(combination)
            for combination in self.combination_list
        }

        self.combination_distance_stay_dict = {
            combination: (
                self.combination_distance_dict[combination]
                + self.duration_dict[combination[0]][0]
            )
            for combination in self.combination_list
        }

        output_dict = {
            "combination_distance_dict": self.combination_distance_dict,
            "combination_distance_stay_dict": self.combination_distance_stay_dict,
            "places": self.places,
            "duration_dict": self.duration_dict,
            "placed_geocoding_dict": self.places_geocoding,
        }

        if save_output:
            LOGGER.info("Saving output under latest_distances.pickle")
            with open(
                Path("00_saved_data/saved_distances") / "latest_distances.pickle", "wb"
            ) as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return output_dict

