"""Calculate distances between places."""

import logging
import pickle
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import googlemaps  # type: ignore
import pandas as pd
import yaml

from sched_opt.g0_utils.utils import load_standard_date

LOGGER = logging.getLogger("DistanceMatrixCalculator")


@dataclass
class DistanceOutput:
    """Hold all outputs from distance calculator."""

    combination_distance_dict: dict[tuple[str, str], float]
    combination_distance_stay_dict: dict[tuple[str, str], float]
    places: list[str]
    duration_dict: dict[str, list[float]]
    places_geocoding_dict: dict[str, float]


class DistanceMatrixCalculator:
    """Calculate distances between destinations."""

    def __init__(self, input_data: pd.DataFrame) -> None:
        """Initiate class."""
        input_data = self.__check_input_data(input_data)

        self.standard_date = load_standard_date()

        with open(Path("sched_opt/g0_utils/keys.yml")) as file:
            keys = yaml.full_load(file)

        self.gmaps = googlemaps.Client(key=keys["google_api_key"])
        LOGGER.debug("Google Maps Client initiated")

        self.places = [str(place) for place in input_data["Place"]]
        self.durations = [float(duration) for duration in input_data["Duration"]]
        self.duration_dict = {x: [self.durations[i]] for i, x in enumerate(self.places)}

        LOGGER.info("DistanceMatrixCalculator initiated")

    def __check_input_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Check if input data fulfills required criteria."""
        assert (
            pd.Series(["Place", "Duration"]).isin(input_data.columns).all()
        ), "Place or Duration are not present in input data"
        return input_data

    def get_combination_list(self, places_list: list[str]) -> list[tuple[str, str]]:
        """Generate a list of all combination tuples in a given places list."""
        LOGGER.info("Getting all place combinations")

        combination_list = list(combinations(places_list, 2))

        return combination_list

    def get_geocoding(self, place: str) -> Any:
        """Get geocoding of a given input."""
        LOGGER.debug(f"Gathering geocoding for {place}")
        time.sleep(5)
        geocoding_result = self.gmaps.geocode(place)

        return geocoding_result

    def get_travel_time(self, place_tuple: tuple[str, str]) -> float:
        """Calculate travel time in minutes for a give place pair in tuple format."""
        LOGGER.debug(f"Fetching travel time from {place_tuple[0]} to {place_tuple[1]}")

        time.sleep(5)
        directions_result = self.gmaps.directions(
            place_tuple[0],
            place_tuple[1],
            mode="driving",
            departure_time=self.standard_date,
        )

        travel_time = (
            directions_result[0]["legs"][0]["duration"]["value"] if len(directions_result) > 0 else 100000000000
        )

        return float(round(travel_time / 60, 3))

    def execute_pipeline(self, save_output: bool = False) -> DistanceOutput:
        """Execute complete pipeline."""
        LOGGER.info("Executing pipeline")

        LOGGER.info(f"Gathering geocoding for {len(self.places)} combinations.", f"ETA: {len(self.places)*5} seconds")
        self.places_geocoding = {place: self.get_geocoding(place) for place in self.places}

        self.combination_list = self.get_combination_list(self.places)

        LOGGER.info(
            f"Gathering travel times for {len(self.combination_list)} combinations.",
            f"ETA: {len(self.combination_list)*5} seconds",
        )
        self.combination_distance_dict = {
            combination: self.get_travel_time(combination) for combination in self.combination_list
        }

        self.combination_distance_stay_dict = {
            combination: (self.combination_distance_dict[combination] + self.duration_dict[combination[0]][0])
            for combination in self.combination_list
        }

        output = DistanceOutput(
            combination_distance_dict=self.combination_distance_dict,
            combination_distance_stay_dict=self.combination_distance_stay_dict,
            places=self.places,
            duration_dict=self.duration_dict,
            places_geocoding_dict=self.places_geocoding,
        )

        if save_output:
            LOGGER.info("Saving output under latest_distances.pickle")
            with open(Path("00_saved_data/saved_distances") / "latest_distances.pickle", "wb") as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return output
