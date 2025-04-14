"""Hold all elements in the model."""

from dataclasses import dataclass


@dataclass
class Coordinates:
    """Determine a place coordinates."""

    lat: float
    lng: float


@dataclass
class Activity:
    """Hold elements for a destination."""

    name: str
    activity_duration: float
    coords: Coordinates


@dataclass
class Trip:
    """Hold elements for a trip."""

    origin: str
    destination: str
    duration: float
    duration_incl_activity_length: float


@dataclass
class ModelInputs:
    """Gather all model inputs."""

    activities: dict[str, Activity]
    trips: dict[tuple[str, str], Trip]
    hotel: str
    no_of_days: int
    total_hours_in_day: int
