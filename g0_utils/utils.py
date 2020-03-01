import yaml
from datetime import datetime
from pathlib import Path


def load_standard_date():
    """Loads standard date for google maps queries
        from config file
    
    Returns:
        datetime.datetime: standard datetime
    """

    with open(Path("config.yml")) as file:
        CONFIG = yaml.full_load(file)

    standard_date_config = CONFIG["standard_time"]

    standard_date = datetime(
        year=standard_date_config["year"],
        month=standard_date_config["month"],
        day=standard_date_config["day"],
        hour=standard_date_config["hour"],
        minute=standard_date_config["minute"],
    )

    return standard_date

