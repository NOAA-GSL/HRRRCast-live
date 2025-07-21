import logging
from dateutil import parser
from typing import Tuple, Union
from pathlib import Path

def validate_datetime(datetime_str: str) -> Tuple[object, str, str, str, str]:
    """Validate and format any datetime string that Python can parse.
    Returns (datetime_object, year, month, day, hour) as strings with proper padding.
    Raises ValueError if parsing fails.
    """
    try:
        dt = parser.parse(datetime_str)
        year = f"{dt.year:04d}"
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}"
        return dt, year, month, day, hour
    except (ValueError, TypeError, parser.ParserError) as e:
        logging.error(f"Invalid date/time: {e}")
        raise ValueError(f"Invalid date/time: {e}")

def make_directory(path: Union[str, Path]) -> None:
    """
    Create a directory (and any necessary parent directories).
    Accepts either a string or Path object. Does nothing if the directory already exists.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True) 