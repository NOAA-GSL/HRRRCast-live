import logging
from dateutil import parser
from typing import Tuple, Union
from pathlib import Path
import requests
import time

# Retry / download defaults (can be overridden in callers)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_TIMEOUT = 300    # seconds

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

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure root logging (idempotent across modules)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_output_directory(base_dir: str, date_str: str) -> Path:
    """Create and return an output directory base_dir/date_str."""
    out = Path(base_dir) / date_str
    make_directory(out)
    return out

def download_file_with_retry(url: str, output_path: Union[str, Path], max_retries: int = DEFAULT_MAX_RETRIES,
                              retry_delay: int = DEFAULT_RETRY_DELAY, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Download a file with retries and basic progress logging.

    Args:
        url: Source URL
        output_path: Destination path
        max_retries: Attempts before failing
        retry_delay: Delay between attempts (s)
        timeout: Per-request timeout (s)
    Returns:
        True on success, False on failure.
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt+1}/{max_retries})")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if total:
                            downloaded += len(chunk)
                            if downloaded % max((total // 10), 1) == 0:
                                logger.info(f"Progress: {(downloaded/total)*100:.1f}%")
            logger.info(f"Downloaded {output_path.name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed ({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    return False
