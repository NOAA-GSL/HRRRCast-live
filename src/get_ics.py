#!/usr/bin/env python3
"""
HRRR Initial Conditions Downloader
Downloads HRRR GRIB2 files for initial conditions.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for HRRR data downloader."""
    
    # Base URLs
    HRRR_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    
    # File types
    HRRR_FILE_TYPES = {
        'pressure': 'wrfprsf00.grib2',
        'surface': 'wrfsfcf00.grib2'
    }
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    TIMEOUT = 300    # seconds

# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# -------------------------------
# Utility Functions
# -------------------------------
def validate_datetime(year: str, month: str, day: str, hour: str) -> Tuple[str, str, str, str]:
    """Validate and format datetime components."""
    try:
        # Pad with zeros if necessary
        year = year.zfill(4)
        month = month.zfill(2)
        day = day.zfill(2)
        hour = hour.zfill(2)
        
        # Validate the date
        datetime(int(year), int(month), int(day), int(hour))
        
        return year, month, day, hour
    except ValueError as e:
        raise ValueError(f"Invalid date/time: {e}")

def create_output_directory(base_dir: str, date_str: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(base_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def download_file_with_retry(url: str, output_path: str, max_retries: int = Config.MAX_RETRIES) -> bool:
    """Download a file with retry logic and progress tracking."""
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, stream=True, timeout=Config.TIMEOUT)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress for large files
                        if total_size > 0 and downloaded % (total_size // 10) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded: {os.path.basename(output_path)}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {Config.RETRY_DELAY} seconds...")
                time.sleep(Config.RETRY_DELAY)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    return False

# -------------------------------
# HRRR Download Functions
# -------------------------------
def get_hrrr_urls(year: str, month: str, day: str, hour: str) -> List[Tuple[str, str]]:
    """Generate HRRR download URLs and filenames."""
    urls = []
    date_str = f"{year}{month}{day}"
    
    for file_type, file_suffix in Config.HRRR_FILE_TYPES.items():
        url = f"{Config.HRRR_BASE_URL}/hrrr.{date_str}/conus/hrrr.t{hour}z.{file_suffix}"
        filename = f"hrrr_{date_str}_{hour}_{file_type}.grib2"
        urls.append((url, filename))
    
    return urls

def download_hrrr_files(year: str, month: str, day: str, hour: str, output_dir: Path) -> List[bool]:
    """Download HRRR GRIB2 files."""
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading HRRR initial conditions for {year}-{month}-{day} {hour}:00 UTC")
    
    urls = get_hrrr_urls(year, month, day, hour)
    results = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_url = {
            executor.submit(download_file_with_retry, url, str(output_dir / filename)): (url, filename)
            for url, filename in urls
        }
        
        for future in as_completed(future_to_url):
            url, filename = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                if result:
                    logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                results.append(False)
    
    logger.info(f"HRRR downloads completed: {sum(results)}/{len(results)} successful")
    return results

# -------------------------------
# Main Functions
# -------------------------------
def download_hrrr_data(year: str, month: str, day: str, hour: str, base_dir: str = "./") -> dict:
    """Download HRRR initial condition data for specified date and time."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    year, month, day, hour = validate_datetime(year, month, day, hour)
    date_str = f"{year}{month}{day}_{hour}"
    
    # Create output directory
    output_dir = create_output_directory(base_dir, date_str)
    logger.info(f"Output directory: {output_dir}")
    
    results = {'hrrr': []}
    
    # Download HRRR data
    try:
        hrrr_results = download_hrrr_files(year, month, day, hour, output_dir)
        results['hrrr'] = hrrr_results
    except Exception as e:
        logger.error(f"Error downloading HRRR data: {e}")
        results['hrrr'] = [False]
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download HRRR initial conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_ics.py 2024 01 15 12
  python get_ics.py 2024 1 15 12 --base-dir /data/weather
  python get_ics.py 2024 01 15 12 --log-level DEBUG
        """
    )
    
    parser.add_argument('year', help='Year (e.g., 2024)')
    parser.add_argument('month', help='Month (1-12)')
    parser.add_argument('day', help='Day (1-31)')
    parser.add_argument('hour', help='Hour (0-23)')
    parser.add_argument('--base-dir', default='./', help='Base directory for downloads (default: ./)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Download HRRR data
        results = download_hrrr_data(
            args.year, args.month, args.day, args.hour, args.base_dir
        )
        
        # Summary
        total_successful = sum(results['hrrr'])
        total_attempted = len(results['hrrr'])
        
        logger.info(f"Download summary: {total_successful}/{total_attempted} files successful")
        
        if total_successful == 0:
            logger.error("No files were downloaded successfully")
            sys.exit(1)
        elif total_successful < total_attempted:
            logger.warning("Some downloads failed")
            sys.exit(2)
        else:
            logger.info("All downloads completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
