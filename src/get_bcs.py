#!/usr/bin/env python3
"""
GFS Lateral Boundary Conditions Downloader
Downloads GFS GRIB2 files for lateral boundary conditions.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from dateutil import parser
from pathlib import Path
from typing import List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for GFS data downloader."""
    
    # Base URLs
    GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    
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
def validate_datetime(datetime_str: str) -> Tuple[str, str, str, str]:
    """Validate and format any datetime string that Python can parse."""
    try:
        # Parse the datetime string using dateutil parser (very flexible)
        dt = parser.parse(datetime_str)
        
        # Format components with proper padding
        year = f"{dt.year:04d}"
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}"
        
        return dt, year, month, day, hour
        
    except (ValueError, TypeError, parser.ParserError) as e:
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
# GFS Download Functions
# -------------------------------
def get_gfs_urls(year: str, month: str, day: str, hour: str, lead_hours: int) -> List[Tuple[str, str]]:
    """Generate GFS download URLs and filenames for boundary conditions."""
    urls = []
    hour_int = int(hour)
    cycle_hours = [0, 6, 12, 18]
    
    # Find the appropriate GFS cycle (must be synoptic hour for initialization)
    if hour_int in cycle_hours:
        init_cycle = hour_int
        init_date_str = f"{year}{month}{day}"
    else:
        # Use the most recent synoptic hour
        previous_cycle = max([c for c in cycle_hours if c < hour_int], default=18)
        if previous_cycle >= hour_int:
            # Need to go to previous day
            dt = datetime(int(year), int(month), int(day)) - timedelta(days=1)
            init_date_str = dt.strftime("%Y%m%d")
            init_cycle = 18
        else:
            init_date_str = f"{year}{month}{day}"
            init_cycle = previous_cycle
    
    cycle_str = f"{init_cycle:02d}"
    
    # Calculate forecast hours needed
    if hour_int in cycle_hours:
        start_forecast_hour = 0
    else:
        # Calculate offset from the initialization cycle
        if init_date_str != f"{year}{month}{day}":
            # Previous day's 18Z cycle
            start_forecast_hour = (24 - 18) + hour_int
        else:
            start_forecast_hour = hour_int - init_cycle
    
    # Generate URLs for all forecast hours from start to start + lead_hours
    for fh in range(start_forecast_hour, start_forecast_hour + lead_hours + 1):
        forecast_str = f"{fh:03d}"
        url = f"{Config.GFS_BASE_URL}/gfs.{init_date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{forecast_str}"
        
        # Calculate valid time for this forecast hour
        init_dt = datetime(int(init_date_str[:4]), int(init_date_str[4:6]), int(init_date_str[6:8]), init_cycle)
        valid_dt = init_dt + timedelta(hours=fh)
        valid_str = valid_dt.strftime("%Y%m%d_%H")
        
        filename = f"gfs_{valid_str}.grib2"
        urls.append((url, filename))
    
    return urls

def download_gfs_files(year: str, month: str, day: str, hour: str, lead_hours: int, output_dir: Path) -> List[bool]:
    """Download GFS GRIB2 files for boundary conditions."""
    logger = logging.getLogger(__name__)
    
    if lead_hours == 0:
        logger.info(f"Downloading GFS data for {year}-{month}-{day} {hour}:00 UTC")
    else:
        logger.info(f"Downloading GFS boundary conditions: {year}-{month}-{day} {hour}:00 UTC + {lead_hours} hours")
    
    urls = get_gfs_urls(year, month, day, hour, lead_hours)
    results = []
    
    # Use ThreadPoolExecutor for multiple files
    with ThreadPoolExecutor(max_workers=4) as executor:
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
    
    logger.info(f"GFS downloads completed: {sum(results)}/{len(results)} successful")
    return results

# -------------------------------
# Main Functions
# -------------------------------
def download_gfs_data(datetime_str: str, lead_hours: int, base_dir: str = "./") -> dict:
    """Download GFS boundary condition data for specified date and time."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    init_datetime, year, month, day, hour = validate_datetime(datetime_str)
    date_str = f"{year}{month}{day}_{hour}"
    
    # Create output directory
    output_dir = create_output_directory(base_dir, date_str)
    logger.info(f"Output directory: {output_dir}")
    
    results = {'gfs': []}
    
    # Download GFS data
    try:
        gfs_results = download_gfs_files(year, month, day, hour, lead_hours, output_dir)
        results['gfs'] = gfs_results
    except Exception as e:
        logger.error(f"Error downloading GFS data: {e}")
        results['gfs'] = [False]
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download GFS lateral boundary conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_lbcs.py 2024-01-15T12 0    # Single file
  python get_lbcs.py 2024-01-15T12 24   # 24-hour boundary conditions
  python get_lbcs.py 2024-01-15T12 48 --base-dir /data/weather
  python get_lbcs.py 2024-01-15T12 36 --log-level DEBUG
        """
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument('lead_hours', type=int, help='Lead time in hours for boundary conditions')
    parser.add_argument('--base-dir', default='./', help='Base directory for downloads (default: ./)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate lead_hours
    if args.lead_hours < 0:
        logger.error("Lead hours must be >= 0")
        sys.exit(1)
    
    try:
        # Download GFS data
        results = download_gfs_data(
            args.inittime, args.lead_hours, args.base_dir
        )
        
        # Summary
        total_successful = sum(results['gfs'])
        total_attempted = len(results['gfs'])
        
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
