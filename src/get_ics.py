#!/usr/bin/env python3
"""
HRRR Initial Conditions Downloader
Downloads HRRR GRIB2 files for initial conditions.
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from utils import setup_logging, create_output_directory, download_file_with_retry

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for HRRR data downloader."""
    
    # Base URLs
    HRRR_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    
    # File types (forecast hour is appended when constructing the URL)
    HRRR_FILE_TYPES = {
        'pressure': 'wrfprs',
        'surface': 'wrfsfc'
    }
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    TIMEOUT = 300    # seconds


# -------------------------------
# HRRR Download Functions
# -------------------------------
def get_hrrr_urls(year: str, month: str, day: str, hour: str,
                  forecast_hour: int = 0, output_date: Optional[str] = None,
                  output_hour: Optional[str] = None) -> List[Tuple[str, str]]:
    """Generate HRRR URLs and filenames for an analysis or forecast hour."""
    urls = []
    date_str = f"{year}{month}{day}"
    output_date = output_date or date_str
    output_hour = output_hour or hour
    
    for file_type, file_prefix in Config.HRRR_FILE_TYPES.items():
        file_suffix = f"{file_prefix}f{forecast_hour:02d}.grib2"
        url = f"{Config.HRRR_BASE_URL}/hrrr.{date_str}/conus/hrrr.t{hour}z.{file_suffix}"
        filename = f"hrrr_{output_date}_{output_hour}_{file_type}.grib2"
        urls.append((url, filename))
    
    return urls

def download_hrrr_files(year: str, month: str, day: str, hour: str,
                        output_dir: Path, forecast_hour: int = 0,
                        output_date: Optional[str] = None,
                        output_hour: Optional[str] = None) -> List[bool]:
    """Download HRRR GRIB2 files."""
    logger = logging.getLogger(__name__)
    data_kind = "analysis" if forecast_hour == 0 else f"f{forecast_hour:02d} forecast"
    logger.info(
        f"Downloading HRRR {data_kind} from the {year}-{month}-{day} "
        f"{hour}:00 UTC cycle"
    )
    
    urls = get_hrrr_urls(
        year, month, day, hour, forecast_hour, output_date, output_hour
    )
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
def download_hrrr_data(datetime_str: str, base_dir: str = "{DATAROOT}/",
                       forecast_background: bool = False) -> dict:
    """Download HRRR analysis data or the previous cycle's f01 background."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    init_datetime, year, month, day, hour = utils.validate_datetime(datetime_str)
    date_str = f"{year}{month}{day}/{hour}"
    
    # Create output directory
    output_dir = create_output_directory(base_dir, date_str)
    logger.info(f"Output directory: {output_dir}")
    
    results = {'hrrr': []}
    
    # Download HRRR data
    try:
        source_dt = init_datetime - timedelta(hours=1) if forecast_background else init_datetime
        source_year = source_dt.strftime('%Y')
        source_month = source_dt.strftime('%m')
        source_day = source_dt.strftime('%d')
        source_hour = source_dt.strftime('%H')
        forecast_hour = 1 if forecast_background else 0

        hrrr_results = download_hrrr_files(
            source_year, source_month, source_day, source_hour, output_dir,
            forecast_hour=forecast_hour,
            output_date=f"{year}{month}{day}",
            output_hour=hour
        )
        results['hrrr'] = hrrr_results
    except Exception as e:
        logger.error(f"Error downloading HRRR data: {e}")
        results['hrrr'] = [False, False]

    # Analysis APCP is replaced downstream with the previous cycle's f01 APCP.
    # In forecast-background mode the main surface file is already that f01 file.
    if not forecast_background:
        try:
            prev_cycle_dt = init_datetime - timedelta(hours=1)
            _, prev_year, prev_month, prev_day, prev_hour = utils.validate_datetime(prev_cycle_dt.strftime('%Y-%m-%dT%H'))
            # Keep file naming with original cycle timestamp but store in current output_dir
            prev_cycle_date = f"{prev_year}{prev_month}{prev_day}"
            prev_url = f"{Config.HRRR_BASE_URL}/hrrr.{prev_cycle_date}/conus/hrrr.t{prev_hour}z.wrfsfcf01.grib2"
            prev_filename = f"hrrr_{prev_cycle_date}_{prev_hour}_surface_f01.grib2"
            prev_path = output_dir / prev_filename
            if not prev_path.exists():
                logger.info(f"Downloading previous hour surface 1h forecast (APCP source) into current directory from {prev_url}")
                results['prev_hour_surface_f01'] = download_file_with_retry(prev_url, str(prev_path))
            else:
                logger.info(f"Previous hour surface 1h forecast already present in current directory: {prev_path}")
                results['prev_hour_surface_f01'] = True
        except Exception as e:
            logger.error(f"Failed downloading previous hour surface f01 file: {e}")
            results['prev_hour_surface_f01'] = False
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download HRRR initial conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_ics.py 2024-01-15T12
  python get_ics.py 2024-01-15T12 --forecast-background
  python get_ics.py 2024-01-15T12 --base_dir /data/weather
  python get_ics.py 2024-01-15T12 --log_level DEBUG
        """
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument('--base_dir', default='./', help='Base directory for downloads (default: ./)')
    parser.add_argument('--forecast-background', '--forecast_background', action='store_true',
                       help='Use pressure and surface f01 forecasts from the previous HRRR cycle instead of analyses')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Download HRRR data
        results = download_hrrr_data(
            args.inittime, args.base_dir, args.forecast_background
        )
        
        # Summary
        supplemental_results = [
            result for name, result in results.items() if name != 'hrrr'
        ]
        total_successful = sum(results['hrrr']) + sum(supplemental_results)
        total_attempted = len(results['hrrr']) + len(supplemental_results)

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
