"""Impact extension: crop-stress index from the twin's baseline and shocked rollouts (numpy only).

    impact.regions      approximate agricultural regions as lat/lon boxes
    impact.climatology  smoothed day-of-year climatology from the training years (matches the organisers' kit)
    impact.crop_stress  rain deficit %, hot-day count (t2m daily-mean anomaly) and a 0-100 index per region

The maths is written out in docs/IMPACT.md.
"""
from .climatology import clim_for_dates, daily_climatology, doy_index, forecast_dates, train_climatology
from .crop_stress import DEFAULT_WEIGHTS, crop_stress, region_weights
from .regions import REGIONS, region_boxes

__all__ = [
    "REGIONS", "region_boxes", "crop_stress", "region_weights", "DEFAULT_WEIGHTS",
    "daily_climatology", "train_climatology", "clim_for_dates", "forecast_dates", "doy_index",
]
