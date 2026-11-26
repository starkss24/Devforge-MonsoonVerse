"""APPROXIMATE agricultural regions of India as lat/lon boxes on the twin's 1 degree grid.

These are NOT administrative or agro-climatic-zone boundaries. Each box is a rough rectangle drawn on cell
centres (x.5 degrees) around a well-known farming belt, chosen so that the boxes do not overlap and avoid the
Himalaya. A real product would use district polygons; on a 1 degree grid a box is about as precise as the
data allows. Boxes are inclusive on cell centres, like common.box_mask. When the land-sea mask is passed to
impact.crop_stress, only cells with land_sea_mask > 0.5 are used, so the coastal boxes lose their sea cells.

Cell counts on the organisers' grid (all cells / land cells with mask > 0.5) are in docs/IMPACT.md.
"""
from __future__ import annotations

# name -> box (lat_min, lat_max, lon_min, lon_max) on cell centres, short label for maps, and a note.
REGIONS: dict[str, dict] = {
    "NW plains (Punjab–Haryana)": {
        "box": (28.5, 30.5, 73.5, 76.5), "short": "NW plains",
        "note": "Irrigated wheat-rice belt; stops at 30.5°N to keep the Himalayan foothills out.",
    },
    "Gangetic plain (UP–Bihar)": {
        "box": (24.5, 27.5, 77.5, 85.5), "short": "Gangetic plain",
        "note": "Kharif rice and rabi wheat; the north-east corner touches the Nepal foothills.",
    },
    "Central India (Vidarbha–Marathwada)": {
        "box": (19.5, 22.5, 74.5, 80.5), "short": "Central India",
        "note": "Rain-fed cotton, soybean and pulses; starts at 19.5°N so it does not overlap the west-coast box.",
    },
    "Deccan interior (Telangana–Rayalaseema)": {
        "box": (13.5, 17.5, 77.5, 80.5), "short": "Deccan",
        "note": "Semi-arid rain-fed groundnut, millets and pulses.",
    },
    "East coast (Odisha–Andhra)": {
        "box": (16.5, 21.5, 81.5, 86.5), "short": "East coast",
        "note": "Rice deltas (Godavari, Mahanadi); land cells only. Its southern part lies inside the warm_ocean shock box.",
    },
    "West coast (Konkan–Malabar)": {
        "box": (8.5, 18.5, 72.5, 76.5), "short": "West coast",
        "note": "Rice, coconut and spices; land cells only, includes some Western Ghats cells. "
                "Overlaps the monsoon_stall shock box.",
    },
}


def region_boxes(regions: dict | None = None) -> dict[str, tuple]:
    """name -> (lat_min, lat_max, lon_min, lon_max), the form impact.crop_stress expects."""
    regions = REGIONS if regions is None else regions
    return {name: tuple(float(x) for x in (r["box"] if isinstance(r, dict) else r)) for name, r in regions.items()}
