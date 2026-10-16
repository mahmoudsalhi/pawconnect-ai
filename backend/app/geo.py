from math import asin, cos, radians, sin, sqrt

# Search areas offered by AI Matching.
AREAS = {
    "ennasr": {"label": "Ennasr, Ariana", "lat": 36.8625, "lng": 10.1647},
    "marsa": {"label": "La Marsa, Tunis", "lat": 36.8782, "lng": 10.3247},
    "centre": {"label": "Tunis centre", "lat": 36.8008, "lng": 10.18},
    "sousse": {"label": "Sousse", "lat": 35.8256, "lng": 10.6084},
    "sfax": {"label": "Sfax", "lat": 34.7406, "lng": 10.7603},
}

GOVERNORATES = ["Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Sousse", "Sfax"]


def km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres."""
    x = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lng2 - lng1) / 2) ** 2
    return 12742 * asin(sqrt(x))
