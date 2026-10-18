"""Everything the M7 dataset generator assumes, in one place.

Three kinds of numbers live here:
  * REAL       - geography comes from OpenStreetMap (see osm.py); nothing to set here.
  * PUBLISHED  - calibration targets taken from studies of real lost pets (TARGETS).
                 The simulator is tuned until it reproduces them (calibrate.py).
  * ASSUMED    - behaviour parameters (profiles, weights, base rates). These shape *how*
                 pets move between the published end points. Change them here, rebuild,
                 and the validation report shows the effect.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # downloaded OSM extracts and rasterised grids (cache)
OUTPUT = ROOT / "output"      # the dataset

CELL_M = 20.0                 # raster resolution in metres
STEP_MIN = 10                 # simulation time step in minutes
MAX_DAYS = 60                 # cases still missing after this are "not_found" (censored)

# ---------------------------------------------------------------------------------------
# Regions: bounding boxes (south, west, north, east). Each box reaches well past the city
# so long-distance dogs don't hit the edge. Governorates match backend/app/geo.py.
# ---------------------------------------------------------------------------------------
REGIONS = {
    "grand_tunis": {
        "bbox": (36.74, 10.05, 36.94, 10.35),
        "governorates": ["Tunis", "Ariana", "Ben Arous", "Manouba"],
        "cases": 9000,
        "temp_offset": 0.0,
    },
    "sousse": {
        "bbox": (35.75, 10.52, 35.90, 10.66),
        "governorates": ["Sousse"],
        "cases": 5000,
        "temp_offset": 0.5,
    },
    "sfax": {
        "bbox": (34.67, 10.62, 34.83, 10.82),
        "governorates": ["Sfax"],
        "cases": 5000,
        "temp_offset": 1.0,
    },
    "nabeul_hammamet": {
        "bbox": (36.36, 10.54, 36.50, 10.78),
        "governorates": ["Nabeul"],
        "cases": 5000,
        "temp_offset": 0.0,
    },
}

# Mean daily temperature by month for coastal Tunisia (°C, Jan..Dec). Climate normals,
# rounded; used for the midday-heat effect (animals rest in shade when it is hot).
MONTH_TEMP_C = [11.5, 12.0, 14.0, 16.5, 20.5, 24.5, 27.5, 28.0, 25.5, 21.0, 16.5, 12.5]

# ---------------------------------------------------------------------------------------
# PUBLISHED calibration targets.
# ---------------------------------------------------------------------------------------
TARGETS = {
    # Huang et al. 2018, Animals 8(1):5, 1,210 missing cats (online survey, mostly AU/US/UK).
    #   indoor-only cats: 75% found within 137 m of the point of escape
    #   cats with outdoor access: 75% found within 1,609 m
    #   34% recovered alive by the owner within 7 days; 61% found within a year
    "cat_indoor": {"dist_quantile": 0.75, "dist_m": 137.0},
    "cat_outdoor": {"dist_quantile": 0.75, "dist_m": 1609.0},
    "cat": {"found_alive_7d": 0.34},
    # Lord et al. 2007, JAVMA 230(2):211-216, 187 lost dogs, Montgomery County, Ohio.
    #   71% recovered, median 2 days; of those with a known distance, 71% < 1 mile,
    #   14% 1-5 miles, 7% > 5 miles  ->  normalised: 77% within 1,609 m, 7.6% beyond 8,047 m
    "dog": {"dist_quantile": 0.77, "dist_m": 1609.0, "recovered": 0.71},
}

# Checked in the validation report but not tuned for (they follow from the model).
VALIDATION_CHECKS = {
    "cat_all_within_500m": 0.75,   # Huang 2018: 75% of all cats found within 500 m
    "dog_beyond_8km": 0.076,        # Lord 2007, normalised (also used to tune dog relocations)
    "dog_median_days": 2.0,         # Lord 2007 (also used to tune dog search decay)
    "cat_found_ever": 0.61,         # Huang 2018 (one year; we censor at 60 days, so expect a bit less)
}

# ---------------------------------------------------------------------------------------
# ASSUMED population mix for Tunisian platform users. Weights, not probabilities.
# ---------------------------------------------------------------------------------------
SPECIES = {"cat": 0.5, "dog": 0.5}

CAT = {
    "lifestyle": {"indoor_only": 0.55, "outdoor_access": 0.45},  # many apartment cats in cities
    # Temperament classes from Missing Animal Response practice (Albrecht).
    "temperament": {"curious": 0.3, "cautious": 0.4, "xenophobic": 0.3},
    "age_group": {"kitten": 0.15, "adult": 0.7, "senior": 0.15},
    "sex": {"male": 0.5, "female": 0.5},
    "neutered": 0.35,
    "collar_tag": 0.10,
    "microchipped": 0.03,
    "circumstance": {
        "indoor_only": {"door_window_escape": 0.50, "balcony_fall": 0.15, "moved_house": 0.12,
                        "transport_escape": 0.08, "fireworks_noise": 0.15},
        "outdoor_access": {"did_not_return": 0.60, "moved_house": 0.15, "fireworks_noise": 0.15,
                           "transport_escape": 0.10},
    },
}

DOG = {
    "size": {"small": 0.3, "medium": 0.4, "large": 0.3},
    # "local_mixed" = the Tunisian street-dog type (kelb arbi) and mixes.
    "breed_group": {"local_mixed": 0.35, "companion": 0.2, "terrier": 0.1, "herding": 0.1,
                    "guard": 0.15, "hound_sighthound": 0.1},
    "temperament": {"gregarious": 0.45, "aloof": 0.35, "xenophobic": 0.2},
    "age_group": {"puppy": 0.15, "adult": 0.7, "senior": 0.15},
    "sex": {"male": 0.55, "female": 0.45},
    "neutered": 0.2,
    "collar_tag": 0.35,
    "microchipped": 0.10,
    "circumstance": {"escaped_home_yard": 0.45, "slipped_leash_walk": 0.20, "fireworks_noise": 0.15,
                     "moved_house": 0.05, "transport_escape": 0.05, "car_accident": 0.05,
                     "followed_dogs": 0.05},
}

# Chance the pet bolts in panic at the start, by circumstance.
FLEE_START = {
    "door_window_escape": 0.15, "balcony_fall": 0.05, "moved_house": 0.3, "transport_escape": 0.7,
    "fireworks_noise": 0.9, "did_not_return": 0.0, "escaped_home_yard": 0.25,
    "slipped_leash_walk": 0.6, "car_accident": 0.9, "followed_dogs": 0.2,
}

# How precisely the owner knows where the pet was last seen (std dev in metres).
LAST_SEEN_SIGMA_M = {
    "door_window_escape": 5, "balcony_fall": 5, "moved_house": 10, "transport_escape": 30,
    "fireworks_noise": 60, "did_not_return": 15, "escaped_home_yard": 10,
    "slipped_leash_walk": 40, "car_accident": 50, "followed_dogs": 40,
}

# Hour-of-loss weights (24 values) by circumstance family.
HOURS = {
    "evening": [1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 6, 4, 2],
    "walk": [0, 0, 0, 0, 0, 1, 4, 7, 7, 4, 2, 2, 2, 2, 2, 3, 5, 7, 8, 7, 5, 3, 1, 0],
    "fireworks": [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10, 7],
    "day": [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 2, 2, 1],
}
HOUR_FAMILY = {
    "door_window_escape": "evening", "balcony_fall": "evening", "moved_house": "day",
    "transport_escape": "day", "fireworks_noise": "fireworks", "did_not_return": "evening",
    "escaped_home_yard": "day", "slipped_leash_walk": "walk", "car_accident": "day",
    "followed_dogs": "day",
}
# Month weights (Jan..Dec). Summer: doors and windows open, wedding fireworks season.
MONTHS = {
    "default": [7, 7, 8, 8, 9, 10, 11, 11, 9, 8, 7, 7],
    "fireworks_noise": [1, 1, 1, 2, 3, 8, 12, 12, 8, 3, 1, 2],
}

# ---------------------------------------------------------------------------------------
# ASSUMED movement and detection behaviour. Groups: cat_indoor, cat_outdoor, dog.
# Speeds are *net* speeds while moving (m/s); the calibrated speed multiplier scales them.
# ---------------------------------------------------------------------------------------
SPEED = {  # REST, EXPLORE, FORAGE, FLEE, HOMING
    "cat_indoor": [0.0, 0.06, 0.08, 0.5, 0.15],
    "cat_outdoor": [0.0, 0.10, 0.12, 0.6, 0.25],
    "dog": [0.0, 0.35, 0.30, 1.5, 0.6],
}
# Direction persistence (von Mises kappa) by state.
KAPPA = [0.0, 1.0, 0.5, 4.0, 2.0]

# Habitat preference weights used to score candidate steps.
WEIGHTS = {
    #            cover food people corridor green building farm major_cross exposed
    "cat_indoor": [2.0, 0.5, 0.0, 0.1, 0.5, -0.4, 0.0, -2.5, -0.8],
    "cat_outdoor": [1.2, 0.7, 0.0, 0.3, 0.5, -0.4, 0.2, -2.0, -0.5],
    "dog": [0.3, 1.0, 0.0, 1.2, 0.6, -2.0, 0.3, -1.5, 0.0],
}
PEOPLE_WEIGHT = {"curious": 0.3, "cautious": -0.8, "xenophobic": -2.0,
                 "gregarious": 1.0, "aloof": -0.2}
APPROACHABILITY = {"curious": 0.7, "cautious": 0.35, "xenophobic": 0.08,
                   "gregarious": 1.0, "aloof": 0.45}
# Relative distance modifiers (the calibrated multiplier sets the overall level).
SPEED_MOD = {
    "cat_age": {"kitten": 0.6, "adult": 1.0, "senior": 0.6},
    "cat_intact_male": 1.6,
    "dog_size": {"small": 0.7, "medium": 1.0, "large": 1.3},
    "dog_breed": {"local_mixed": 1.1, "companion": 0.7, "terrier": 1.0, "herding": 1.2,
                  "guard": 1.0, "hound_sighthound": 1.5},
    "dog_age": {"puppy": 0.7, "adult": 1.0, "senior": 0.6},
    "temperament": {"curious": 1.0, "cautious": 1.0, "xenophobic": 1.2,
                    "gregarious": 0.9, "aloof": 1.1},
}
HOMING_ABILITY = {"cat_indoor": 0.15, "cat_outdoor": 0.7, "dog": 0.5}
HOMING_RANGE_M = {"cat_indoor": 150, "cat_outdoor": 600, "dog": 3000}
OWNER_SEARCH_R_M = {"cat_indoor": 300, "cat_outdoor": 500, "dog": 1500}

# Per-step (10 min) base hazards before the calibrated detection multiplier.
HAZARD = {
    "public": {"cat_indoor": 0.0015, "cat_outdoor": 0.0020, "dog": 0.0040},
    "owner": {"cat_indoor": 0.0060, "cat_outdoor": 0.0040, "dog": 0.0050},
    "taken_in": {"cat_indoor": 0.0002, "cat_outdoor": 0.0003, "dog": 0.0012},
    "picked_up": {"cat_indoor": 0.00005, "cat_outdoor": 0.00005, "dog": 0.0004},
    "self_return": {"cat_indoor": 0.0005, "cat_outdoor": 0.0030, "dog": 0.0020},
    "sighting": {"cat_indoor": 0.004, "cat_outdoor": 0.006, "dog": 0.015},
    "false_sightings_per_day": 0.25,
}
ROAD_DEATH_P = {"cat_indoor": 0.015, "cat_outdoor": 0.010, "dog": 0.008}
BACKGROUND_DEATH_P = {"cat_indoor": 0.00003, "cat_outdoor": 0.00002, "dog": 0.00001}
BODY_FOUND_P = 0.5
TAKEN_IN_REPORTED_P = 0.6          # stranger later posts / is traced; else the pet stays lost
# Strangers drive the pet home; dogs get carried further (finders in cars, rural pick-ups).
TAKEN_IN_RANGE_M = {"min": 500, "cat": (500, 6000), "dog": (500, 15000)}
# Search effort / public attention: floor + (1 - floor) * exp(-t / decay_h). The dog floor is
# calibrated to the median recovery time (a lower floor puts more finds in the first days);
# cats keep their starting floor.
SEARCH_DECAY_H = {"cat_indoor": 168.0, "cat_outdoor": 168.0, "dog": 48.0}
SEARCH_FLOOR = {"cat_indoor": 0.15, "cat_outdoor": 0.15, "dog": 0.10}

# Calibration starting points and run size.
CALIBRATION = {"start_speed_mult": 1.0, "start_detect_mult": 1.0,
               "cases_per_group": 600, "iterations": 14}  # spread over every region

# Training-table generation.
FEATURES = {"negatives": 31, "snapshot_hours": [6, 24, 72, 168], "snapshots_per_case": 2,
            "radius_m": {"cat": 4000, "dog": 12000}, "min_r_m": 10}
