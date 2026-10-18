"""Build the M3 adoption-speed training set.

Sources (put the raw files in data/raw/, see README.md for download commands):
    austin_intakes.csv, austin_outcomes.csv   Austin Animal Center open data (always used)
    petfinder/train.csv (+ breed_labels.csv, color_labels.csv)
                                               Kaggle PetFinder.my Adoption Prediction (used if present)

Outputs (data/processed/):
    adoption_speed.csv          clean training table, one row per adoptable stay
    adoption_speed_stats.json   row counts, label distribution, filter log

Labels follow the PetFinder / PawConnect classes (backend/app/ai/base.py):
    0 adopted the same day, 1 within 1-7 days, 2 within 8-30 days, 3 within 31-99 days,
    4 not adopted after 100+ days

Usage:  python build_dataset.py [--raw data/raw] [--out data/processed]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# Rough adult size for the dog breeds that cover ~98% of Austin dog intakes.
# Small < ~10 kg, Medium 10-25 kg, Large > 25 kg.
DOG_SIZE = {
    "Small": [
        "Chihuahua Shorthair", "Chihuahua Longhair", "Dachshund", "Dachshund Longhair", "Dachshund Wirehair",
        "Miniature Poodle", "Toy Poodle", "Yorkshire Terrier", "Miniature Schnauzer", "Jack Russell Terrier",
        "Rat Terrier", "Cairn Terrier", "Shih Tzu", "Maltese", "Pug", "Miniature Pinscher", "Border Terrier",
        "Pomeranian", "Boston Terrier", "Lhasa Apso", "Manchester Terrier", "Norfolk Terrier", "Pekingese",
        "Wire Hair Fox Terrier", "Bichon Frise", "West Highland", "Papillon", "Norwich Terrier",
        "Smooth Fox Terrier", "Shetland Sheepdog", "Bruss Griffon", "Parson Russell Terrier", "Scottish Terrier",
        "Cavalier Span", "Tibetan Spaniel", "Havanese", "Toy Fox Terrier", "Pbgv", "Italian Greyhound",
        "Silky Terrier", "Japanese Chin", "Affenpinscher", "Chinese Crested", "Coton De Tulear", "Schipperke",
        "Lowchen", "Dandie Dinmont", "Sealyham Terr", "Skye Terrier", "Lakeland Terrier", "Welsh Terrier",
        "Patterdale Terr", "Feist", "Mexican Hairless", "Bichon Frise", "French Bulldog", "Cardigan Welsh Corgi",
        "Pembroke Welsh Corgi", "Shiba Inu", "Basenji", "American Eskimo",
    ],
    "Medium": [
        "Australian Cattle Dog", "Border Collie", "Australian Shepherd", "Beagle", "Pointer", "Staffordshire",
        "Black Mouth Cur", "American Pit Bull Terrier", "Pit Bull", "Plott Hound", "Australian Kelpie",
        "American Staffordshire Terrier", "Blue Lacy", "Cocker Spaniel", "Chinese Sharpei", "Queensland Heeler",
        "English Bulldog", "Bulldog", "Bull Terrier", "Collie Smooth", "Carolina Dog", "Redbone Hound", "Whippet",
        "Standard Schnauzer", "Harrier", "Soft Coated Wheaten Terrier", "Treeing Walker Coonhound",
        "American Foxhound", "English Coonhound", "Pharaoh Hound", "Brittany", "English Pointer",
        "Basset Hound", "Siberian Husky", "Catahoula", "Chow Chow", "Spinone Italiano", "Boykin Span",
        "English Springer Spaniel", "Welsh Springer Spaniel", "Field Spaniel", "Keeshond", "Finnish Spitz",
        "Canaan Dog", "Entlebucher", "Bluetick Hound", "Treeing Cur", "Mountain Cur", "Bearded Collie",
        "Nova Scotia Duck Tolling Retriever", "Portuguese Water Dog", "Samoyed", "Schnauzer Giant",
    ],
    "Large": [
        "Labrador Retriever", "German Shepherd", "Boxer", "Great Pyrenees", "Rottweiler", "Anatol Shepherd",
        "American Bulldog", "Doberman Pinsch", "Golden Retriever", "Alaskan Husky", "Mastiff", "Great Dane",
        "Belgian Malinois", "Rhod Ridgeback", "Flat Coat Retriever", "Akita", "German Shorthair Pointer",
        "Weimaraner", "Dalmatian", "Vizsla", "Dutch Shepherd", "Collie Rough", "Standard Poodle", "Alaskan Malamute",
        "Greyhound", "Bloodhound", "St. Bernard Rough Coat", "St. Bernard Smooth Coat", "Cane Corso", "Bullmastiff",
        "Black", "Black/Tan Hound", "Bernese Mountain Dog", "Newfoundland", "Leonberger", "Irish Wolfhound",
        "Neapolitan Mastiff", "Dogo Argentino", "Presa Canario", "Bull Mastiff", "Kuvasz", "Old English Sheepdog",
        "Chesa Bay Retr", "German Wirehaired Pointer", "Belgian Tervuren", "Belgian Sheepdog", "Irish Setter",
        "English Setter", "Gordon Setter", "Boerboel", "Tosa", "Dogue De Bordeaux", "Saluki", "Borzoi",
        "Greater Swiss Mountain Dog", "Bouv Flandres", "Beauceron", "Briard", "Kangal", "Landseer",
        "Swiss Hound", "Wolf Hybrid", "Hovawart",
    ],
}
DOG_SIZE_BY_BREED = {b: size for size, breeds in DOG_SIZE.items() for b in breeds}
SMALL_CAT_BREEDS = {"Singapura", "Devon Rex", "Cornish Rex", "Munchkin"}
LARGE_CAT_BREEDS = {"Maine Coon", "Ragdoll", "Norwegian Forest Cat", "Siberian", "Savannah", "Bengal", "Ragamuffin"}

# Animals that are not adoptable when they arrive, so their time in care is not a measure of demand.
NOT_YET_ADOPTABLE_CONDITIONS = {"Neonatal", "Nursing", "Pregnant", "Feral", "Space", "Agonal", "Panleuk", "Parvo"}
ADOPTION_INTAKE_TYPES = {"Stray", "Owner Surrender", "Abandoned"}
MIN_AGE_MONTHS = 2.0  # under 8 weeks animals wait in foster until they can be listed
LONG_STAY_DAYS = 100
COLUMNS = [
    "record_id", "source", "animal_id", "intake_date", "split",
    # The model inputs PawConnect has for every listing (ListingFeatures)
    "species", "age_group", "breed", "size", "photo_count", "description", "fee_amount", "fee_currency",
    # Extra signals, useful for training and analysis
    "age_months", "breed_primary", "is_mixed", "size_source", "sex", "sterilized", "color_primary",
    "is_multicolor", "has_name", "intake_type", "intake_condition", "intake_month",
    # Target
    "days_to_outcome", "outcome_type", "adoption_speed",
]


def speed_class(days: pd.Series, adopted: pd.Series) -> pd.Series:
    """PawConnect speed class from days in care; non-adoptions are only labelled once they pass 100 days."""
    cls = pd.cut(days, [-1, 0, 7, 30, LONG_STAY_DAYS - 1, np.inf], labels=[0, 1, 2, 3, 4]).astype("float")
    return cls.where(adopted | (days >= LONG_STAY_DAYS))


def age_group(species: pd.Series, months: pd.Series) -> pd.Series:
    senior_from = np.where(species == "Cat", 10 * 12, 8 * 12)
    return pd.Series(np.select([months < 12, months >= senior_from], ["young", "senior"], "adult"), index=months.index)


def sentence_case(s: pd.Series) -> pd.Series:
    """'Labrador Retriever' -> 'Labrador retriever', matching how breeds are written in the app."""
    return s.str[:1].str.upper() + s.str[1:].str.lower()


def parse_age_months(text: pd.Series) -> pd.Series:
    m = text.str.extract(r"(-?\d+)\s*(year|month|week|day)")
    n = pd.to_numeric(m[0], errors="coerce")
    per_unit = m[1].map({"year": 12.0, "month": 1.0, "week": 7 / 30.44, "day": 1 / 30.44})
    months = n * per_unit
    return months.where(months >= 0)


# --------------------------------------------------------------------------- Austin


def load_austin(raw: Path, log: dict) -> pd.DataFrame:
    intakes = pd.read_csv(raw / "austin_intakes.csv")
    outcomes = pd.read_csv(raw / "austin_outcomes.csv")
    intakes["in_dt"] = pd.to_datetime(intakes["DateTime"], format="%m/%d/%Y %I:%M:%S %p")
    # Outcome times are local; some rows carry a UTC offset, which we drop to stay in local time.
    outcomes["out_dt"] = pd.to_datetime(outcomes["DateTime"].str[:19], format="%Y-%m-%dT%H:%M:%S")
    log["austin_raw"] = {"intakes": len(intakes), "outcomes": len(outcomes)}

    intakes = intakes.drop_duplicates(["Animal ID", "in_dt"])
    outcomes = outcomes.drop_duplicates(["Animal ID", "out_dt"])

    # Pair each intake with the first outcome after it, as long as the animal was not taken in again first.
    intakes = intakes.sort_values("in_dt")
    intakes["next_in_dt"] = intakes.groupby("Animal ID")["in_dt"].shift(-1)
    outcomes = outcomes.sort_values("out_dt")
    stays = pd.merge_asof(
        intakes,
        outcomes[["Animal ID", "out_dt", "Outcome Type", "Outcome Subtype"]],
        left_on="in_dt", right_on="out_dt", by="Animal ID", direction="forward",
    )
    stays.loc[stays["out_dt"] > stays["next_in_dt"], ["out_dt", "Outcome Type"]] = [pd.NaT, np.nan]

    data_end = outcomes["out_dt"].max()
    steps = []

    def keep(mask: pd.Series, why: str) -> None:
        nonlocal stays
        steps.append({"filter": why, "dropped": int((~mask).sum())})
        stays = stays[mask]

    keep(stays["Animal Type"].isin(["Dog", "Cat"]), "not a dog or cat")
    keep(stays["Intake Type"].isin(ADOPTION_INTAKE_TYPES), "intake type not headed for adoption (wildlife, public assist, euthanasia request)")
    keep(~stays["Intake Condition"].isin(NOT_YET_ADOPTABLE_CONDITIONS), "not adoptable on arrival (neonatal, nursing, pregnant, feral, ...)")
    stays = stays.assign(age_months=parse_age_months(stays["Age upon Intake"]))
    keep(stays["age_months"] >= MIN_AGE_MONTHS, "younger than 8 weeks or unknown age")
    keep(stays["in_dt"] <= data_end - pd.Timedelta(days=LONG_STAY_DAYS), f"taken in during the last {LONG_STAY_DAYS} days of data (outcome not known yet)")
    keep(stays["out_dt"].notna() & stays["Outcome Type"].notna(), "no recorded outcome")
    keep(~stays["Outcome Type"].isin(["Return to Owner", "Rto-Adopt", "Missing", "Stolen", "Lost"]), "reclaimed by owner or lost/stolen")

    days = (stays["out_dt"].dt.normalize() - stays["in_dt"].dt.normalize()).dt.days
    stays = stays.assign(days_to_outcome=days, adoption_speed=speed_class(days, stays["Outcome Type"] == "Adoption"))
    keep(stays["days_to_outcome"] >= 0, "outcome before intake")
    keep(stays["adoption_speed"].notna(), f"left the shelter (transfer, euthanasia, death) before {LONG_STAY_DAYS} days without adoption")
    log["austin_filters"] = steps

    breed_raw = stays["Breed"].fillna("Unknown")
    primary = breed_raw.str.split("/").str[0].str.replace(r"\s+Mix$", "", regex=True).str.strip()
    is_mixed = breed_raw.str.contains(r"Mix|/", regex=True)
    species = stays["Animal Type"]
    size = pd.Series(np.where(species == "Dog", primary.map(DOG_SIZE_BY_BREED), None), index=stays.index)
    size_source = np.where(size.notna(), "breed_lookup", "default")
    cat_size = np.select([primary.isin(SMALL_CAT_BREEDS), primary.isin(LARGE_CAT_BREEDS)], ["Small", "Large"], "Medium")
    size = size.fillna(pd.Series(np.where(species == "Cat", cat_size, "Medium"), index=stays.index))
    size_source = np.where((species == "Cat") & (cat_size != "Medium"), "breed_lookup", size_source)

    sex = stays["Sex upon Intake"].fillna("Unknown")
    color = stays["Color"].fillna("Unknown")
    out = pd.DataFrame({
        "source": "austin",
        "animal_id": stays["Animal ID"],
        "intake_date": stays["in_dt"].dt.date.astype(str),
        "species": species,
        "age_group": age_group(species, stays["age_months"]),
        "breed": sentence_case(primary) + np.where(is_mixed, " mix", ""),
        "size": size,
        "photo_count": np.nan,
        "description": "",
        "fee_amount": np.nan,
        "fee_currency": "",
        "age_months": stays["age_months"].round(1),
        "breed_primary": primary,
        "is_mixed": is_mixed,
        "size_source": size_source,
        "sex": np.select([sex.str.contains("Female"), sex.str.contains("Male")], ["Female", "Male"], "Unknown"),
        "sterilized": np.select([sex.str.contains("Neutered|Spayed"), sex.str.contains("Intact")], ["yes", "no"], "unknown"),
        "color_primary": color.str.split("/").str[0],
        "is_multicolor": color.str.contains("/"),
        "has_name": stays["Name"].notna(),
        "intake_type": stays["Intake Type"],
        "intake_condition": stays["Intake Condition"],
        "intake_month": stays["in_dt"].dt.month,
        "days_to_outcome": stays["days_to_outcome"].astype(int),
        "outcome_type": stays["Outcome Type"],
        "adoption_speed": stays["adoption_speed"].astype(int),
    })
    out["split"] = time_split(out, stays["in_dt"])
    return out


def time_split(df: pd.DataFrame, when: pd.Series) -> pd.Series:
    """Newest 12 months -> test, the 12 months before -> val, older -> train. An animal's stays share one split."""
    first_seen = when.groupby(df["animal_id"]).transform("min")
    end = when.max()
    return pd.Series(
        np.select([first_seen > end - pd.DateOffset(months=12), first_seen > end - pd.DateOffset(months=24)], ["test", "val"], "train"),
        index=df.index,
    )


# --------------------------------------------------------------------------- PetFinder


def load_petfinder(raw: Path, log: dict) -> pd.DataFrame | None:
    folder = raw / "petfinder"
    if not (folder / "train.csv").exists():
        log["petfinder"] = "not found, skipped (see README.md to add it)"
        return None
    pf = pd.read_csv(folder / "train.csv")
    breeds = pd.read_csv(folder / "breed_labels.csv").set_index("BreedID")["BreedName"].to_dict()
    colors = pd.read_csv(folder / "color_labels.csv").set_index("ColorID")["ColorName"].to_dict()
    log["petfinder_raw"] = len(pf)
    # Listings of several animals at once (Quantity > 1) behave differently from a single pet listing.
    pf = pf[pf["Quantity"] == 1]

    species = pf["Type"].map({1: "Dog", 2: "Cat"})
    b1 = pf["Breed1"].map(breeds)
    b2 = pf["Breed2"].map(breeds)
    is_mixed = (b1 == "Mixed Breed") | (b2.notna() & (b2 != b1))
    primary = b1.where(b1 != "Mixed Breed", b2).fillna("Mixed Breed")
    size = pf["MaturitySize"].map({1: "Small", 2: "Medium", 3: "Large", 4: "Large"})
    yes_no = {1: "yes", 2: "no", 3: "unknown"}

    out = pd.DataFrame({
        "source": "petfinder",
        "animal_id": pf["PetID"],
        "intake_date": "",
        "species": species,
        "age_group": age_group(species, pf["Age"].astype(float)),
        "breed": sentence_case(primary.str.replace("Mixed Breed", "Mixed")) + np.where(is_mixed & (primary != "Mixed Breed"), " mix", ""),
        "size": size.fillna("Medium"),
        "photo_count": pf["PhotoAmt"].astype(int),
        "description": pf["Description"].fillna("").str.strip(),
        "fee_amount": pf["Fee"].astype(float),
        "fee_currency": "MYR",
        "age_months": pf["Age"].astype(float),
        "breed_primary": primary,
        "is_mixed": is_mixed,
        "size_source": np.where(size.notna(), "reported", "default"),
        "sex": pf["Gender"].map({1: "Male", 2: "Female"}).fillna("Unknown"),
        "sterilized": pf["Sterilized"].map(yes_no),
        "color_primary": pf["Color1"].map(colors).fillna("Unknown"),
        "is_multicolor": pf["Color2"] > 0,
        "has_name": pf["Name"].notna() & ~pf["Name"].fillna("").str.contains(r"no name|unknown|^\W*$", case=False, regex=True),
        "intake_type": "",
        "intake_condition": pf["Health"].map({1: "Normal", 2: "Injured", 3: "Injured"}),
        "intake_month": np.nan,
        "days_to_outcome": np.nan,
        "outcome_type": np.where(pf["AdoptionSpeed"] == 4, "Not adopted", "Adoption"),
        "adoption_speed": pf["AdoptionSpeed"].astype(int),
    })
    # PetFinder has no dates; split by rescuer so one rescuer's listings never leak across splits.
    bucket = pf["RescuerID"].map(lambda r: int(re.sub(r"[^0-9a-f]", "", str(r))[:8] or "0", 16) % 100)
    out["split"] = np.select([bucket < 15, bucket < 30], ["test", "val"], "train")
    return out


# --------------------------------------------------------------------------- main


def describe(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
        "by_species": df["species"].value_counts().to_dict(),
        "adoption_speed": {str(k): int(v) for k, v in df["adoption_speed"].value_counts().sort_index().items()},
        "adoption_speed_by_source": {
            s: {str(k): round(float(v), 3) for k, v in g["adoption_speed"].value_counts(normalize=True).sort_index().items()}
            for s, g in df.groupby("source")
        },
        "age_group": df["age_group"].value_counts().to_dict(),
        "size": df["size"].value_counts().to_dict(),
        "size_source": df["size_source"].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "raw")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "processed")
    args = ap.parse_args()

    log: dict = {}
    parts = [load_austin(args.raw, log), load_petfinder(args.raw, log)]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    df.insert(0, "record_id", df["source"] + "-" + df["animal_id"].astype(str) + "-" + df.groupby(["source", "animal_id"]).cumcount().astype(str))
    df = df[COLUMNS]

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "adoption_speed.csv", index=False)
    stats = {"filters": log, "dataset": describe(df)}
    (args.out / "adoption_speed_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
