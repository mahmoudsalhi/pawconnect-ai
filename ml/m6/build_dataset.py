"""Merge every batch in data/raw/ into one clean, grouped, cross-validation-ready dataset.

    ml/.venv/Scripts/python ml/m6/build_dataset.py

Fixes applied (see the M6 review notes):
1. v1 over-merged templates: the review pass folded different scenarios into one template_id,
   so whole families landed on one side of the split. The original id is recovered from the
   `changed` column ("template_id OLD -> NEW").
2. B05 gave each translation its own id (T5-HN-FOSTER-FR, -EN...). Language suffixes are stripped
   so translations of one message stay in one group.
3. Near-copies across batches (e.g. B06-079 ~ B05-079) are merged into one group, so a message
   and its near-copy can never sit on both sides of a split.
Rows are then assigned to 5 folds with StratifiedGroupKFold (groups kept whole, scam ratio balanced).
"""

import csv
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "m6_dataset.csv"

LABELS = {"upfront_payment", "fake_fee", "off_platform", "phishing_link", "impersonation", "pressure", "legit"}
LANGUAGES = {"darija_latin", "darija_arabic", "french", "msa_arabic", "english", "mixed"}
CHANNELS = {"message", "listing"}
DIFFICULTIES = {"easy", "hard_negative", "obfuscated"}
LANG_SUFFIX = re.compile(r"-(DL|DA|FR|MSA|MIX|EN)$")
RENAMED = re.compile(r"template_id (\S+) -> (\S+)")
NEAR_DUP = 0.80
N_FOLDS, SEED = 5, 42

# Legit look-alike families that must be seen in training AND tested (problem 1 in the review).
FAMILIES = {
    "reward paid on return": r"REWARD|PAID-AFTER|RIB-TO-PAY",
    "shelter fee paid on site": r"SHELTER-FEE-ONSITE|SHELTER-HOMECHECK|SHELTER-FULL|PAID-AT-SHELTER",
    "WhatsApp / phone by choice": r"WHATSAPP|ASK-NUMBER|LOCATION",
    "photo or video link": r"PHOTO-LINK|VIDEO",
    "real vet or clinic": r"VET",
    "free help (foster, transport)": r"FOSTER|NO-MONEY|REFUSES-REWARD|TRANSPORT|TAXI",
    "meet in public / verify owner": r"MEET|VERIFY|ID-IN-PERSON|BRING-PAPERS",
    "real urgency": r"URGENT|DEADLINE|MEDS|PREGNANT|LANDLORD",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            df = pd.DataFrame(list(csv.DictReader(fh)), dtype=str).fillna("")
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        sys.exit(f"No CSV files in {RAW}")
    df = pd.concat(frames, ignore_index=True).fillna("")
    if "changed" not in df:
        df["changed"] = ""
    return df


def validate(df: pd.DataFrame) -> list[str]:
    problems = []
    dup_ids = df["id"][df["id"].duplicated()].tolist()
    if dup_ids:
        problems.append(f"duplicate ids: {dup_ids}")
    for col, allowed in (("language", LANGUAGES), ("channel", CHANNELS), ("difficulty", DIFFICULTIES)):
        bad = df.loc[~df[col].isin(allowed), "id"].tolist()
        if bad:
            problems.append(f"unknown {col} in: {bad}")
    for _, r in df.iterrows():
        labels = r["labels"].split("|")
        if set(labels) - LABELS:
            problems.append(f"{r['id']}: unknown label(s) {set(labels) - LABELS}")
        if r["is_scam"] not in ("0", "1"):
            problems.append(f"{r['id']}: is_scam must be 0 or 1")
        elif ("legit" in labels) != (r["is_scam"] == "0") or ("legit" in labels and len(labels) > 1):
            problems.append(f"{r['id']}: labels '{r['labels']}' contradict is_scam={r['is_scam']}")
        if not r["text"].strip():
            problems.append(f"{r['id']}: empty text")
    return problems


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} rows from {df['source_file'].nunique()} files: {', '.join(df['source_file'].unique())}")

    problems = validate(df)
    if problems:
        print("\nDATA PROBLEMS (fix in the raw file, then rerun):")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # Exact duplicate texts: keep the first occurrence.
    df["_norm"] = df["text"].map(norm)
    exact = df[df["_norm"].duplicated(keep="first")]
    if len(exact):
        print(f"Dropped {len(exact)} exact duplicate text(s): {exact['id'].tolist()}")
    df = df.drop(exact.index).reset_index(drop=True)

    # Fix 1 + 2: recover the original template id, then strip language suffixes.
    before = df["template_id"].copy()
    df["template_id"] = [
        LANG_SUFFIX.sub("", m.group(1) if (m := RENAMED.search(c)) else t)
        for t, c in zip(df["template_id"], df["changed"])
    ]
    print(f"Template ids changed by fixes 1+2: {int((before != df['template_id']).sum())} rows")

    # Fix 3: groups = template ids, joined wherever two texts are near-copies.
    uf = UnionFind()
    for t in df["template_id"]:
        uf.find(t)
    texts, tids, ids = df["_norm"].tolist(), df["template_id"].tolist(), df["id"].tolist()
    near = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if tids[i] == tids[j]:
                continue
            sm = SequenceMatcher(None, texts[i], texts[j])
            if sm.quick_ratio() >= NEAR_DUP and sm.ratio() >= NEAR_DUP:
                near.append((ids[i], ids[j], sm.ratio()))
                uf.union(tids[i], tids[j])
    roots = {t: uf.find(t) for t in df["template_id"].unique()}
    numbered = {root: f"G{n:03d}" for n, root in enumerate(sorted(set(roots.values())), 1)}
    df["group"] = df["template_id"].map(lambda t: numbered[roots[t]])

    print(f"Template ids: {df['template_id'].nunique()}  ->  groups after near-copy merge: {df['group'].nunique()}")
    print(f"Near-copies merged across different templates: {len(near)}")
    for a, b, s in near:
        print(f"  {s:.2f}  {a} ~ {b}")

    # 5 folds: whole groups per fold, scam ratio balanced.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, test_idx) in enumerate(sgkf.split(df, df["is_scam"], df["group"])):
        df.loc[df.index[test_idx], "fold"] = k

    df["batch"] = df["id"].str.slice(0, 3)
    cols = ["id", "batch", "text", "channel", "language", "is_scam", "labels", "difficulty", "template_id", "group", "fold"]
    df[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(HERE.parent.parent)}  ({len(df)} rows)")

    print("\nRows per fold (scam / legit):")
    for k in range(N_FOLDS):
        f = df[df["fold"] == k]
        print(f"  fold {k}: {len(f):3} rows  ({(f['is_scam'] == '1').sum()} / {(f['is_scam'] == '0').sum()})")

    print("\nLegit look-alike families (hard negatives): each should span 2+ folds,")
    print("so whenever it is tested, other examples of it are in training.")
    hn = df[(df["is_scam"] == "0") & (df["difficulty"] == "hard_negative")]
    print(f"  {'family':32} {'rows':>4} {'groups':>6}  folds")
    for name, pattern in FAMILIES.items():
        fam = hn[hn["template_id"].str.contains(pattern)]
        folds = sorted(fam["fold"].unique().tolist())
        status = "OK" if len(folds) >= 2 else "ONE FOLD ONLY"
        print(f"  {name:32} {len(fam):4} {fam['group'].nunique():6}  {folds}  {status}")


if __name__ == "__main__":
    main()
