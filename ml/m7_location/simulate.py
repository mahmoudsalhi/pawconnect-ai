"""Simulate lost pets moving through a real city until they are found, die or stay missing.

Movement is a step-selection model, the standard tool in movement ecology: every 10 minutes
a moving animal draws K candidate steps (length from a gamma distribution, direction from a
von Mises around its current heading or goal), scores each endpoint by the habitat it lands
in (cover, food, people, corridors, roads, water) and picks one with softmax probability.
A behavioural state machine (rest/hide, explore, forage, flee, homing) drives speed and
goals; time of day, heat and hunger modulate it.

Recovery is a set of competing hazards evaluated every step: found by a member of the
public (depends on foot traffic, visibility, how approachable the pet is, and flyers near
the last-seen point), found by the owner searching near the last-seen point, picked up by a
vet/municipality, taken in by a stranger (moved by car), returned home on its own, or died
(road crossings and a small background rate).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np

import config as C

REST, EXPLORE, FORAGE, FLEE, HOMING = range(5)
STATE_NAMES = np.array(["rest_hide", "explore", "forage", "flee", "homing"])
GROUPS = ["cat_indoor", "cat_outdoor", "dog"]
FEATS = ["cover", "food", "people", "corridor", "green", "building", "farm"]
K = 10
STEP_S = C.STEP_MIN * 60
STEPS = C.MAX_DAYS * 24 * 60 // C.STEP_MIN

OUTCOMES = ["returned_home", "found_by_owner", "found_by_public", "picked_up_vet_municipality",
            "taken_in_by_stranger", "found_dead", "not_found"]
O = {name: i for i, name in enumerate(OUTCOMES)}


def _pick(rng, table: dict, n: int) -> np.ndarray:
    keys = list(table)
    p = np.array([table[k] for k in keys], float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), n, p=p / p.sum())]


@dataclass
class Params:
    speed_mult: np.ndarray   # per group: how far pets travel
    detect_mult: np.ndarray  # per group: how easily they are found
    floor: np.ndarray          # per group: search effort left after the first days (share of initial)
    relocate_mult: np.ndarray  # per group: how often strangers take the pet in and drive it away

    KEYS = ("speed_mult", "detect_mult", "search_floor", "relocate_mult")

    @classmethod
    def default(cls):
        return cls(np.full(3, C.CALIBRATION["start_speed_mult"]), np.full(3, C.CALIBRATION["start_detect_mult"]),
                   np.array([C.SEARCH_FLOOR[g] for g in GROUPS], float), np.ones(3))

    def as_array(self):
        return np.stack([self.speed_mult, self.detect_mult, self.floor, self.relocate_mult])

    @classmethod
    def from_array(cls, a):
        return cls(*(np.array(r, float) for r in a))

    def to_dict(self):
        a = self.as_array()
        return {g: {k: float(a[j, i]) for j, k in enumerate(self.KEYS)} for i, g in enumerate(GROUPS)}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.array([d[g].get(k, 1.0) for g in GROUPS], float) for k in cls.KEYS))


# --- case generation -----------------------------------------------------------------------
def sample_cases(n: int, rng: np.random.Generator, grid, layers, force_group: np.ndarray | None = None) -> dict:
    """Draw n lost-pet profiles, loss times and home / last-seen locations."""
    if force_group is None:
        species = _pick(rng, C.SPECIES, n)
        lifestyle = np.where(species == "cat", _pick(rng, C.CAT["lifestyle"], n), "household")
    else:
        species = np.where(force_group == 2, "dog", "cat").astype(object)
        lifestyle = np.array(["indoor_only", "outdoor_access", "household"], dtype=object)[force_group]
    cat = species == "cat"
    grp = np.where(~cat, 2, np.where(lifestyle == "indoor_only", 0, 1))

    def per_species(cat_table, dog_table):
        return np.where(cat, _pick(rng, cat_table, n), _pick(rng, dog_table, n))

    temperament = per_species(C.CAT["temperament"], C.DOG["temperament"])
    age_group = per_species(C.CAT["age_group"], C.DOG["age_group"])
    sex = per_species(C.CAT["sex"], C.DOG["sex"])
    size = np.where(cat, "small", _pick(rng, C.DOG["size"], n))
    breed = np.where(cat, "domestic_cat", _pick(rng, C.DOG["breed_group"], n))
    neutered = rng.random(n) < np.where(cat, C.CAT["neutered"], C.DOG["neutered"])
    collar = rng.random(n) < np.where(cat, C.CAT["collar_tag"], C.DOG["collar_tag"])
    chip = rng.random(n) < np.where(cat, C.CAT["microchipped"], C.DOG["microchipped"])
    circ = np.empty(n, dtype=object)
    for g, table in ((0, C.CAT["circumstance"]["indoor_only"]), (1, C.CAT["circumstance"]["outdoor_access"]),
                     (2, C.DOG["circumstance"])):
        m = grp == g
        circ[m] = _pick(rng, table, int(m.sum()))

    # When: month by circumstance, hour by circumstance family, years 2023-2025.
    month = np.empty(n, int)
    hour = np.empty(n, float)
    for c in np.unique(circ):
        m = circ == c
        mw = np.array(C.MONTHS.get(c, C.MONTHS["default"]), float)
        month[m] = rng.choice(12, m.sum(), p=mw / mw.sum()) + 1
        hw = np.array(C.HOURS[C.HOUR_FAMILY[c]], float)
        hour[m] = rng.choice(24, m.sum(), p=hw / hw.sum()) + rng.random(m.sum())
    year = rng.integers(2023, 2026, n)
    day = np.array([rng.integers(1, calendar.monthrange(int(y), int(mo))[1] + 1) for y, mo in zip(year, month)])

    # Where: homes in residential / built-up cells away from the box edge.
    margin = int(3000 / grid.cell)
    res = layers["residential"].copy()
    res[layers["water"] > 0] = 0
    res[:margin], res[-margin:], res[:, :margin], res[:, -margin:] = 0, 0, 0, 0
    w = (res * (1 + 2 * layers["building"])).ravel()
    cells = rng.choice(w.size, n, p=w / w.sum())
    hy = (cells // grid.W + rng.random(n)) * grid.cell
    hx = (cells % grid.W + rng.random(n)) * grid.cell

    def offset(mask, lo, hi):
        """Move points by a log-uniform distance in [lo, hi] m, avoiding water."""
        ox, oy = hx.copy(), hy.copy()
        idx = np.flatnonzero(mask)
        for _ in range(10):
            if not idx.size:
                break
            d = np.exp(rng.uniform(np.log(lo), np.log(hi), idx.size))
            a = rng.uniform(0, 2 * np.pi, idx.size)
            nx = np.clip(hx[idx] + d * np.cos(a), 0, grid.W * grid.cell - 1)
            ny = np.clip(hy[idx] + d * np.sin(a), 0, grid.H * grid.cell - 1)
            ok = layers["water"][(ny // grid.cell).astype(int), (nx // grid.cell).astype(int)] == 0
            ox[idx[ok]], oy[idx[ok]] = nx[ok], ny[ok]
            idx = idx[~ok]
        return ox, oy

    ox, oy = hx.copy(), hy.copy()
    for c, lo, hi in (("slipped_leash_walk", 100, 1500), ("transport_escape", 500, 5000),
                      ("car_accident", 1000, 8000)):
        m = circ == c
        px_, py_ = offset(m, lo, hi)
        ox[m], oy[m] = px_[m], py_[m]
    fw_walk = (circ == "fireworks_noise") & ~cat & (rng.random(n) < 0.3)
    wx, wy = offset(fw_walk, 50, 800)
    ox[fw_walk], oy[fw_walk] = wx[fw_walk], wy[fw_walk]

    # Homing target: home, except after a move, when pets head for the old territory.
    tx, ty = hx.copy(), hy.copy()
    moved = circ == "moved_house"
    mx_, my_ = offset(moved, 1000, 8000)
    tx[moved], ty[moved] = mx_[moved], my_[moved]

    sigma = np.array([C.LAST_SEEN_SIGMA_M[c] for c in circ], float)
    rx = ox + rng.normal(0, 1, n) * sigma
    ry = oy + rng.normal(0, 1, n) * sigma

    return dict(n=n, grp=grp, species=species, lifestyle=lifestyle, temperament=temperament,
                age_group=age_group, sex=sex, size=size, breed_group=breed, neutered=neutered,
                collar_tag=collar, microchipped=chip, circumstance=circ, year=year, month=month,
                day=day, hour=hour, home_x=hx, home_y=hy, origin_x=ox, origin_y=oy,
                target_x=tx, target_y=ty, old_home_x=np.where(moved, tx, np.nan),
                old_home_y=np.where(moved, ty, np.nan), reported_x=rx, reported_y=ry,
                last_seen_sigma_m=sigma)


def _gauss_hour(h, mu, sd):
    d = np.abs((h - mu + 12) % 24 - 12)
    return np.exp(-0.5 * (d / sd) ** 2)


# --- simulation ----------------------------------------------------------------------------
def simulate(cases: dict, grid, layers, params: Params, rng: np.random.Generator,
             record: bool = True, temp_offset: float = 0.0) -> dict:
    n = cases["n"]
    grp = cases["grp"]
    is_cat = grp < 2
    cell = grid.cell
    H, W = grid.H, grid.W
    F = np.stack([layers[f] for f in FEATS])            # (7, H, W)
    major, exposed, water, people_l = layers["major"], layers["exposed"], layers["water"], layers["people"]

    # Per-case behaviour parameters.
    tem = cases["temperament"]
    mod = np.ones(n)
    mod *= np.array([C.SPEED_MOD["temperament"][t] for t in tem])
    cat_age = np.array([C.SPEED_MOD["cat_age"].get(a, 1.0) for a in cases["age_group"]])
    dog_age = np.array([C.SPEED_MOD["dog_age"].get(a, 1.0) for a in cases["age_group"]])
    mod *= np.where(is_cat, cat_age, dog_age)
    mod *= np.where(is_cat & (cases["sex"] == "male") & ~cases["neutered"], C.SPEED_MOD["cat_intact_male"], 1.0)
    mod *= np.where(is_cat, 1.0, [C.SPEED_MOD["dog_size"].get(s, 1.0) for s in cases["size"]])
    mod *= np.where(is_cat, 1.0, [C.SPEED_MOD["dog_breed"].get(b, 1.0) for b in cases["breed_group"]])
    speed = np.array([C.SPEED[g] for g in GROUPS])[grp] * (params.speed_mult[grp] * mod)[:, None]  # (n, 5)
    wts = np.array([C.WEIGHTS[g] for g in GROUPS])[grp].copy()                          # (n, 9)
    wts[:, 2] = [C.PEOPLE_WEIGHT[t] for t in tem]
    approach = np.array([C.APPROACHABILITY[t] for t in tem])
    homing_ab = np.array([C.HOMING_ABILITY[g] for g in GROUPS])[grp]
    homing_r = np.array([C.HOMING_RANGE_M[g] for g in GROUPS])[grp]
    search_r = np.array([C.OWNER_SEARCH_R_M[g] for g in GROUPS])[grp]
    dm = params.detect_mult[grp]
    decay = np.array([C.SEARCH_DECAY_H[g] for g in GROUPS])[grp]
    floor = params.floor[grp]
    reloc_m = params.relocate_mult[grp]
    reloc_hi = np.where(is_cat, C.TAKEN_IN_RANGE_M["cat"][1], C.TAKEN_IN_RANGE_M["dog"][1])
    hz = {k: np.array([v[g] for g in GROUPS])[grp] for k, v in C.HAZARD.items() if isinstance(v, dict)}
    road_p = np.array([C.ROAD_DEATH_P[g] for g in GROUPS])[grp]
    bg_p = np.array([C.BACKGROUND_DEATH_P[g] for g in GROUPS])[grp]
    tag = 1 + 0.5 * cases["collar_tag"] + 0.3 * cases["microchipped"]
    xeno_dog = (~is_cat) & (tem == "xenophobic")
    temp_mean = np.array(C.MONTH_TEMP_C)[cases["month"] - 1] + temp_offset
    kappa = np.array(C.KAPPA)
    vis_rest = np.where(is_cat, 0.05, 0.3)
    vis_owner_rest = np.where(is_cat, 0.25, 0.5)

    x, y = cases["origin_x"].copy(), cases["origin_y"].copy()
    ox, oy = cases["origin_x"], cases["origin_y"]
    tx, ty = cases["target_x"], cases["target_y"]
    hx, hy = cases["home_x"], cases["home_y"]
    heading = rng.uniform(0, 2 * np.pi, n)
    flee0 = rng.random(n) < np.array([C.FLEE_START[c] for c in cases["circumstance"]])
    state = np.where(flee0, FLEE, np.where(is_cat, REST, EXPLORE))
    active = np.ones(n, bool)
    outcome = np.full(n, O["not_found"])
    t_end = np.full(n, float(C.MAX_DAYS * 24))
    end_x, end_y = np.full(n, np.nan), np.full(n, np.nan)
    path_len = np.zeros(n)
    max_d = np.zeros(n)
    relocated = np.zeros(n, bool)
    reloc_x, reloc_y = np.full(n, np.nan), np.full(n, np.nan)
    died_unfound = np.zeros(n, bool)

    # Residential cells strangers might live in, for taken-in relocations.
    res_cells = np.flatnonzero((layers["residential"] > 0).ravel() & (water.ravel() == 0))
    res_pool = rng.choice(res_cells, min(200_000, res_cells.size), replace=False)

    rec_steps = set(range(0, 72 * 6 + 1, 6)) | set(range(72 * 6, STEPS + 1, 36))
    tracks, sightings = [], []

    def end(i, code, t, ex, ey):
        active[i] = False
        outcome[i] = code
        t_end[i] = t
        end_x[i], end_y[i] = ex, ey

    for s in range(STEPS):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        t_h = s * C.STEP_MIN / 60.0
        m = idx.size
        g = grp[idx]
        cat = is_cat[idx]
        hour = (cases["hour"][idx] + t_h) % 24
        hunger = min(1.0, t_h / 72.0)
        circ = np.where(cat, 0.35 + 0.65 * np.maximum(_gauss_hour(hour, 6, 1.5), _gauss_hour(hour, 21, 2.5)),
                        np.where(xeno_dog[idx], 0.3 + 0.7 * _gauss_hour(hour, 2, 4), 0.3 + 0.7 * _gauss_hour(hour, 13, 5)))
        temp = temp_mean[idx] + 5 * np.cos(2 * np.pi * (hour - 15) / 24)
        heat = np.where(temp > 28, 0.4, 1.0)
        daylight = np.where((hour >= 7) & (hour < 20), 1.0, 0.3)
        cx, cy = x[idx], y[idx]
        d_target = np.hypot(cx - tx[idx], cy - ty[idx])

        # 1. behavioural state
        st = state[idx]
        u = rng.random(m)
        fleeing = st == FLEE
        st = np.where(fleeing & (u < 0.25), np.where(cat, REST, EXPLORE), st)
        resample = ~fleeing & (rng.random(m) < 0.25)
        base_act = np.choose(g, [0.08 + 0.3 * hunger, 0.35 + 0.15 * hunger, 0.55 + 0.1 * hunger])
        moving = rng.random(m) < np.clip(base_act * circ * heat, 0, 0.95)
        w_home = homing_ab[idx] * (d_target < homing_r[idx]) * 2.0
        w_for = 0.2 + 1.5 * hunger
        tot = 1.0 + w_for + w_home
        r2 = rng.random(m) * tot
        chosen = np.where(r2 < 1.0, EXPLORE, np.where(r2 < 1.0 + w_for, FORAGE, HOMING))
        st = np.where(resample, np.where(moving, chosen, REST), st)
        state[idx] = st

        # 2. movement
        mv = np.flatnonzero(st != REST)
        crossed = np.zeros(m, bool)
        if mv.size:
            gi = idx[mv]
            sm = st[mv]
            L = speed[gi, sm] * STEP_S * rng.gamma(2.0, 0.5, mv.size)
            base = heading[gi].copy()
            hom = sm == HOMING
            base[hom] = np.arctan2(ty[gi[hom]] - cy[mv[hom]], tx[gi[hom]] - cx[mv[hom]])
            ang = base[:, None] + rng.vonmises(0.0, np.broadcast_to(kappa[sm][:, None], (mv.size, K)))
            ang[:, 7:] = rng.uniform(0, 2 * np.pi, (mv.size, K - 7))
            px, py = cx[mv][:, None], cy[mv][:, None]
            ex = px + L[:, None] * np.cos(ang)
            ey = py + L[:, None] * np.sin(ang)
            inside = (ex >= 0) & (ey >= 0) & (ex < W * cell) & (ey < H * cell)
            rr = np.clip((ey // cell).astype(int), 0, H - 1)
            cc = np.clip((ex // cell).astype(int), 0, W - 1)
            f = F[:, rr, cc]                                            # (7, mv, K)
            wv = wts[gi]
            hunger_food = (0.3 + hunger) * np.where(sm == FORAGE, 2.0, 1.0)
            score = (wv[:, 0, None] * f[0] + wv[:, 1, None] * hunger_food[:, None] * f[1]
                     + wv[:, 2, None] * f[2] + wv[:, 3, None] * f[3] + wv[:, 4, None] * f[4]
                     + wv[:, 5, None] * f[5] + wv[:, 6, None] * f[6])
            cross = major[rr, cc] > 0
            wet = water[rr, cc] > 0
            for frac in (1 / 3, 2 / 3):
                mr = np.clip(((py + (ey - py) * frac) // cell).astype(int), 0, H - 1)
                mc = np.clip(((px + (ex - px) * frac) // cell).astype(int), 0, W - 1)
                cross |= major[mr, mc] > 0
                wet |= water[mr, mc] > 0
            score += wv[:, 7, None] * cross + wv[:, 8, None] * exposed[rr, cc]
            fl = sm == FLEE
            if fl.any():
                d_now = np.hypot(px[fl] - ox[gi[fl], None], py[fl] - oy[gi[fl], None])
                d_new = np.hypot(ex[fl] - ox[gi[fl], None], ey[fl] - oy[gi[fl], None])
                score[fl] += 3.0 * (d_new - d_now) / L[fl, None]
            if hom.any():
                d_now = np.hypot(px[hom] - tx[gi[hom], None], py[hom] - ty[gi[hom], None])
                d_new = np.hypot(ex[hom] - tx[gi[hom], None], ey[hom] - ty[gi[hom], None])
                score[hom] += 3.0 * (d_now - d_new) / L[hom, None]
            score[~inside | wet] = -np.inf
            ok = np.isfinite(score).any(axis=1)
            score[~ok] = 0.0
            p = np.exp(score - score.max(axis=1, keepdims=True))
            p[~np.isfinite(p)] = 0
            p /= p.sum(axis=1, keepdims=True)
            pick = (p.cumsum(axis=1) < rng.random((mv.size, 1))).sum(axis=1).clip(0, K - 1)
            ar = np.arange(mv.size)
            nx = np.where(ok, ex[ar, pick], px[:, 0])
            ny = np.where(ok, ey[ar, pick], py[:, 0])
            x[gi], y[gi] = nx, ny
            heading[gi] = np.where(ok, ang[ar, pick], heading[gi] + np.pi)
            path_len[gi] += np.where(ok, L, 0)
            crossed[mv] = ok & cross[ar, pick]
            cx, cy = x[idx], y[idx]

        d_origin = np.hypot(cx - ox[idx], cy - oy[idx])
        max_d[idx] = np.maximum(max_d[idx], d_origin)

        if record and s in rec_steps:
            tracks.append(np.column_stack([idx, np.full(m, t_h), cx, cy, st]))

        # 3. death
        night = np.where(daylight < 1, 0.5, 1.0)
        dead = (crossed & (rng.random(m) < road_p[idx] * night)) | (rng.random(m) < bg_p[idx])
        if dead.any():
            di = idx[dead]
            # Road deaths are usually seen; animals that die hidden rarely are.
            body = rng.random(di.size) < np.where(crossed[dead], 1.6 * C.BODY_FOUND_P, 0.2 * C.BODY_FOUND_P)
            delay = rng.exponential(24.0, di.size)
            found_body = body & (t_h + delay < C.MAX_DAYS * 24)
            end(di[found_body], O["found_dead"], t_h + delay[found_body], cx[dead][found_body], cy[dead][found_body])
            lost_body = di[~found_body]
            died_unfound[lost_body] = True
            end(lost_body, O["not_found"], C.MAX_DAYS * 24, np.nan, np.nan)
        alive = ~dead

        # 4. recovery hazards (competing risks)
        r = np.clip((cy // cell).astype(int), 0, H - 1)
        c = np.clip((cx // cell).astype(int), 0, W - 1)
        ppl = people_l[r, c]
        vis = np.choose(st, [vis_rest[idx], 1.0, 1.0, 0.5, 0.8])
        vis_own = np.choose(st, [vis_owner_rest[idx], 1.0, 1.0, 0.5, 0.8])
        aware = 1 + 2 * np.exp(-d_origin / 800) * min(1.0, t_h / 24)
        d_home = np.hypot(cx - hx[idx], cy - hy[idx])
        # Owners search hardest and posters are freshest in the first days.
        effort = dm[idx] * (floor[idx] + (1 - floor[idx]) * np.exp(-t_h / decay[idx]))
        haz = np.column_stack([
            hz["self_return"][idx] * np.exp(-d_home / 60) * (st != REST),
            hz["owner"][idx] * effort * np.exp(-d_origin / search_r[idx]) * vis_own * daylight,
            hz["public"][idx] * effort * ppl ** 0.7 * vis * daylight * approach[idx] * aware * tag[idx],
            hz["picked_up"][idx] * effort * ppl ** 0.5 * vis * daylight * tag[idx],
            hz["taken_in"][idx] * reloc_m[idx] * effort * approach[idx] * ppl * daylight * (1 - 0.5 * cases["collar_tag"][idx]),
        ])
        haz *= alive[:, None]
        total = haz.sum(axis=1)
        hit = rng.random(m) < total
        if hit.any():
            hi = np.flatnonzero(hit)
            cum = haz[hi].cumsum(axis=1) / total[hi, None]
            which = (cum < rng.random((hi.size, 1))).sum(axis=1)
            for code, k in ((O["returned_home"], 0), (O["found_by_owner"], 1), (O["found_by_public"], 2),
                            (O["picked_up_vet_municipality"], 3)):
                sel = hi[which == k]
                if sel.size:
                    if k == 0:
                        end(idx[sel], code, t_h, hx[idx[sel]], hy[idx[sel]])
                    else:
                        end(idx[sel], code, t_h, cx[sel], cy[sel])
            sel = hi[which == 4]
            if sel.size:
                gi = idx[sel]
                cand = res_pool[rng.integers(0, res_pool.size, (sel.size, 30))]
                cry = (cand // W + 0.5) * cell
                crx = (cand % W + 0.5) * cell
                dd = np.hypot(crx - cx[sel, None], cry - cy[sel, None])
                good = (dd >= C.TAKEN_IN_RANGE_M["min"]) & (dd <= reloc_hi[gi, None])
                j = np.where(good.any(axis=1), good.argmax(axis=1), 0)
                ar = np.arange(sel.size)
                relocated[gi] = True
                reloc_x[gi], reloc_y[gi] = crx[ar, j], cry[ar, j]
                delay = rng.exponential(72.0, sel.size)
                rep = (rng.random(sel.size) < C.TAKEN_IN_REPORTED_P) & (t_h + delay < C.MAX_DAYS * 24)
                end(gi[rep], O["taken_in_by_stranger"], t_h + delay[rep], reloc_x[gi[rep]], reloc_y[gi[rep]])
                end(gi[~rep], O["not_found"], C.MAX_DAYS * 24, np.nan, np.nan)

        # 5. sightings (reported by the public; positions are true here, noise is added later)
        if record:
            ps = hz["sighting"][idx] * ppl ** 0.7 * vis * daylight * aware * alive
            sh = rng.random(m) < ps
            if sh.any():
                sightings.append(np.column_stack([idx[sh], np.full(sh.sum(), t_h), cx[sh], cy[sh]]))

    # Cases still active ran out the clock.
    still = np.flatnonzero(active)
    end(still, O["not_found"], C.MAX_DAYS * 24, np.nan, np.nan)

    return dict(outcome=outcome, t_end=t_end, end_x=end_x, end_y=end_y, path_len=path_len, max_d=max_d,
                relocated=relocated, reloc_x=reloc_x, reloc_y=reloc_y, died_unfound=died_unfound,
                tracks=np.concatenate(tracks) if tracks else np.empty((0, 5)),
                sightings=np.concatenate(sightings) if sightings else np.empty((0, 4)))


def summarize(cases: dict, res: dict) -> dict:
    """Statistics compared against the published targets."""
    grp = cases["grp"]
    found = ~np.isnan(res["end_x"])
    dist = np.hypot(res["end_x"] - cases["origin_x"], res["end_y"] - cases["origin_y"])
    alive_found = found & (res["outcome"] != O["found_dead"])
    out = {}
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        fm = m & found
        d = dist[fm]
        out[g] = {
            "n": int(m.sum()),
            "found": float(fm.sum() / max(m.sum(), 1)),
            "found_alive": float((m & alive_found).sum() / max(m.sum(), 1)),
            "found_alive_7d": float((m & alive_found & (res["t_end"] <= 168)).sum() / max(m.sum(), 1)),
            "median_days": float(np.median(res["t_end"][m & alive_found]) / 24) if (m & alive_found).any() else np.nan,
            "q50_m": float(np.quantile(d, 0.5)) if d.size else np.nan,
            "q75_m": float(np.quantile(d, 0.75)) if d.size else np.nan,
            "q77_m": float(np.quantile(d, 0.77)) if d.size else np.nan,
            "within_500m": float((d <= 500).mean()) if d.size else np.nan,
            "within_1609m": float((d <= 1609).mean()) if d.size else np.nan,
            "beyond_8km": float((d > 8047).mean()) if d.size else np.nan,
        }
    cat = grp < 2
    fc = cat & found
    out["cat"] = {
        "found_alive_7d": float((cat & alive_found & (res["t_end"] <= 168)).sum() / max(cat.sum(), 1)),
        "found": float(fc.sum() / max(cat.sum(), 1)),
        "within_500m": float((dist[fc] <= 500).mean()) if fc.any() else np.nan,
    }
    return out
