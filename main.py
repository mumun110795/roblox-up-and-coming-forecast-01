#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of the chart: each game's rank and
player count, plus per-game details where the recorder provided them) and
/input/task.json, and writes one forecast to /output/forecast.json: for
each of the top 10 places, a probability for each named game being at that
place, plus a top-10 guess in order. The single forecast is scored against
every hourly chart of the 72 hours after the history ends, by log score,
so the probabilities estimate the *mix over horizons* 0..72h of each
place's occupant distribution: calibrated spread beats confident
persistence (the house persistence baseline scores -43.5 on the practice
window; this model's design target is honest turnover).

Model (v2)
----------
All dynamics are fit on the fly from whatever history the program is
given (the practice archive, or the longer final history), shrunk toward
priors measured on the kit's public archive (34 reads, 2026-10-30 to
2026-11-07). The archive measurements behind the priors:

* In log-rank space the chart mean-reverts like an Ornstein-Uhlenbeck
  process:  log rank(t+h) ~= A(h) + B(h) * log rank(t) + Normal(0, SD(h)),
  least squares over all snapshot pairs per horizon bucket:
      dt 0-12h : A=0.121, B=0.965, SD=0.240
      dt 12-24h: A=0.400, B=0.887, SD=0.434
      dt 24-48h: A=0.972, B=0.728, SD=0.651
      dt 48-96h: A=1.959, B=0.457, SD=0.847
  Short horizons are nearly persistent; by 72h the slope has halved and
  the noise is wide. A last-rank predictor beats any smoothed rank
  (MAE 0.343 vs 0.411 for a 12h-half-life EWMA at ~24h), so positions
  start from the closing rank itself, unsmoothed.

* Player-count growth is real signal on top of rank: regressing the
  ~24h-ahead log-rank residual (after the transition above) on the past
  ~24h log growth of `playing` gives slope -0.49 (growth now -> better
  rank later, beyond what the current rank already reflects). We tilt
  each game's starting log-rank by -0.40 * clipped growth.

* Rank moves have fat tails that a single Gaussian understates: games
  suffer regime shocks (an update lands, a video drops, growth stalls)
  and effectively restart from a random chart position. The simulation
  therefore uses a mixture: with probability q(h) = 0.02 + 0.0045h
  (capped at 0.35) a game's drawn position is a uniform-random chart
  position (the stationary distribution) instead of its transition draw.

* Newcomers: a share of each place's occupants over the window were not
  on the closing chart at all. Measured window-averaged shares ~4% for
  places 1-3 and ~12% for places 4-10 (rising with horizon: 0% under a
  day for the top places, up to 11%/27% at 48-96h). That share is fit on
  the fly the same way and left unnamed: the scorer splits a place's
  unnamed remainder evenly over every game on that hour's chart that we
  did not name, the best available cover for true newcomers.

* Ageing out: only games published within the last 28 days can be on the
  chart. Where the history's details give a game's `created` date, a game
  whose 28th day falls inside the scored window is removed from the
  simulation for horizons past that moment. (The practice archive has no
  details, so this only bites on recorder histories.)

Forecast construction: Monte Carlo. For each horizon on a grid over the
scored window, draw each candidate game's future log-rank from the
mixture above, sort the draws, and record who holds each of the top 10
places. Averaging occupancy over horizons and draws gives per-place
probabilities that are automatically normalised (each simulated chart
fills every place exactly once) and honest about uncertainty.

Candidates are every game on any of the last five snapshots' charts;
games missing from the closing chart start from a penalised position
(they occasionally climb back). Everything is deterministic: fixed seed,
fixed iteration order, so two runs on the same history write
byte-identical output.

Standard library only. Runs offline.
"""

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
DRAWS_PER_HORIZON = 500
HORIZON_STEP = 3.0
MIN_P = 0.0003  # rows below this are dropped; their mass joins the unnamed share

GROWTH_TILT = 0.40       # log-rank improvement per unit of log playing growth
GROWTH_CLIP = 1.2        # clip log growth to +- this before tilting
ABSENT_PENALTY = 0.9     # log-rank penalty for missing the closing chart
SHOCK_BASE = 0.02        # regime-shock probability at horizon 0
SHOCK_PER_HOUR = 0.0045  # ... growing with horizon
SHOCK_CAP = 0.35

# Prior transition anchors (horizon_hours, A, B, SD), measured on the kit
# archive as described in the module docstring.
PRIOR_ANCHORS = [
    (2.0, 0.05, 0.99, 0.15),
    (7.5, 0.121, 0.965, 0.240),
    (18.0, 0.400, 0.887, 0.434),
    (36.0, 0.972, 0.728, 0.651),
    (72.0, 1.959, 0.457, 0.847),
]
# Fit buckets for the on-the-fly transition fit: (lo_h, hi_h, anchor_h).
FIT_BUCKETS = [
    (0.0, 8.0, 4.0),
    (8.0, 16.0, 12.0),
    (16.0, 32.0, 24.0),
    (32.0, 64.0, 48.0),
    (64.0, 96.0, 80.0),
]
# Prior unseen (newcomer) shares by dt bucket, for places 1-3 and 4-10.
UNSEEN_BUCKETS = [(0.0, 12.0), (12.0, 24.0), (24.0, 48.0), (48.0, 72.0)]
UNSEEN_PRIOR_TOP = [0.000, 0.000, 0.006, 0.114]
UNSEEN_PRIOR_LOW = [0.003, 0.016, 0.075, 0.274]
UNSEEN_DEFAULT = (0.04, 0.12)  # (places 1-3, places 4-10) window averages


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_history():
    task_path = os.path.join(INPUT_DIR, "task.json")
    with open(task_path, "r", encoding="utf-8") as fh:
        task = json.load(fh)
    snapshots = []
    history_path = os.path.join(INPUT_DIR, "history.ndjson")
    with open(history_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chart = {}
            for game in row.get("up_and_coming") or []:
                chart[int(game["universe_id"])] = game
            snapshots.append(
                {
                    "at": parse_time(row["at"]),
                    "chart": chart,
                    "details": row.get("details") or {},
                }
            )
    snapshots.sort(key=lambda item: item["at"])
    return task, snapshots


def window_hours_of(task):
    window = task.get("window") or {}
    try:
        start = parse_time(window["from"])
        end = parse_time(window["until"])
        return start, max(1.0, (end - start).total_seconds() / 3600.0)
    except Exception:
        return None, 72.0


def fit_transitions(snapshots):
    """Least-squares (A, B, SD) per horizon bucket from all snapshot pairs,
    shrunk toward the archive priors by pair count. Returns anchor list."""
    # Accumulate sums per bucket: n, sx, sy, sxx, sxy, and residuals later.
    stats = [[0, 0.0, 0.0, 0.0, 0.0] for _ in FIT_BUCKETS]
    times = [snap["at"] for snap in snapshots]
    charts = [snap["chart"] for snap in snapshots]
    n_snaps = len(snapshots)
    for i in range(n_snaps):
        for j in range(i + 1, n_snaps):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt <= 0 or dt > 96.0:
                continue
            bucket = None
            for bi, (lo, hi, _) in enumerate(FIT_BUCKETS):
                if lo < dt <= hi:
                    bucket = bi
                    break
            if bucket is None:
                continue
            ci, cj = charts[i], charts[j]
            st = stats[bucket]
            for uid, game in ci.items():
                other = cj.get(uid)
                if other is None:
                    continue
                x = math.log(max(1, int(game["rank"])))
                y = math.log(max(1, int(other["rank"])))
                st[0] += 1
                st[1] += x
                st[2] += y
                st[3] += x * x
                st[4] += x * y
    anchors = []
    for bi, (_, _, anchor_h) in enumerate(FIT_BUCKETS):
        n, sx, sy, sxx, sxy = stats[bi]
        prior = prior_at(anchor_h)
        if n >= 60:
            mx = sx / n
            my = sy / n
            var_x = sxx / n - mx * mx
            if var_x > 1e-9:
                b = (sxy / n - mx * my) / var_x
                a = my - b * mx
                # Residual sd around the fitted line.
                # E[y^2] is not accumulated; approximate sd from the prior
                # by scaling with the slope miss is unreliable, so refit
                # in a second pass below only when cheap. Use prior sd
                # blended with a slope-based estimate here; the second
                # pass (fit_residual_sd) replaces it when data allows.
                sd = prior[2]
                w = n / (n + 400.0)
                a = w * a + (1 - w) * prior[0]
                b = w * b + (1 - w) * prior[1]
                sd = w * sd + (1 - w) * prior[2]
                anchors.append((anchor_h, a, clamp(b, 0.2, 1.0), clamp(sd, 0.1, 1.1)))
                continue
        anchors.append((anchor_h, prior[0], prior[1], prior[2]))
    anchors = fit_residual_sd(snapshots, anchors)
    anchors.sort()
    return anchors


def prior_at(h):
    """(A, B, SD) from the prior anchors at horizon h (interpolated)."""
    anchors = PRIOR_ANCHORS
    if h <= anchors[0][0]:
        return anchors[0][1], anchors[0][2], anchors[0][3]
    if h >= anchors[-1][0]:
        return anchors[-1][1], anchors[-1][2], anchors[-1][3]
    for i in range(len(anchors) - 1):
        h0, a0, b0, s0 = anchors[i]
        h1, a1, b1, s1 = anchors[i + 1]
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return a0 + f * (a1 - a0), b0 + f * (b1 - b0), s0 + f * (s1 - s0)
    return anchors[-1][1], anchors[-1][2], anchors[-1][3]


def interp_anchors(anchors, h):
    if h <= anchors[0][0]:
        return anchors[0][1], anchors[0][2], anchors[0][3]
    if h >= anchors[-1][0]:
        return anchors[-1][1], anchors[-1][2], anchors[-1][3]
    for i in range(len(anchors) - 1):
        h0, a0, b0, s0 = anchors[i]
        h1, a1, b1, s1 = anchors[i + 1]
        if h0 <= h <= h1:
            f = (h - h0) / (h1 - h0)
            return a0 + f * (a1 - a0), b0 + f * (b1 - b0), s0 + f * (s1 - s0)
    return anchors[-1][1], anchors[-1][2], anchors[-1][3]


def fit_residual_sd(snapshots, anchors):
    """Second pass: per-bucket residual sd around the blended (A, B) line,
    shrunk toward the prior sd. Keeps the Gaussian width honest when the
    given history churns faster or slower than the archive did."""
    sums = [[0, 0.0] for _ in FIT_BUCKETS]
    times = [snap["at"] for snap in snapshots]
    charts = [snap["chart"] for snap in snapshots]
    n_snaps = len(snapshots)
    for i in range(n_snaps):
        for j in range(i + 1, n_snaps):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt <= 0 or dt > 96.0:
                continue
            bucket = None
            for bi, (lo, hi, _) in enumerate(FIT_BUCKETS):
                if lo < dt <= hi:
                    bucket = bi
                    break
            if bucket is None:
                continue
            anchor_h = FIT_BUCKETS[bucket][2]
            line = None
            for entry in anchors:
                if entry[0] == anchor_h:
                    line = entry
                    break
            a, b = line[1], line[2]
            ci, cj = charts[i], charts[j]
            st = sums[bucket]
            for uid, game in ci.items():
                other = cj.get(uid)
                if other is None:
                    continue
                x = math.log(max(1, int(game["rank"])))
                y = math.log(max(1, int(other["rank"])))
                resid = y - (a + b * x)
                st[0] += 1
                st[1] += resid * resid
    out = []
    for bi, entry in enumerate(anchors):
        anchor_h = FIT_BUCKETS[bi][2] if bi < len(FIT_BUCKETS) else entry[0]
        n, ss = sums[bi] if bi < len(sums) else (0, 0.0)
        prior_sd = prior_at(entry[0])[2]
        if n >= 60:
            sd = math.sqrt(ss / n)
            w = n / (n + 400.0)
            sd = w * sd + (1 - w) * prior_sd
            out.append((entry[0], entry[1], entry[2], clamp(sd, 0.1, 1.15)))
        else:
            out.append(entry)
    return out


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def fit_unseen(snapshots, window_hours):
    """Window-averaged newcomer share for places 1-3 and places 4-10:
    from snapshot pairs, the share of a place's occupants that were not
    on the earlier snapshot's chart at all, per dt bucket, averaged over
    the window's horizons and shrunk toward the archive priors."""
    counts = [[0, 0] for _ in UNSEEN_BUCKETS]  # per bucket: [slots, unseen] top
    counts_low = [[0, 0] for _ in UNSEEN_BUCKETS]
    times = [snap["at"] for snap in snapshots]
    charts = [snap["chart"] for snap in snapshots]
    n_snaps = len(snapshots)
    for i in range(n_snaps):
        by_rank_i = charts[i]
        for j in range(i + 1, n_snaps):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt <= 0 or dt > 72.0:
                continue
            bucket = None
            for bi, (lo, hi) in enumerate(UNSEEN_BUCKETS):
                if lo < dt <= hi:
                    bucket = bi
                    break
            if bucket is None:
                continue
            for game in charts[j].values():
                rank = int(game["rank"])
                if rank > 10:
                    continue
                target = counts[bucket] if rank <= 3 else counts_low[bucket]
                target[0] += 1
                if game["universe_id"] not in by_rank_i:
                    target[1] += 1
    def window_average(counts, priors):
        total = 0.0
        weight_sum = 0.0
        for bi, (lo, hi) in enumerate(UNSEEN_BUCKETS):
            width = max(0.0, min(hi, window_hours) - lo)
            if width <= 0:
                continue
            n, unseen = counts[bi]
            if n >= 40:
                share = unseen / n
                w = n / (n + 200.0)
                share = w * share + (1 - w) * priors[bi]
            else:
                share = priors[bi]
            total += share * width
            weight_sum += width
        if weight_sum <= 0:
            return None
        return total / weight_sum
    top = window_average(counts, UNSEEN_PRIOR_TOP)
    low = window_average(counts_low, UNSEEN_PRIOR_LOW)
    if top is None:
        top = UNSEEN_DEFAULT[0]
    if low is None:
        low = UNSEEN_DEFAULT[1]
    return clamp(top, 0.005, 0.30), clamp(low, 0.005, 0.35)


def growth_tilts(snapshots):
    """Per-game log playing growth over the ~24h before the closing
    snapshot (nearest earlier snapshot within tolerance, else the longest
    available span up to 36h, normalised per 24h). Returns uid -> tilt
    already multiplied by GROWTH_TILT and clipped."""
    if len(snapshots) < 2:
        return {}
    last = snapshots[-1]
    last_time = last["at"]
    base = None
    base_span = None
    for snap in snapshots[:-1]:
        span = (last_time - snap["at"]).total_seconds() / 3600.0
        if span <= 0:
            continue
        if base is None or abs(span - 24.0) < abs(base_span - 24.0):
            if span <= 40.0:
                base = snap
                base_span = span
    if base is None or not base_span:
        return {}
    tilts = {}
    for uid, game in last["chart"].items():
        old = base["chart"].get(uid)
        if old is None:
            continue
        now_playing = game.get("playing")
        old_playing = old.get("playing")
        if not now_playing or not old_playing:
            continue
        growth = math.log(now_playing / old_playing) * (24.0 / base_span)
        growth = clamp(growth, -GROWTH_CLIP, GROWTH_CLIP)
        tilts[uid] = -GROWTH_TILT * growth
    return tilts


def exit_hours(snapshots, window_start):
    """Per-game hours from the window start at which it ages out of the
    chart (published 28 days), from the closing snapshot's details.
    Games without a usable `created` date are absent from the result."""
    if not snapshots or window_start is None:
        return {}
    details = snapshots[-1]["details"]
    if not details:
        return {}
    exits = {}
    for uid, game in snapshots[-1]["chart"].items():
        info = details.get(str(uid)) or details.get(uid)
        if not info:
            continue
        created = info.get("created")
        if not created:
            continue
        try:
            created_at = parse_time(created)
        except Exception:
            continue
        exit_at = created_at + timedelta(days=28)
        hours = (exit_at - window_start).total_seconds() / 3600.0
        exits[uid] = hours
    return exits


def candidate_positions(snapshots, tilts):
    """Map universe_id -> starting log-rank for the simulation.

    Closing-chart games start at their closing log-rank plus their growth
    tilt. Games seen in the four snapshots before the closing one but
    missing from it start from their last seen log-rank plus a penalty:
    they occasionally climb back, but rarely.
    """
    if not snapshots:
        return {}
    positions = {}
    last_chart = snapshots[-1]["chart"]
    for uid, game in last_chart.items():
        positions[uid] = math.log(max(1, int(game["rank"]))) + tilts.get(uid, 0.0)
    for snap in reversed(snapshots[-5:-1]):
        for uid, game in snap["chart"].items():
            if uid not in positions:
                positions[uid] = (
                    math.log(max(1, int(game["rank"]))) + ABSENT_PENALTY
                )
    return positions


def simulate(positions, exits, window_start, window_hours, places, anchors, chart_size):
    """Monte Carlo occupancy counts: counts[uid][place] over all draws."""
    rng = random.Random(SEED)
    uids = sorted(positions)
    x0 = [positions[uid] for uid in uids]
    exit_h = [exits.get(uid) for uid in uids]
    counts = {uid: [0] * (places + 1) for uid in uids}
    log_chart = math.log(max(chart_size, places + 1))
    horizons = []
    h = 1.5
    while h <= window_hours:
        horizons.append(h)
        h += HORIZON_STEP
    if not horizons:
        horizons = [1.0]
    total_draws = 0
    for h in horizons:
        a, b, sd = interp_anchors(anchors, h)
        shock_q = min(SHOCK_CAP, SHOCK_BASE + SHOCK_PER_HOUR * h)
        for _ in range(DRAWS_PER_HORIZON):
            draws = []
            for idx, x in enumerate(x0):
                ex = exit_h[idx]
                if ex is not None and h > ex:
                    continue  # aged out of the chart by this horizon
                if rng.random() < shock_q:
                    # Regime shock: restart from a uniform chart position
                    # (the stationary distribution of ranks).
                    y = math.log(1.0 + rng.random() * (chart_size - 1))
                else:
                    y = a + b * x + rng.gauss(0.0, sd)
                if y < 0.0:
                    y = 0.0
                elif y > log_chart + 0.7:
                    y = log_chart + 0.7
                draws.append((y, idx))
            draws.sort()
            for place in range(1, places + 1):
                if place <= len(draws):
                    counts[uids[draws[place - 1][1]]][place] += 1
            total_draws += 1
    return counts, total_draws


def main():
    task, snapshots = load_history()
    places = int(task.get("places", 10))
    window_start, window_hours = window_hours_of(task)

    anchors = fit_transitions(snapshots)
    unseen_top, unseen_low = fit_unseen(snapshots, window_hours)
    tilts = growth_tilts(snapshots)
    exits = exit_hours(snapshots, window_start)
    positions = candidate_positions(snapshots, tilts)
    chart_size = len(snapshots[-1]["chart"]) if snapshots else 97

    counts, total_draws = simulate(
        positions, exits, window_start, window_hours, places, anchors, chart_size
    )

    probabilities = []
    for place in range(1, places + 1):
        unseen = unseen_top if place <= 3 else unseen_low
        named_mass = 1.0 - unseen
        rows = []
        for uid in sorted(positions):
            p = counts[uid][place] / total_draws * named_mass
            if p >= MIN_P:
                rows.append((uid, p))
        rows.sort(key=lambda item: (-item[1], item[0]))
        rows = rows[:100]
        scale = 1.0
        total = sum(p for _, p in rows)
        if total > named_mass > 0:
            scale = named_mass / total
        for uid, p in rows:
            probabilities.append(
                {"universe_id": uid, "place": place, "p": round(p * scale, 6)}
            )

    expected_place = {}
    for uid in sorted(positions):
        total_p = sum(counts[uid][place] for place in range(1, places + 1)) / total_draws
        exp_place = sum(
            place * counts[uid][place] for place in range(1, places + 1)
        ) / total_draws + (1.0 - total_p) * (places + 3)
        expected_place[uid] = exp_place
    top10 = sorted(positions, key=lambda uid: (expected_place[uid], uid))[:places]
    if len(top10) < places and snapshots:
        closing = sorted(
            snapshots[-1]["chart"].items(), key=lambda item: int(item[1]["rank"])
        )
        for uid, _ in closing:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break

    probabilities.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
    output_name = task.get("output", "forecast.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"probabilities": probabilities, "top10": top10}))
        fh.write("\n")


if __name__ == "__main__":
    main()
