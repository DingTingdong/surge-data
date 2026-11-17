"""Build versioned public market-data objects from a local surge-data checkout."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

MARKETS = ("TWSE", "TPEX")
FLOOR = {"TWSE": 700, "TPEX": 450}
WINDOW = 100
PRICE = "date,market,code,name,open,high,low,close,volume,amount,chg"
ACTION = "date,market,code,kind,before,ref,ratio"
CHIP = "date,market,code,foreign,trust,dealer,inst,margin,short"
NOTICE = "market,code,name,announced,start,end,measure"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def records(path, header):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != header.split(","):
            raise ValueError(f"unexpected CSV schema: {path.name}")
        yield from reader


def num(value):
    if value is None or not value.strip():
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite source number")
    return int(number) if number.is_integer() else number


def iso(value):
    return date.fromisoformat(value).isoformat()


def validate_quote(p):
    if any(v is not None and v < 0 for v in p):
        raise ValueError("negative quote")
    o, h, lo, c, vol, amt = p
    if None not in (o, h, lo, c) and min(o, h, lo, c) > 0:
        if h < max(o, lo, c) or lo > min(o, h, c):
            raise ValueError("inconsistent OHLC range")
        return True
    return False


def reference(previous, prices, change, actions, prior_market_date):
    close = prices[3]
    if not previous or not close or close <= 0 or previous[0] != prior_market_date or not previous[1]:
        return None
    daily = close - change if change is not None else None
    if actions:
        # Where there are multiple factors, a missing or conflicting reference,
        # or an inconsistent pre-action close, start a new comparable segment.
        pairs = {(a[1], a[2]) for a in actions}
        if len(pairs) != 1:
            return None
        before, ref = next(iter(pairs))
        if not before or not ref or ref <= 0 or abs(previous[1] - before) > .000001:
            return None
        if daily is not None and abs(daily - ref) > .000001:
            return None
        return ref
    # A valid exchange change normally implies the preceding unadjusted close.
    # A difference indicates an uncorroborated price-basis change.
    if daily is None or daily <= 0 or abs(previous[1] - daily) > .000001:
        return None
    return daily


def put_object(out, payload):
    raw = encoded(payload)
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    sha = digest(packed)
    path = out / "objects" / (sha + ".json.gz")
    if not path.exists():
        path.write_bytes(packed)
    if digest(path.read_bytes()) != sha:
        raise ValueError("object readback mismatch")
    return {"sha256": sha, "bytes": len(packed), "rawBytes": len(raw)}


def persist_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encoded(value) + b"\n")
    tmp.replace(path)


def validate_window(dates, counts, holidays, floor):
    if len(dates) != WINDOW or dates != sorted(set(dates)):
        raise ValueError("insufficient or unsorted market calendar")
    expected = date.fromisoformat(dates[0])
    for day in dates:
        target = date.fromisoformat(day)
        while expected < target:
            if expected.weekday() < 5 and expected.isoformat() not in holidays:
                raise ValueError("missing market session " + expected.isoformat())
            expected += timedelta(days=1)
        expected = target + timedelta(days=1)
    for market in MARKETS:
        typical = median(counts[d][market] for d in dates)
        for d in dates:
            if counts[d][market] < max(floor[market], .8 * typical):
                raise ValueError(f"incomplete market {market} {d}: {counts[d][market]}")


def build(data, out, commit, captured, floor=FLOOR):
    if not re.fullmatch("[a-f0-9]{40}", commit):
        raise ValueError("source commit must be immutable")
    now = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    if now.tzinfo is None:
        raise ValueError("capture timezone missing")
    taipei = now.astimezone(timezone(timedelta(hours=8)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "objects").mkdir(exist_ok=True)
    (out / "evidence").mkdir(exist_ok=True)
    pointer = out / "manifest.json"
    old = json.loads(pointer.read_bytes()) if pointer.is_file() else None
    sources = {}

    def source(path):
        sources[path.relative_to(data).as_posix()] = digest(path.read_bytes())
        return path

    holiday_file = source(data / "holidays.txt")
    holidays = {iso(t) for t in holiday_file.read_text(encoding="utf-8-sig").split() if t}
    actions = defaultdict(list)
    share_changes = defaultdict(list)
    for a in records(source(data / "adjust.csv"), ACTION):
        market, code, d, kind = a["market"], a["code"], iso(a["date"]), a["kind"]
        if market not in MARKETS:
            raise ValueError("unknown action market")
        values = [kind, num(a["before"]), num(a["ref"]), num(a["ratio"])]
        if values not in actions[market, code, d]:
            actions[market, code, d].append(values)
        if kind != "除息":
            share_changes[market, code].append(d)
    histories, previous, names = defaultdict(lambda: deque(maxlen=WINDOW)), {}, {}
    counts = {}
    recent_dates = deque(maxlen=WINDOW)
    day, prior_date, keys = None, None, set()
    price_files = sorted((data / "history").glob("*.csv.gz")) + [data / "prices.csv"]
    for path in price_files:
        for q in records(source(path), PRICE):
            d, m, code = q["date"], q["market"], q["code"]
            if m not in MARKETS or not re.fullmatch(r"[0-9]{4}", code):
                raise ValueError("unexpected source security")
            if d != day:
                if iso(d) != d or (day and d <= day):
                    raise ValueError("non-monotone CSV dates")
                prior_date, day = day, d
                recent_dates.append(d)
                counts[d] = dict.fromkeys(MARKETS, 0)
                keys = set()
            key = m, code
            if key in keys:
                raise ValueError(f"duplicate price {day} {key}")
            keys.add(key)
            counts[d][m] += 1
            p = [num(q[k]) for k in ("open", "high", "low", "close", "volume", "amount")]
            close, change = p[3], num(q["chg"])
            complete = validate_quote(p)
            prev = previous.get(key)
            ref = reference(prev, p, change, actions.get((m, code, d)), prior_date) if complete else None
            seg = prev[3] if ref else int(d.replace("-", ""))
            adjusted = prev[2] * close / ref if ref else close
            if adjusted is not None and not math.isfinite(adjusted):
                raise ValueError("bad comparable close")
            previous[key] = (d, close if complete else None, adjusted, seg)
            names[key] = q["name"]
            histories[key].append([d, *p, adjusted, seg, change])
    dates = list(recent_dates)
    validate_window(dates, counts, holidays, floor)
    as_of = dates[-1]
    if as_of > taipei.date().isoformat() or (as_of == taipei.date().isoformat() and taipei.hour < 14):
        raise ValueError("future or intraday price file")
    if old and (old["asOf"] > as_of or datetime.fromisoformat(old["capturedAt"].replace("Z", "+00:00")) > now):
        raise ValueError("source regressed")
    date_index = {d: i for i, d in enumerate(dates)}
    # Historical full months can be reused when a new month starts. This
    # avoids uploading both markets' complete rolling histories every day.
    months = sorted({d[:7] for d in dates})
    markets = {}
    for m in MARKETS:
        packages = []
        universe = sorted(k for k in histories if k[0] == m and previous[k][0] >= dates[0])
        for month in months:
            group = [d for d in dates if d.startswith(month)]
            idx = {d: i for i, d in enumerate(group)}
            securities = []
            for key in universe:
                bars = [[idx[b[0]], *b[1:]] for b in histories[key] if b[0] in idx]
                if bars:
                    securities.append([key[1], bars])
            pkg = {"v": 1, "market": m, "dates": group, "stocks": securities}
            packages.append({**put_object(out, pkg), "from": group[0], "to": group[-1], "rows": sum(len(s[1]) for s in securities)})
        directory = [[k[1], names[k], max((d for d in share_changes[k] if d <= as_of), default=None)] for k in universe]
        markets[m] = {"count": counts[as_of][m], "counts": [counts[d][m] for d in dates], "directory": put_object(out, directory), "packages": packages}
    active_actions = [[m, c, d, *a] for (m, c, d), values in sorted(actions.items()) if d in date_index for a in values]
    notices = []
    for n in records(source(data / "disposal.csv"), NOTICE):
        m, announced, start, end = n["market"], iso(n["announced"]), iso(n["start"]), iso(n["end"])
        if m not in MARKETS or start > end:
            raise ValueError("bad disposal record")
        if announced <= as_of and end >= dates[0]:
            notices.append([m, n["code"], n["name"], announced, start, end, n["measure"]])
    auxiliary = {"actions": put_object(out, active_actions), "disposals": put_object(out, notices)}
    chip_dates = set(dates[-11:])
    chips = []
    keys = set()
    for year in sorted({d[:4] for d in chip_dates}):
        # The producer switched the active year to appendable plain CSV.
        # During rollover or migration it can coexist with the old archive.
        # Choose exactly one file, and bind the actual path/hash in sources.
        current = data / "chips" / (year + ".csv")
        path = current if current.is_file() else data / "chips" / (year + ".csv.gz")
        if not path.exists():
            continue
        for c in records(source(path), CHIP):
            d, m, code = c["date"], c["market"], c["code"]
            if d not in chip_dates:
                continue
            if m not in MARKETS or (d, m, code) in keys:
                raise ValueError("invalid or duplicate chips")
            keys.add((d, m, code))
            chips.append([d, m, code, *[num(c[k]) for k in ("foreign", "trust", "dealer", "inst", "margin", "short")]])
    auxiliary["chips"] = put_object(out, chips)
    chip_through = {m: max((r[0] for r in chips if r[1] == m), default=None) for m in MARKETS}
    manifest = {"schema": "public-market-data/v1", "sourceCommit": commit, "asOf": as_of, "capturedAt": captured,
                "dates": dates, "sources": sources, "markets": markets, "auxiliary": auxiliary, "chipsThrough": chip_through}
    unchanged = old and all(manifest[k] == old[k] for k in ("schema", "asOf", "dates", "sources", "markets", "auxiliary", "chipsThrough"))
    if unchanged:
        return old
    evidence = out / "evidence" / (as_of + ".json")
    if not evidence.is_file():
        persist_json(evidence, manifest)
    persist_json(pointer, manifest)
    return manifest


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--data-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--source-commit", required=True)
    cli.add_argument("--captured-at", default=datetime.now(timezone.utc).isoformat())
    args = cli.parse_args()
    manifest = build(args.data_dir, args.output, args.source_commit, args.captured_at)
    print(json.dumps({"date": manifest["asOf"], "source": manifest["sourceCommit"], "counts": {m: manifest["markets"][m]["count"] for m in MARKETS},
                      "chipsThrough": manifest["chipsThrough"], "chipFiles": [s for s in manifest["sources"] if s.startswith("chips/")]}))
