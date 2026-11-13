#!/usr/bin/env python3
"""Diff the Rust oracle's output against the Cyrius probe's.

Usage: compare.py RUST_OUT CYRIUS_OUT [--detail GROUP ...] [--limit N]

Both files hold lines `group<TAB>key<TAB>type<TAB>value` (see README.md). Prints one
row per group: values compared, values that differ, and the worst distance (ULPs for
f32/f64 and transition points, counts for bytes). Exits 1 if any group differs that
is not listed in ACCEPTED.
"""
import struct
import sys
from collections import OrderedDict, defaultdict

# Groups whose divergence is decided and documented, as group -> (bound, reason).
# A group within its bound is reported but does not fail the run. One past its bound
# fails like any other, so an acceptance cannot hide a new, larger difference.
# The bound is in the group's distance unit (ULPs for floats).
_ADR002 = "ADR 002: glibc's f32 libm is not correctly rounded; ranga's is"
_ADR004 = "ADR 004: bilateral rounds its weighted mean; Rust truncated it"
_ADR008 = ("ADR 008: JSON numbers decode correctly rounded; serde_json's default "
           "float parser is best effort")
ACCEPTED = {
    "oklch_to_oklab.a": (2, _ADR002),
    "oklch_to_oklab.b": (2, _ADR002),
    "color_temperature": (2, _ADR002),
    "bilateral": (1, _ADR004),
    "serde_dec.f64": (1, _ADR008),
    "serde_long.f64": (2, _ADR008),
}


def load(path):
    rows = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            g, k, t, v = parts
            rows[(g, k)] = (t, v)
    return rows


def ordered(bits, width):
    """Map a float's bit pattern onto a line where adjacent floats are 1 apart:
    positive floats keep their bits, negative ones become minus their magnitude."""
    sign = 1 << (width - 1)
    return bits if bits < sign else -(bits - sign)


def ulps(a, b, width):
    ia, ib = int(a, 16), int(b, 16)
    return abs(ordered(ia, width) - ordered(ib, width))


def distance(t, a, b):
    if t == "f32" or t == "x":
        if a in ("none", "singular") or b in ("none", "singular"):
            return None
        return ulps(a, b, 32)
    if t == "f64":
        return ulps(a, b, 64)
    if t == "u8":
        if len(a) != len(b):
            return None
        ba, bb = bytes.fromhex(a), bytes.fromhex(b)
        return max(abs(x - y) for x, y in zip(ba, bb))
    return None


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    rust, cy = load(argv[1]), load(argv[2])
    if not rust or not cy:
        # An empty side means a group name matched nothing or a build failed. Never
        # report success over zero comparisons.
        print(f"no values to compare (rust: {len(rust)}, cyrius: {len(cy)})")
        return 2
    detail, limit = set(), 5
    i = 3
    while i < len(argv):
        if argv[i] == "--detail":
            detail.add(argv[i + 1]); i += 2
        elif argv[i] == "--limit":
            limit = int(argv[i + 1]); i += 2
        else:
            i += 1

    stats = defaultdict(lambda: {"n": 0, "diff": 0, "worst": 0, "bytes": 0, "missing": 0, "examples": []})
    for key in list(rust.keys()) + [k for k in cy.keys() if k not in rust]:
        g = key[0]
        st = stats[g]
        if key not in rust or key not in cy:
            st["missing"] += 1
            st["examples"].append((key[1], rust.get(key, ("", "<absent>"))[1], cy.get(key, ("", "<absent>"))[1]))
            continue
        (t, a), (_, b) = rust[key], cy[key]
        st["n"] += 1
        if a == b:
            continue
        st["diff"] += 1
        d = distance(t, a, b)
        if t == "u8" and len(a) == len(b):
            st["bytes"] += sum(1 for x, y in zip(bytes.fromhex(a), bytes.fromhex(b)) if x != y)
        if t in ("f32", "f64"):
            fmt = "<f" if t == "f32" else "<d"
            pk = "<I" if t == "f32" else "<Q"
            fa = struct.unpack(fmt, struct.pack(pk, int(a, 16)))[0]
            fb = struct.unpack(fmt, struct.pack(pk, int(b, 16)))[0]
            if fa == fa and fb == fb:
                st["abs"] = max(st.get("abs", 0.0), abs(fa - fb))
        if d is None:
            st["incomparable"] = True
        else:
            st["worst"] = max(st["worst"], d)
        if len(st["examples"]) < limit or g in detail:
            st["examples"].append((key[1], a, b))

    failing = 0
    print(f"{'group':28} {'compared':>9} {'differ':>8} {'worst':>10}  note")
    for g in sorted(stats):
        st = stats[g]
        note = ""
        worst = "-" if st["diff"] == 0 else str(st["worst"])
        if st.get("incomparable"):
            worst += "+"  # some pairs had no distance (e.g. "none" vs a bit pattern)
        if st["diff"] and "abs" in st:
            note = (note + "; " if note else "") + f"max |diff| {st['abs']:.3g}"
        if st["missing"]:
            note = (note + "; " if note else "") + f"{st['missing']} key(s) on one side only"
        if st["bytes"]:
            note = (note + "; " if note else "") + f"{st['bytes']} byte(s) differ"
        bad = st["diff"] or st["missing"]
        accepted = (bad and g in ACCEPTED and not st["missing"] and not st.get("incomparable")
                    and st["worst"] <= ACCEPTED[g][0])
        if accepted:
            note = (note + "; " if note else "") + f"accepted (<= {ACCEPTED[g][0]}): {ACCEPTED[g][1]}"
        elif bad:
            failing += 1
            if g in ACCEPTED:
                note = (note + "; " if note else "") + f"EXCEEDS accepted bound {ACCEPTED[g][0]}"
        mark = "  " if not bad else ("~ " if accepted else "✗ ")
        print(f"{mark}{g:26} {st['n']:>9} {st['diff']:>8} {worst:>10}  {note}")
        if bad and not accepted and (g in detail or not detail):
            for k, a, b in st["examples"][: (None if g in detail else 3)]:
                ra = a if len(a) <= 40 else a[:37] + "..."
                rb = b if len(b) <= 40 else b[:37] + "..."
                print(f"      {k:20} rust={ra}  cyrius={rb}")
    print(f"\n{failing} group(s) differ from Rust without an accepted reason.")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
