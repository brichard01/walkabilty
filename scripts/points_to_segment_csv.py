#!/usr/bin/env python
# points_to_segment_csv.py
import os, argparse, json, csv, re
from pathlib import Path

def read_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection.")
    feats = data.get("features", [])
    if not feats:
        raise ValueError("No features found in GeoJSON.")
    # sort by explicit 'index' if present, else keep original order
    def key_fn(ft):
        return ft.get("properties", {}).get("index", 1e12)
    feats = sorted(feats, key=key_fn)
    # keep only Point features with lon/lat
    cleaned = []
    for i, ft in enumerate(feats):
        if ft.get("geometry", {}).get("type") != "Point":
            continue
        coords = ft["geometry"].get("coordinates", None)
        if not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = ft.get("properties", {}) or {}
        label = str(props.get("label", ""))
        cleaned.append({"i": i, "lat": lat, "lon": lon, "label": label})
    return cleaned

def infer_segment_number(label, fallback_idx):
    """
    Extract the numeric part from labels like 'S1' or 'E3'. If not found, use pair index.
    """
    m = re.search(r"(\d+)$", label or "")
    if m:
        return int(m.group(1))
    return fallback_idx + 1  # 1-based

def pair_points_to_segments(points):
    """
    Pair (S, E) in order clicked: S1,E1,S2,E2,...
    Returns rows: {segment_id,start_lat,start_lon,end_lat,end_lon}
    """
    if len(points) < 2:
        return []

    if len(points) % 2 != 0:
        # Drop the trailing odd point with a warning
        print(f"⚠️  Odd number of points ({len(points)}). Dropping the last one.")
        points = points[:-1]

    rows = []
    for pair_idx in range(0, len(points), 2):
        p1 = points[pair_idx]
        p2 = points[pair_idx + 1]
        # Prefer label numbers; otherwise use pair index
        seg_id = infer_segment_number(p1.get("label"), pair_idx // 2)
        # If labels look swapped (e.g., clicked E then S), just treat first as start.
        start_lat, start_lon = p1["lat"], p1["lon"]
        end_lat, end_lon     = p2["lat"], p2["lon"]
        rows.append({
            "segment_id": seg_id,
            "start_lat":  start_lat,
            "start_lon":  start_lon,
            "end_lat":    end_lat,
            "end_lon":    end_lon
        })

    # Sort final rows by segment_id to keep them ordered 1..N
    rows.sort(key=lambda r: int(r["segment_id"]))
    return rows

def default_out_path(geojson_path: str) -> str:
    p = Path(geojson_path)
    # e.g. P03_segment_points.geojson -> P03_segments_coords.csv
    stem = p.stem.replace("_segment_points", "")  # handle your pattern
    if not stem.endswith("_segments"):
        stem = stem + "_segments"
    out_name = stem + "_coords.csv"
    return str(p.with_name(out_name))

def main():
    ap = argparse.ArgumentParser(description="Convert clicked S1,E1,S2,E2,... points GeoJSON to a segments coords CSV.")
    ap.add_argument("--geojson", required=True, help="Path to *_segment_points.geojson")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    gj = os.path.normpath(args.geojson)
    if not os.path.exists(gj):
        raise FileNotFoundError(f"Cannot find: {gj}")

    pts = read_geojson(gj)
    rows = pair_points_to_segments(pts)
    if not rows:
        raise SystemExit("No segment rows could be produced (need at least 2 points).")

    out_csv = os.path.normpath(args.out or default_out_path(gj))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id","start_lat","start_lon","end_lat","end_lon"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ Wrote {len(rows)} rows → {out_csv}")

if __name__ == "__main__":
    main()
