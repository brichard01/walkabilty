#!/usr/bin/env python
# plot_segments_map.py
import argparse, os, yaml, json
import pandas as pd
import folium
from folium import Tooltip
from branca.colormap import LinearColormap

def load_subjective(base, pid):
    """Load subjective from CSV if present, else from segments_{pid}.yaml"""
    subj_csv = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    yml_path = os.path.join(base, pid, f"segments_{pid}.yaml")
    if os.path.exists(subj_csv):
        df = pd.read_csv(subj_csv)
        df["segment_id"] = df["segment_id"].astype(int)
        # themes may be comma string -> normalize to string for display
        df["themes_str"] = df.get("themes","").apply(lambda x: ", ".join([t.strip() for t in str(x).split(",")]) if not isinstance(x, list) else ", ".join(x))
        return df[["segment_id","walkability_rating","feedback","themes_str"]]
    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"Missing subjective: {subj_csv} or {yml_path}")
    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for seg in data.get("segments", []):
        rows.append({
            "segment_id": int(seg["segment_id"]),
            "walkability_rating": float(seg["walkability_rating"]),
            "feedback": str(seg.get("feedback","")).strip(),
            "themes_str": ", ".join(seg.get("themes", []))
        })
    return pd.DataFrame(rows)

def load_coords(base, pid):
    path = os.path.join(base, pid, "maps", f"{pid}_segments_coords.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing coords CSV: {path}")
    df = pd.read_csv(path)
    df["segment_id"] = df["segment_id"].astype(int)
    return df

def rating_color(r, cmap):
    # clamp to [1,5] domain
    r = max(1.0, min(5.0, float(r)))
    return cmap(r)

def main():
    ap = argparse.ArgumentParser(description="Make a Folium map of segments colored by walkability rating")
    ap.add_argument("--participant_id", required=True, help="e.g. P03")
    ap.add_argument("--base_dir", required=True, help="Base data dir, e.g. C:/Abderrahim Internship Data/Data")
    ap.add_argument("--center_lat", type=float, default=None, help="Optional map center lat (else auto)")
    ap.add_argument("--center_lon", type=float, default=None, help="Optional map center lon (else auto)")
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--gpx", default=None, help="Optional GPX path to overlay")
    args = ap.parse_args()

    pid = args.participant_id
    base = args.base_dir

    # Load
    subj = load_subjective(base, pid)
    coords = load_coords(base, pid)

    # Merge meta
    df = pd.merge(coords, subj, on="segment_id", how="left")

    # Determine center
    if args.center_lat is None or args.center_lon is None:
        lat_vals = pd.concat([df["lat_start"].dropna(), df["lat_end"].dropna()])
        lon_vals = pd.concat([df["lon_start"].dropna(), df["lon_end"].dropna()])
        ctr_lat = float(lat_vals.mean())
        ctr_lon = float(lon_vals.mean())
    else:
        ctr_lat, ctr_lon = args.center_lat, args.center_lon

    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=args.zoom, control_scale=True)

    # Optional GPX overlay
    if args.gpx and os.path.exists(args.gpx):
        try:
            import gpxpy
            with open(args.gpx, "r", encoding="utf-8") as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for seg in track.segments:
                    coords_gpx = [(p.latitude, p.longitude) for p in seg.points]
                    if coords_gpx:
                        folium.PolyLine(coords_gpx, color="purple", weight=3, opacity=0.7, tooltip="GPX track").add_to(m)
        except Exception as e:
            folium.map.Popup(f"GPX load failed: {e}").add_to(m)

    # Color scale for ratings 1–5
    cmap = LinearColormap(
        colors=["#d73027","#fc8d59","#fee090","#91bfdb","#1a9850"][::-1],  # low red -> high green (reversed to be: red=1, green=5)
        vmin=1.0, vmax=5.0
    ).to_step(9)
    cmap.caption = "Walkability rating (1=low … 5=high)"
    cmap.add_to(m)

    # Draw each segment as a straight line between start/end (works even without GPX)
    for _, row in df.iterrows():
        seg = int(row["segment_id"])
        lat_s, lon_s = row["lat_start"], row["lon_start"]
        lat_e, lon_e = row["lat_end"], row["lon_end"]
        rating = row.get("walkability_rating", None)
        color = rating_color(rating if pd.notnull(rating) else 3.0, cmap)

        # Tooltip / popup content
        themes = row.get("themes_str", "")
        fb = str(row.get("feedback","")).strip().replace("\n"," ")
        if len(fb) > 240: fb = fb[:237] + "…"

        html = f"""
        <div style="font-family:system-ui; font-size:13px; line-height:1.35;">
          <b>Segment {seg}</b><br/>
          Rating: <b>{rating:.1f}</b><br/>
          Themes: {themes if themes else "—"}<br/>
          <div style="margin-top:4px;"><i>“{fb}”</i></div>
        </div>
        """

        # line
        if pd.notnull(lat_s) and pd.notnull(lat_e):
            folium.PolyLine(
                [(lat_s, lon_s), (lat_e, lon_e)],
                color=color, weight=6, opacity=0.9,
                tooltip=Tooltip(f"Segment {seg} – Rating {rating:.1f}"),
                popup=folium.Popup(html, max_width=400)
            ).add_to(m)

        # markers
        if pd.notnull(lat_s):
            folium.CircleMarker(
                location=(lat_s, lon_s), radius=4, color=color, fill=True, fill_opacity=1.0,
                tooltip=f"S{seg} start", popup=folium.Popup(html, max_width=400)
            ).add_to(m)
        if pd.notnull(lat_e):
            folium.CircleMarker(
                location=(lat_e, lon_e), radius=4, color=color, fill=True, fill_opacity=1.0,
                tooltip=f"S{seg} end", popup=folium.Popup(html, max_width=400)
            ).add_to(m)

    out_dir = os.path.join(base, pid, "maps")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"{pid}_segments_colored.html")
    m.save(out_html)
    print(f"✅ Saved map: {out_html}")
    print("   • Hover for quick info; click for ratings + themes + feedback.")
    print("   • Works even without GPX (straight lines between start/end).")
    print("   • If you have GPX, pass --gpx to show the exact path as context.")

if __name__ == "__main__":
    main()
