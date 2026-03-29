#!/usr/bin/env python
# make_segment_map_from_coords.py  (auto-detect lines or coords)
import os, argparse, yaml, pandas as pd, folium, ast, json
from statistics import mean

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

def _parse_themes(x):
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x or "").strip()
    if not s: return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t).strip() for t in v if str(t).strip()]
    except Exception:
        pass
    return [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]

def load_subjective(base, pid):
    yml = os.path.join(base, pid, f"segments_{pid}.yaml")
    if os.path.exists(yml):
        with open(yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rows = []
        for seg in data.get("segments", []):
            seg_id = seg.get("segment_id")
            if seg_id is None: continue
            rating = seg.get("walkability_rating", seg.get("rating"))
            try: rating = float(rating) if rating not in (None, "") else None
            except: rating = None
            themes = _parse_themes(seg.get("themes", seg.get("selected_themes", [])))
            fb = (seg.get("feedback","") or "").strip()
            rows.append({"segment_id": int(seg_id), "rating": rating, "themes": themes, "feedback": fb})
        return pd.DataFrame(rows)
    # fallback CSV
    csvp = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    if not os.path.exists(csvp):
        raise FileNotFoundError(f"Subjective data not found: {yml} or {csvp}")
    df = pd.read_csv(csvp)
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64").dropna().astype(int)
    rc = "walkability_rating" if "walkability_rating" in df.columns else "rating"
    df["rating"] = pd.to_numeric(df[rc], errors="coerce")
    df["themes"] = df.get("themes","").apply(_parse_themes)
    df["feedback"] = df.get("feedback","").fillna("").astype(str).str.strip()
    return df[["segment_id","rating","themes","feedback"]].copy()

def color_for_rating(r):
    try: t = max(0.0, min(1.0, (float(r)-1.0)/4.0))
    except: t = 0.5
    r0,g0,b0 = 220,50,50; r1,g1,b1 = 40,170,60
    rc = int(r0 + t*(r1-r0)); gc = int(g0 + t*(g1-g0)); bc = int(b0 + t*(b1-b0))
    return f"#{rc:02x}{gc:02x}{bc:02x}"

def load_geometry(base, pid):
    """
    Returns (mode, df)
      mode='lines': df columns -> segment_id:int, latlngs:list[(lat,lon)]
      mode='coords': df columns -> segment_id:int, start_lat, start_lon, end_lat, end_lon
    """
    lines_path = os.path.join(base, pid, "spatial", f"{pid}_segment_lines.geojson")
    if os.path.exists(lines_path):
        with open(lines_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features", [])
        rows = []
        for ft in feats:
            if ft.get("geometry", {}).get("type") != "LineString": continue
            coords = ft["geometry"].get("coordinates") or []
            if len(coords) < 2: continue
            latlngs = [(c[1], c[0]) for c in coords]
            seg_id = ft.get("properties", {}).get("segment_id")
            if seg_id is None: continue
            rows.append({"segment_id": int(seg_id), "latlngs": latlngs})
        if not rows:
            raise ValueError(f"{lines_path} has no valid LineStrings.")
        return "lines", pd.DataFrame(rows)

    # fallback to coords CSV
    csv_path = os.path.join(base, pid, "spatial", f"{pid}_segments_coords.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing both: {lines_path} and {csv_path}")
    df = pd.read_csv(csv_path)
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64").dropna().astype(int)
    for c in ["start_lat","start_lon","end_lat","end_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start_lat","start_lon","end_lat","end_lon"])
    return "coords", df

def main():
    ap = argparse.ArgumentParser(description="Map segments colored by rating (uses LineStrings if present, else coords CSV).")
    ap.add_argument("--participant_id", required=True)
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--zoom", type=int, default=16)
    args = ap.parse_args()
    pid = args.participant_id

    mode, geom = load_geometry(args.base_dir, pid)
    subj = load_subjective(args.base_dir, pid)

    if mode == "lines":
        # center
        all_pts = [pt for row in geom["latlngs"] for pt in row]
        latc = mean([p[0] for p in all_pts]); lonc = mean([p[1] for p in all_pts])
        m = folium.Map(location=[latc, lonc], zoom_start=args.zoom, control_scale=True, tiles="OpenStreetMap")

        df = geom.merge(subj, on="segment_id", how="left")
        for _, row in df.sort_values("segment_id").iterrows():
            rating = row.get("rating", None)
            rating_str = f"{float(rating):.1f}" if pd.notna(rating) else "NA"
            color = color_for_rating(rating)
            themes_val = row.get("themes", [])
            themes_str = ", ".join(map(str, themes_val)) if isinstance(themes_val, list) else (str(themes_val) if pd.notna(themes_val) else "—")
            fb = str(row.get("feedback","") or "").strip().replace("\n"," ")
            if len(fb) > 500: fb = fb[:497]+"..."
            html = f"""
            <div>
              <b>Segment {int(row['segment_id'])}</b><br>
              Rating: <b>{rating_str}</b><br>
              Themes: {themes_str or "—"}<br>
              <div style="margin-top:4px;"><i>{fb or "No feedback"}</i></div>
            </div>
            """
            folium.PolyLine(row["latlngs"], color=color, weight=7, opacity=0.9,
                            tooltip=f"Seg {int(row['segment_id'])} • rating {rating_str}",
                            popup=folium.Popup(html, max_width=420)).add_to(m)
            # endpoints
            s, e = row["latlngs"][0], row["latlngs"][-1]
            for (la, lo), lab in zip((s, e), (f"S{int(row['segment_id'])}", f"E{int(row['segment_id'])}")):
                folium.CircleMarker([la, lo], radius=4, color="#2b6cb0", fill=True, fill_opacity=0.9, tooltip=lab).add_to(m)
    else:
        latc = mean(geom["start_lat"].tolist() + geom["end_lat"].tolist())
        lonc = mean(geom["start_lon"].tolist() + geom["end_lon"].tolist())
        m = folium.Map(location=[latc, lonc], zoom_start=args.zoom, control_scale=True, tiles="OpenStreetMap")

        df = geom.merge(subj, on="segment_id", how="left")
        for _, row in df.sort_values("segment_id").iterrows():
            latlngs = [(row["start_lat"], row["start_lon"]), (row["end_lat"], row["end_lon"])]
            rating = row.get("rating", None)
            rating_str = f"{float(rating):.1f}" if pd.notna(rating) else "NA"
            color = color_for_rating(rating)
            themes_val = row.get("themes", [])
            themes_str = ", ".join(map(str, themes_val)) if isinstance(themes_val, list) else (str(themes_val) if pd.notna(themes_val) else "—")
            fb = str(row.get("feedback","") or "").strip().replace("\n"," ")
            if len(fb) > 500: fb = fb[:497]+"..."
            html = f"""
            <div>
              <b>Segment {int(row['segment_id'])}</b><br>
              Rating: <b>{rating_str}</b><br>
              Themes: {themes_str or "—"}<br>
              <div style="margin-top:4px;"><i>{fb or "No feedback"}</i></div>
            </div>
            """
            folium.PolyLine(latlngs, color=color, weight=7, opacity=0.9,
                            tooltip=f"Seg {int(row['segment_id'])} • rating {rating_str}",
                            popup=folium.Popup(html, max_width=420)).add_to(m)
            for (la, lo), lab in zip(latlngs, (f"S{int(row['segment_id'])}", f"E{int(row['segment_id'])}")):
                folium.CircleMarker([la, lo], radius=4, color="#2b6cb0", fill=True, fill_opacity=0.9, tooltip=lab).add_to(m)

    out_dir = os.path.join(args.base_dir, pid, "spatial"); os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"{pid}_segments_map.html")
    m.save(out_html)
    print(f"✅ Saved: {out_html}")

if __name__ == "__main__":
    main()
