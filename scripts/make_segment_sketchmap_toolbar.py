#!/usr/bin/env python
# make_segment_sketchmap_toolbar.py
import os, argparse, yaml
import folium
from branca.element import Element

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

def get_num_segments(base_dir: str, pid: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    yml = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    if not os.path.exists(yml):
        raise FileNotFoundError(f"Cannot find {yml}. Pass --segments N to override.")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return int(len(data.get("segments", [])))

def main():
    ap = argparse.ArgumentParser(
        description="Manual sketch mode: draw S, midpoints..., E for each segment, then Download GeoJSON (LineStrings)."
    )
    ap.add_argument("--participant_id", required=True, help="e.g. P03")
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--center", default="47.218,-1.553", help="lat,lon")
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--segments", type=int, default=None, help="override #segments if YAML not present")
    args = ap.parse_args()

    pid = args.participant_id
    base = args.base_dir
    lat, lon = map(float, args.center.split(","))

    n_segments = get_num_segments(base, pid, args.segments)
    out_geojson = f"{pid}_segment_lines.geojson"

    # Map
    m = folium.Map(
        location=[lat, lon],
        zoom_start=args.zoom,
        control_scale=True,
        tiles="OpenStreetMap",
        width="100%", height="100%"
    )
    map_div_id = m.get_name()  # e.g. "map_abcdef"

    # CSS (use placeholder to avoid f-string brace issues)
    css = """
    <style>
      html, body, #__MAPDIV__ {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
      }
      .seg-toolbar {
        position: fixed;
        top: 10px; left: 10px;
        z-index: 9999;
        background: #fff;
        border: 1px solid #aaa;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,.2);
        padding: 8px 10px;
        font: 13px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        max-width: 320px;
      }
      .seg-toolbar .title { font-weight: 700; margin-bottom: 6px; }
      .seg-toolbar button { margin: 2px 4px 0 0; }
      .seg-label {
        background: rgba(255,255,255,0.9);
        border: 1px solid #888;
        padding: 1px 4px;
        border-radius: 3px;
      }
    </style>
    """.replace("__MAPDIV__", map_div_id)
    m.get_root().header.add_child(Element(css))

    # Toolbar
    toolbar = f"""
    <div class="seg-toolbar">
      <div class="title">{pid} — Segment Sketcher</div>
      <div>Segments expected: <b>{n_segments}</b></div>
      <div id="status">Segment <b>1</b> (points: <b>0</b>) • Finished: <b>0</b></div>
      <div style="margin-top:6px;">
        <button id="btnFinish">Finish Segment</button>
        <button id="btnUndoPoint">Undo Point</button>
        <button id="btnClearCurrent">Clear Current</button>
        <button id="btnUndoSegment">Undo Segment</button>
      </div>
      <div style="margin-top:6px;">
        <button id="btnDownload">Download</button>
        <button id="btnCopy">Copy JSON</button>
        <button id="btnResetAll">Reset All</button>
      </div>
      <div style="margin-top:6px; color:#444;">
        Click to draw: <b>S</b>, then any midpoints, then <b>E</b>. Use <i>Finish Segment</i> to lock it and move on.
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(toolbar))

    # JS — wait until Folium's map variable exists (prevents blank page)
    js = """
    (function() {
      const mapVarName = "__MAPVAR__";       // e.g. "map_abcdef"
      const expectedSegs = __EXPECTED__;
      const downloadName = "__OUT__";

      function whenMapReady(cb) {
        if (window[mapVarName]) cb(window[mapVarName]);
        else setTimeout(() => whenMapReady(cb), 30);
      }

      whenMapReady(function(map) {
        // live state
        let segIndex = 1;                 // 1..expectedSegs
        let currentPts = [];              // [{lat: number, lng: number}, ...]
        let currentMarkers = [];          // Leaflet layers
        let currentLine = null;           // Leaflet polyline
        let finished = [];                // [{ seg, pts, line, markers }]

        function updateStatus() {
          const s = document.getElementById("status");
          if (!s) return;
          s.innerHTML = "Segment <b>" + segIndex + "</b> (points: <b>" + currentPts.length + "</b>) • Finished: <b>" + finished.length + "</b>";
        }

        function labelForPoint(idx) {
          if (idx === 0) return "S" + segIndex;
          return "P" + segIndex + "." + idx; // midpoints
        }

        function refreshCurrentLine() {
          if (!currentLine) {
            currentLine = L.polyline(currentPts, {color:"#5a67d8", weight:5, opacity:0.85, dashArray:"6 6"}).addTo(map);
          } else {
            currentLine.setLatLngs(currentPts);
          }
        }

        function addPoint(latlng) {
          const idx = currentPts.length;
          currentPts.push({lat: latlng.lat, lng: latlng.lng});
          const marker = L.circleMarker(latlng, {
            radius: 5, color: "#2b6cb0", fillColor:"#63b3ed", fillOpacity:0.95
          }).addTo(map);
          marker.bindTooltip("<span class='seg-label'>" + labelForPoint(idx) + "</span>", {permanent:true, direction:"right", offset:[8,0]});
          currentMarkers.push(marker);
          refreshCurrentLine();
          updateStatus();
        }

        function clearCurrent() {
          currentPts = [];
          currentMarkers.forEach(m => map.removeLayer(m));
          currentMarkers = [];
          if (currentLine) { map.removeLayer(currentLine); currentLine = null; }
          updateStatus();
        }

        function undoPoint() {
          if (currentPts.length === 0) return;
          currentPts.pop();
          const m = currentMarkers.pop();
          if (m) map.removeLayer(m);
          if (currentPts.length === 0 && currentLine) { map.removeLayer(currentLine); currentLine = null; }
          else if (currentLine) currentLine.setLatLngs(currentPts);
          updateStatus();
        }

        function finishSegment() {
          if (currentPts.length < 2) {
            alert("Need at least 2 points (S and E) to finish a segment.");
            return;
          }
          const lastMarker = currentMarkers[currentMarkers.length - 1];
          if (lastMarker) {
            lastMarker.unbindTooltip();
            lastMarker.bindTooltip("<span class='seg-label'>E" + segIndex + "</span>", {permanent:true, direction:"right", offset:[8,0]});
          }
          if (currentLine) currentLine.setStyle({dashArray:null, color:"#2f855a"}); // finished -> green

          finished.push({ seg: segIndex, pts: currentPts.slice(), line: currentLine, markers: currentMarkers.slice() });

          segIndex += 1;
          currentPts = [];
          currentMarkers = [];
          currentLine = null;
          updateStatus();

          if (segIndex > expectedSegs) {
            alert("All segments drawn (" + finished.length + "). You can still Undo Segment, or Download.");
          }
        }

        function undoSegment() {
          if (finished.length === 0) return;
          const last = finished.pop();
          if (last.line) map.removeLayer(last.line);
          (last.markers || []).forEach(m => map.removeLayer(m));
          segIndex = last.seg;
          clearCurrent();
          updateStatus();
        }

        function resetAll() {
          clearCurrent();
          finished.forEach(obj => {
            if (obj.line) map.removeLayer(obj.line);
            (obj.markers || []).forEach(m => map.removeLayer(m));
          });
          finished = [];
          segIndex = 1;
          updateStatus();
        }

        function toGeoJSON() {
          const feats = finished.map(obj => ({
            type: "Feature",
            properties: { segment_id: obj.seg, vertex_count: obj.pts.length },
            geometry: { type: "LineString", coordinates: obj.pts.map(p => [p.lng, p.lat]) }
          }));
          return { type: "FeatureCollection", features: feats };
        }

        map.on("click", function(e) {
          if (segIndex > expectedSegs) {
            alert("All segments are finished. Use Undo Segment or Reset to draw again.");
            return;
          }
          addPoint(e.latlng);
        });

        document.getElementById("btnFinish").addEventListener("click", finishSegment);
        document.getElementById("btnUndoPoint").addEventListener("click", undoPoint);
        document.getElementById("btnClearCurrent").addEventListener("click", clearCurrent);
        document.getElementById("btnUndoSegment").addEventListener("click", undoSegment);
        document.getElementById("btnResetAll").addEventListener("click", resetAll);

        document.getElementById("btnDownload").addEventListener("click", function() {
          if (finished.length < expectedSegs) {
            if (!confirm("You finished " + finished.length + "/" + expectedSegs + " segments. Download anyway?")) return;
          }
          const data = JSON.stringify(toGeoJSON(), null, 2);
          const blob = new Blob([data], {type: "application/json"});
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = downloadName;
          document.body.appendChild(a); a.click();
          setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
        });

        document.getElementById("btnCopy").addEventListener("click", function() {
          const data = JSON.stringify(toGeoJSON(), null, 2);
          navigator.clipboard.writeText(data).then(() => {
            alert("GeoJSON copied to clipboard (LineStrings).");
          });
        });

        updateStatus();
      });
    })();
    """
    js = (js
          .replace("__MAPVAR__", map_div_id)        # string key to find window[mapVarName]
          .replace("__EXPECTED__", str(n_segments))
          .replace("__OUT__", out_geojson))
    m.get_root().script.add_child(Element(js))

    out_dir = os.path.join(base, pid, "spatial")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"{pid}_segment_sketchmap.html")
    m.save(out_html)

    print(f"✅ Saved: {out_html}")
    print("Open in Chrome/Edge. For each segment:")
    print("  • Click S, add midpoints along the sidewalk, click last point as E.")
    print("  • Press 'Finish Segment' to lock it, then continue to the next.")
    print(f"Download creates: {out_geojson} (LineString per segment).")

if __name__ == "__main__":
    main()
