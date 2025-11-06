import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from scipy.optimize import minimize  # used for the optimizer

st.set_page_config(page_title="Attention-Weighted Walkability (AWV)", layout="wide")

# -------- Paths (relative to your project)
FEATURES_CSV = Path("outputs/features/segment_features_all.csv")
MODEL_MAIN   = Path("outputs/models/final_models.joblib")
MODEL_MONO   = Path("outputs/models/final_models_mono.joblib")  # optional if you train it

SUPER_CATS = ["greenery","ped_infra","people_bike","motor_traffic",
              "roadway","buildings","furniture_signage","sky_water"]

# -------- Loaders (cached)
@st.cache_data
def load_features(fp: Path):
    df = pd.read_csv(fp)
    df = df[df["rating"].notna() & (df["segment_id"] != -1)].copy()
    return df

@st.cache_resource
def load_models():
    bundle = joblib.load(MODEL_MAIN)
    models = {"Data-driven": bundle["gbm"]}
    feat_cols = bundle["features"]
    # optional monotonic model
    if MODEL_MONO.exists():
        mono_bundle = joblib.load(MODEL_MONO)
        models["Normative (monotonic)"] = mono_bundle["gbm_mono"]
    return models, feat_cols

# -------- Small helpers
def normalize_shares(d):
    s = sum(d.values())
    if s <= 1e-12:
        k0 = list(d.keys())[0]
        d = {k: 0.0 for k in d}
        d[k0] = 1.0
        return d
    return {k: v / s for k, v in d.items()}

def optimize_shares(model, base_row: pd.Series, feat_cols, budget=0.10):
    """Maximize prediction by changing only gaze SUPER_CATS within an L1 budget, with shares summing to 1."""
    x0 = np.array([float(base_row.get(sc, 0.0)) for sc in SUPER_CATS], dtype=float)
    # ensure feasible
    x0 = x0 / max(x0.sum(), 1e-9)

    # bounds [0,1]
    bounds = [(0.0, 1.0)] * len(SUPER_CATS)

    # equality: sum(x)==1
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
            # inequality: L1 change <= budget
            {"type": "ineq", "fun": lambda x: budget - np.sum(np.abs(x - x0))}]

    base_vec = base_row[feat_cols].to_numpy().astype(float)
    idx = [feat_cols.index(sc) for sc in SUPER_CATS]

    def predict_with(x):
        v = base_vec.copy()
        v[idx] = x
        return float(model.predict([v])[0])

    def obj(x):
        return -predict_with(x)  # maximize

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300})
    x_opt = x0 if not res.success else res.x
    y_new = predict_with(x_opt)
    return x_opt, y_new, res.success, res.message

# -------- Load data/models
df = load_features(FEATURES_CSV)
models, feat_cols = load_models()

# -------- Sidebar
with st.sidebar:
    st.title("AWV – Demo")
    st.caption("Features file:")
    st.code(str(FEATURES_CSV), language="text")

    pid = st.selectbox("Participant", sorted(df["participant_id"].unique()))
    dfp = df[df["participant_id"] == pid].reset_index(drop=True)
    seg = st.selectbox("Segment", sorted(dfp["segment_id"].unique()))
    row = dfp[dfp["segment_id"] == seg].iloc[0]

    model_name = st.selectbox("Model", list(models.keys()))
    model = models[model_name]

# -------- Header
st.markdown(f"### Participant **{pid}** — Segment **{seg}**  |  Model: **{model_name}**")

# -------- Top metrics
colA, colB, colC = st.columns(3)
xrow = pd.DataFrame([row[feat_cols]])
y_true = float(row["rating"])
y_pred = float(model.predict(xrow)[0])
with colA:
    st.metric("True rating", f"{y_true:.2f}")
with colB:
    st.metric("Predicted", f"{y_pred:.2f}", delta=f"{(y_pred - y_true):+.2f}")
with colC:
    if all(m in df.columns for m in ["align_l1","kl_gaze_pixel","rho_gaze_pixel"]):
        st.write("**Alignment**")
        st.write(f"L1: **{float(row['align_l1']):.2f}**  |  KL: **{float(row['kl_gaze_pixel']):.3f}**  |  ρ: **{float(row['rho_gaze_pixel']):.2f}**")
    else:
        st.caption("Alignment metrics not available.")

st.divider()

# -------- Inputs panels
left, mid, right = st.columns([1.1,1.1,1.1])

with left:
    st.subheader("👁️ Gaze (attention shares)")
    gtbl = (row[SUPER_CATS] * 100).round(1).to_frame("share %")
    st.dataframe(gtbl)

with mid:
    st.subheader("🧩 Segmentation (pixel composition)")
    pix_cols = [f"pixel_{sc}" for sc in SUPER_CATS if f"pixel_{sc}" in df.columns]
    if pix_cols:
        ptbl = (row[pix_cols] * 100).round(1).rename(index=lambda s: s.replace("pixel_", ""))
        st.dataframe(ptbl.to_frame("share %"))
    else:
        st.caption("No pixel_* columns found.")

with right:
    st.subheader("📊 IVW (proxy, from pixels)")
    ivw_cols = [c for c in df.columns if c.lower().startswith("ivw_")]
    if ivw_cols:
        ivw_tbl = (row[ivw_cols]).to_frame("value")
        st.dataframe(ivw_tbl)
    else:
        st.caption("No ivw_* columns found (proxy not merged).")

st.divider()

# -------- SHAP – waterfall (why this score)
st.subheader("Why this prediction (SHAP, per-segment)")
try:
    explainer = shap.TreeExplainer(model)
    sv_row = explainer.shap_values(xrow)
    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_row[0], xrow.iloc[0], show=False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.info(f"SHAP waterfall unavailable: {e}")

st.divider()

# -------- What-if sliders (adjust gaze shares only)
st.subheader("What-if: adjust attention shares (gaze)")
st.caption("Move sliders. Shares are re-normalized to sum to 1; other features stay fixed.")

cols = st.columns(4)
new_shares = {}
for i, sc in enumerate(SUPER_CATS):
    with cols[i % 4]:
        new_shares[sc] = st.slider(sc, 0.0, 1.0, float(row.get(sc, 0.0)), 0.01)
new_shares = normalize_shares(new_shares)

x_new = row[feat_cols].copy()
for sc, v in new_shares.items():
    x_new[sc] = v

y_pred_new = float(model.predict(pd.DataFrame([x_new]))[0])
st.metric("What-if prediction", f"{y_pred_new:.2f}", delta=f"{(y_pred_new - y_pred):+.2f}")

fig2, ax2 = plt.subplots()
ax2.bar(list(new_shares.keys()), list(new_shares.values()))
ax2.set_ylabel("Share (0–1)")
ax2.set_ylim(0, 1)
plt.xticks(rotation=30, ha="right")
st.pyplot(fig2, clear_figure=True)

st.divider()

# -------- Optimizer (suggest changes within a budget)
st.subheader("💡 Suggest improvements (optimizer)")
budget = st.slider("Max total change allowed (L1 budget)", 0.02, 0.30, 0.10, 0.01)
if st.button("Suggest changes"):
    x_opt, y_opt, ok, msg = optimize_shares(model, row, feat_cols, budget=budget)
    if not ok:
        st.warning(f"Optimizer status: {msg}")

    current = np.array([float(row.get(sc, 0.0)) for sc in SUPER_CATS])
    # ensure normalized
    current = current / max(current.sum(), 1e-9)
    delta = x_opt - current
    st.success(f"Pred {y_pred:.2f} → **{y_opt:.2f}** (Δ {y_opt - y_pred:+.2f}) within budget {budget:.2f}")

    changes = pd.DataFrame({
        "share_now": current,
        "share_suggested": x_opt,
        "delta": delta
    }, index=SUPER_CATS).round(3)
    st.dataframe(changes)

    # Download CSV for this segment
    out_row = {
        "participant_id": pid,
        "segment_id": seg,
        "true_rating": y_true,
        "pred_before": y_pred,
        "pred_after": y_opt,
        "delta_pred": y_opt - y_pred,
        "budget": budget,
    }
    for sc, v in zip(SUPER_CATS, current):
        out_row[f"gaze_{sc}_now"] = v
    for sc, v in zip(SUPER_CATS, x_opt):
        out_row[f"gaze_{sc}_suggested"] = v

    dl = pd.DataFrame([out_row]).to_csv(index=False).encode("utf-8")
    st.download_button("Download suggestion CSV", dl, file_name=f"suggestion_{pid}_{seg}.csv", mime="text/csv")
