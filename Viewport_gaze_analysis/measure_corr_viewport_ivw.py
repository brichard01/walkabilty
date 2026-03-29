import pandas as pd
from pathlib import Path
import yaml
from scipy.stats import pearsonr

# -------------------------
# Paramètres
# -------------------------
base_dir = Path(r"D:\Abderrahim Internship Data\Data")
participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]
output_file = base_dir / "corr_walkability_pv_ivw_all.csv"

results = []

# -------------------------
# Boucle sur les participants
# -------------------------
for participant_id in participants:
    pv_ivw_csv = base_dir / participant_id / f"pv_ivw_{participant_id}.csv"
    yaml_file = base_dir / participant_id / f"new_segments_{participant_id}.yaml"

    if not pv_ivw_csv.exists() or not yaml_file.exists():
        print(f"Fichiers manquants pour {participant_id}, skipping...")
        continue

    # Charger PV-IVW
    df_pv = pd.read_csv(pv_ivw_csv)

    # Charger walkability depuis YAML
    with open(yaml_file, "r") as f:
        segments = yaml.safe_load(f)["segments"]
    df_walk = pd.DataFrame(segments)[["segment_id", "walkability_rating"]]

    # Fusionner PV-IVW et walkability
    df_merged = pd.merge(df_pv, df_walk, on="segment_id")
    df_merged["participant_id"] = participant_id 

    # Ajouter aux données globales
    results.append(df_merged)

# -------------------------
# Concatenation
# -------------------------
df_all = pd.concat(results, ignore_index=True)
df_all_clean = df_all.dropna(subset=["PV_IVW_M1", "PV_IVW_M2", "walkability_rating"])

# -------------------------
# Corrélation globale avec p-value
# -------------------------
r_m1, p_m1 = pearsonr(df_all_clean["walkability_rating"], df_all_clean["PV_IVW_M1"])
r_m2, p_m2 = pearsonr(df_all_clean["walkability_rating"], df_all_clean["PV_IVW_M2"])

print(f"PV_IVW_M1 vs Walkability : r = {r_m1:.3f}, p = {p_m1:.3f}")
print(f"PV_IVW_M2 vs Walkability : r = {r_m2:.3f}, p = {p_m2:.3f}")