#!/usr/bin/env python
import os
import argparse
import yaml
import pandas as pd

def main():
    # 1) Parse CLI
    parser = argparse.ArgumentParser(
        description="Parse segments_<PID>.yaml into a flat CSV of subjective data."
    )
    parser.add_argument(
        "--participant_id","-p",
        required=True,
        help="Participant folder name, e.g. P01, P02, etc."
    )
    args = parser.parse_args()
    PID = args.participant_id

    # 2) Build paths
    # assumes this script lives in C:/.../scripts/
    root_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    part_dir    = os.path.join(root_dir, PID)
    yaml_path   = os.path.join(part_dir, f"segments_{PID}.yaml")
    subj_dir    = os.path.join(part_dir, "subjective")
    os.makedirs(subj_dir, exist_ok=True)
    output_csv  = os.path.join(subj_dir, f"{PID}_subjective.csv")

    # 3) Load YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 4) Flatten into records
    records = []
    for seg in cfg.get("segments", []):
        records.append({
            "segment_id"        : seg["segment_id"],
            "walkability_rating": seg["walkability_rating"],
            "feedback"          : seg["feedback"].strip(),
            # keep actual Python list so pandas writes it as "['a','b',...]"
            "themes"            : seg["themes"]
        })

    df = pd.DataFrame(records)

    # 5) Save
    df.to_csv(output_csv, index=False)
    print(f"✅ Parsed {len(df)} segments → {output_csv}")

if __name__=="__main__":
    main()
