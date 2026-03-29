#!/usr/bin/env python
import argparse, os
import pandas as pd
import dataframe_image as dfi

def main():
    ap = argparse.ArgumentParser(description="Render a CSV as a styled PNG table")
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--round", type=int, default=2, help="Decimal places to round floats")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if len(df) > args.max_rows:
        print(f"ℹ️ CSV has {len(df)} rows; truncating to first {args.max_rows} for readability.")
        df = df.head(args.max_rows)

    # try to pretty-format float columns
    float_cols = df.select_dtypes(include="number").columns
    df[float_cols] = df[float_cols].round(args.round)

    sty = (
        df.style
          .set_properties(**{"font-family":"Inter, Arial, sans-serif", "font-size":"12px"})
          .hide(axis="index")
          .set_table_styles([
              {"selector":"th", "props":[("font-weight","600"),("background","#f2f2f2")]},
              {"selector":"tbody tr:nth-child(even)", "props":[("background","rgba(0,0,0,0.03)")]},
              {"selector":"td, th", "props":[("border","1px solid #ddd"), ("padding","6px 8px")]}
          ])
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dfi.export(sty, args.out, table_conversion="matplotlib")  # works without chrome
    print(f"✅ Table saved to {args.out}")

if __name__ == "__main__":
    main()
