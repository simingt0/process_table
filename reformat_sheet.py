import pandas as pd
import numpy as np

input_path  = "benchmark.xlsx"
output_path = "benchmark_reformatted.xlsx"

xls = pd.ExcelFile(input_path)
sheets = pd.read_excel(xls, sheet_name=None)
for name, df in sheets.items():
    if name == "Notes": continue
    mask = df['Numeric outcome'].astype(str).str.contains(r'±', na=False)
    if mask.any():
        ser = df.loc[mask, 'Numeric outcome'].astype(str)
        split_cols = ser.str.split(r'\s*±\s*', n=1, expand=True)
        df.loc[mask, 'Value'] = split_cols.iloc[:, 0]
        df.loc[mask, 'Variation'] = split_cols.iloc[:, 1]
        df.loc[~mask, 'Value'] = df.loc[~mask, 'Numeric outcome']
        df.loc[~mask, 'Variation'] = np.nan
    else:
        df = df.rename(columns={'Numeric outcome': 'Value'})
        df["Variation"] = np.nan

    new_order = ["PMID", "Animal Species", "Sample Size", "Treatment 1", "Dose 1", "Unit 1", "Treatment 2", "Dose 2", "Unit 2", "Terminal time point (the time point at which an animal is euthanized)", "Toxicity biomarker/phenotype", "Value", "Units", "Variation", "Note"]
    sheets[name] = df[new_order]

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)