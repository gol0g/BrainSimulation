import pandas as pd
cls = pd.read_csv("data/flywire/classification.csv.gz")
for col in ["cell_type","hemibrain_type","class","sub_class","super_class"]:
    s = cls[col].fillna("").astype(str)
    n_mbon = s.str.startswith("MBON").sum()
    n_dan  = s.str.match(r"^(PAM|PPL1)").sum()
    n_kc   = s.str.startswith("KC").sum()
    print("%-16s 비어있지않음 %6d | KC %5d | MBON %4d | DAN %4d"
          % (col, (s != "").sum(), n_kc, n_mbon, n_dan))
print("\n=== class 고유값(상위) ===")
print(cls["class"].value_counts().head(12).to_string())
