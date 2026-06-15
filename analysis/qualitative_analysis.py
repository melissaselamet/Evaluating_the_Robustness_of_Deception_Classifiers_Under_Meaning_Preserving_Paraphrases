import pandas as pd

# load results
df = pd.read_csv("../data/results.csv")

## add flip columns for all four conditions
df["flip_ns_8b"]  = (df["pred_original"] != df["pred_non_strategic_8b"]).astype(int)
df["flip_s_8b"]   = (df["pred_original"] != df["pred_strategic_8b"]).astype(int)
df["flip_ns_70b"] = (df["pred_original"] != df["pred_non_strategic_70b"]).astype(int)
df["flip_s_70b"]  = (df["pred_original"] != df["pred_strategic_70b"]).astype(int)

# label map for readable output
label_map = {0: "truthful", 1: "deceptive"}

## flip pattern counts
all_flipped      = df[(df["flip_ns_8b"] == 1) & (df["flip_s_8b"] == 1) & (df["flip_ns_70b"] == 1) & (df["flip_s_70b"] == 1)]
none_flipped     = df[(df["flip_ns_8b"] == 0) & (df["flip_s_8b"] == 0) & (df["flip_ns_70b"] == 0) & (df["flip_s_70b"] == 0)]
s_only           = df[(df["flip_s_8b"] == 1) & (df["flip_ns_8b"] == 0) & (df["flip_ns_70b"] == 0) & (df["flip_s_70b"] == 0)]
ns_only          = df[(df["flip_ns_8b"] == 1) & (df["flip_s_8b"] == 0) & (df["flip_ns_70b"] == 0) & (df["flip_s_70b"] == 0)]

print(f"flip pattern summary:")
print(f"  all four conditions flipped:  {len(all_flipped)}")
print(f"  no conditions flipped:        {len(none_flipped)}")
print(f"  8B strategic only:            {len(s_only)}")
print(f"  8B non-strategic only:        {len(ns_only)}")

## print example function
def print_example(row, case_description):
    print(f"\n{'='*70}")
    print(f"case: {case_description}")
    print(f"true label:               {label_map[row['label']]}")
    print(f"polarity:                 {row['polarity']}")
    print(f"\noriginal prediction:      {label_map[row['pred_original']]}")
    print(f"8B non-strategic:         {label_map[row['pred_non_strategic_8b']]}")
    print(f"8B strategic:             {label_map[row['pred_strategic_8b']]}")
    print(f"70B non-strategic:        {label_map[row['pred_non_strategic_70b']]}")
    print(f"70B strategic:            {label_map[row['pred_strategic_70b']]}")
    print(f"\noriginal text:\n{row['text']}")
    print(f"\n8B non-strategic paraphrase:\n{row['non_strategic_8b']}")
    print(f"\n8B strategic paraphrase:\n{row['strategic_8b']}")
    print(f"\n70B non-strategic paraphrase:\n{row['non_strategic_70b']}")
    print(f"\n70B strategic paraphrase:\n{row['strategic_70b']}")

## print one example from each case
print("\nexample 1: all four conditions flipped")
if len(all_flipped) > 0:
    print_example(all_flipped.iloc[0], "all conditions flipped")

print("\nexample 2: 8B strategic flip only")
if len(s_only) > 0:
    print_example(s_only.iloc[0], "8B strategic flip only")

print("\nexample 3: 8B non-strategic flip only")
if len(ns_only) > 0:
    print_example(ns_only.iloc[0], "8B non-strategic flip only")