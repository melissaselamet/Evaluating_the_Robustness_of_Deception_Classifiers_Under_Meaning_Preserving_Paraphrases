import pandas as pd

# load results
df = pd.read_csv("../data/results.csv")

# add flip columns
df["flip_non_strategic"] = (df["pred_original"] != df["pred_non_strategic"]).astype(int)
df["flip_strategic"]     = (df["pred_original"] != df["pred_strategic"]).astype(int)

# label map for readable output
label_map = {0: "truthful", 1: "deceptive"}

# case 1: flipped in strategic
strategic_only = df[(df["flip_strategic"] == 1) & (df["flip_non_strategic"] == 0)]
print(f"reviews that only flipped in strategic: {len(strategic_only)}")

# case 2: flipped in both conditions
both_flipped = df[(df["flip_strategic"] == 1) & (df["flip_non_strategic"] == 1)]
print(f"reviews that flipped in both conditions: {len(both_flipped)}")

# case 3: flipped in non-strategic
non_strategic_only = df[(df["flip_non_strategic"] == 1) & (df["flip_strategic"] == 0)]
print(f"reviews that flipped only in non-strategic: {len(non_strategic_only)}")

# print the examples for cases
def print_example(row, case_description):
    print(f"\n{'='*70}")
    print(f"case: {case_description}")
    print(f"true label:              {label_map[row['label']]}")
    print(f"polarity:                {row['polarity']}")
    print(f"\noriginal prediction:     {label_map[row['pred_original']]} (confidence: {row['conf_original']:.3f})")
    print(f"non-strategic prediction:{label_map[row['pred_non_strategic']]} (confidence: {row['conf_non_strategic']:.3f})")
    print(f"strategic prediction:    {label_map[row['pred_strategic']]} (confidence: {row['conf_strategic']:.3f})")
    print(f"\noriginal text:\n{row['text']}")
    print(f"\nnon-strategic paraphrase:\n{row['non_strategic']}")
    print(f"\nstrategic paraphrase:\n{row['strategic']}")

# print 1 example from each case
print("example 1: flipped in strategic only")
if len(strategic_only) > 0:
    print_example(strategic_only.iloc[0], "strategic flip only")

print("example 2: flipped in both conditions")
if len(both_flipped) > 0:
    print_example(both_flipped.iloc[0], "both conditions flipped")

print("example 3: flipped in non-strategic only")
if len(non_strategic_only) > 0:
    print_example(non_strategic_only.iloc[0], "non-strategic flip only")