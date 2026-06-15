import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 11
path = 'data/deceptive-opinion.csv'

# Load the dataset and keep only the needed columns
data = pd.read_csv(path)

#print(f"{len(data)} rows with column names: {list(data.columns)}")

clean_data = data[["text", "deceptive", "polarity"]].copy()

#print(f"{len(data)} rows with new column names: {list(clean_data.columns)}")

# Map deceptive labels to 0/1 values
label_map = {"truthful": 0, "deceptive": 1}
clean_data["label"] = clean_data["deceptive"].map(label_map)

#print(f"  Truthful: {(clean_data['label'] == 0).sum()}")
#print(f"  Deceptive: {(clean_data['label'] == 1).sum()}")

#Stratified 80/20 train-test split
train_data, test_data = train_test_split(
clean_data,test_size=0.2,
    stratify=clean_data["label"],
    random_state=SEED,
)

print(f"\nTrain set: (truthful: {(train_data['label']==0).sum()}, "
      f"deceptive: {(train_data['label']==1).sum()})")
print(f"Test set: (truthful: {(test_data['label']==0).sum()}, " 
      f"deceptive: {(test_data['label']==1).sum()})")


# Save the train and test datasets

# Train set doesn't need polarity: BERT only trained on text and label
train_data[["text", "label"]].to_csv("data/train_set.csv", index=False)

# Test set keeps polarity for analysis later
test_data.to_csv("data/test_set.csv", index=False)