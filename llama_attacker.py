import requests
import pandas as pd
import time
from bert_score import BERTScorer

# Configurations
ollama_url     = "http://localhost:11434/api/generate"
model_name     = "llama3.1:8b"
temperature    = 0.7
max_attempts   = 3
bert_threshold = 0.85

# Prompts
non_strategic_prompt = (
    "Rewrite the following hotel review in your own words while preserving "
    "the original meaning. Do not add or remove any information. "
    "Only return the rewritten review, nothing else."
)

strategic_prompt = (
    "Rewrite the following hotel review so that it sounds like it was genuinely "
    "written by a real hotel guest who stayed there. Use first person pronouns, "
    "include specific personal experiences and sensory details, vary your sentence "
    "length, and write in a natural conversational tone. Preserve the original "
    "meaning exactly. Only return the rewritten review, nothing else."
)

# Function to Call Ollama
def generate_paraphrase(review: str, prompt: str) -> str:
    full_prompt = f"{prompt}\n\nReview: {review}"

    response = requests.post(
        ollama_url,
        json={
            "model":  model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        print(f"error: {response.status_code} - {response.text}")
        return ""

# Load Roberta once at startup
scorer = BERTScorer(lang="en", device="mps", verbose=False)

# Function to compute bertscore between original and paraphrase
def compute_bertscore(original: str, paraphrase: str) -> float:
    _, _, F1 = scorer.score([paraphrase], [original])
    return F1.item()


# Function to generate a Paraphrase with BERTscore Quality Check
def generate_valid_paraphrase(review: str, prompt: str, condition: str):
    for attempt in range(1, max_attempts + 1):
        paraphrase = generate_paraphrase(review, prompt)
        bs = compute_bertscore(review, paraphrase)

        print(f"  [{condition}] attempt {attempt} -- bertscore: {bs:.4f}", end="")

        if bs >= bert_threshold:
            print("within treshold")
            return paraphrase, bs, attempt
        else:
            print(f"below {bert_threshold}. regenerating.")

    # If all Attempts Fail, Keep the Last One and Flag it
    print(f"  [{condition}]: all attempts below threshold, keep the last valid result.")
    return paraphrase, bs, max_attempts

# Load the test set
test_df = pd.read_csv("data/test_set.csv")
print(f"number of loaded reviews from test set: {len(test_df)}")

# Generate the Paraphrases
non_strategic_paraphrases = []
strategic_paraphrases     = []
non_strategic_bertscores  = []
strategic_bertscores      = []
non_strategic_attempts    = []
strategic_attempts        = []

for i, row in test_df.iterrows():
    review = row["text"]
    print(f"\nprocessing review {i+1}/{len(test_df)}...")

    # Non-strategic paraphrase
    ns_para, ns_bs, ns_att = generate_valid_paraphrase(review, non_strategic_prompt, "non-strategic")
    non_strategic_paraphrases.append(ns_para)
    non_strategic_bertscores.append(ns_bs)
    non_strategic_attempts.append(ns_att)

    # Strategic paraphrase
    s_para, s_bs, s_att = generate_valid_paraphrase(review, strategic_prompt, "strategic")
    strategic_paraphrases.append(s_para)
    strategic_bertscores.append(s_bs)
    strategic_attempts.append(s_att)

    # Small Delay to not Overwhelm Ollama
    time.sleep(0.5)

# Save the Results
test_df["non_strategic"] = non_strategic_paraphrases
test_df["strategic"] = strategic_paraphrases
test_df["non_strategic_bertscore"] = non_strategic_bertscores
test_df["strategic_bertscore"] = strategic_bertscores
test_df["non_strategic_attempts"] = non_strategic_attempts
test_df["strategic_attempts"] = strategic_attempts

test_df.to_csv("data/paraphrases.csv", index=False)

# quality summary
print(f"  non-strategic avg bertscore: {sum(non_strategic_bertscores)/len(non_strategic_bertscores):.4f}")
print(f"  strategic avg bertscore:     {sum(strategic_bertscores)/len(strategic_bertscores):.4f}")
print(f"  non-strategic reviews needing regeneration: {sum(a > 1 for a in non_strategic_attempts)}")
print(f"  strategic reviews needing regeneration:     {sum(a > 1 for a in strategic_attempts)}")