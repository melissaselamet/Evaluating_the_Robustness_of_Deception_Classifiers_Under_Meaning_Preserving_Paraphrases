import requests
import pandas as pd
import time
import random
import numpy as np
import os
from huggingface_hub import InferenceClient
from bert_score import BERTScorer

# Configurations
SEED = 11
ollama_url = "http://localhost:11434/api/generate"
ollama_model = "llama3.1:8b"
hf_model = "meta-llama/Llama-3.3-70B-Instruct"
max_attempts = 3
bert_threshold = 0.85
checkpoint_path = "data/paraphrases_checkpoint.csv"

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# HuggingFace client
hf_client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_TOKEN"),
)

# Prompts
non_strategic_prompt = (
    "Rewrite the following hotel review in your own words while preserving "
    "the original meaning. Do not add or remove any information. "
    "Only return the rewritten review, nothing else."
)

strategic_deceptive_prompt = (
    "This is a fake hotel review. Rewrite it so that it appears to have been "
    "genuinely written by a guest who actually stayed at the hotel. "
    "Use first-person pronouns, include specific personal experiences and sensory "
    "details such as what you saw, heard, or felt, vary sentence lengths, and write "
    "in a natural conversational tone. These changes should make the review read as "
    "a genuine first-hand account. Preserve the original meaning exactly. "
    "Only return the rewritten review, nothing else."
)

strategic_truthful_prompt = (
    "This is a genuine hotel review. Rewrite it so that it appears to have been "
    "written by someone who never stayed at the hotel. Avoid first-person pronouns, "
    "remove specific personal experiences and sensory details, use generic and "
    "impersonal language, and write in a detached and formal tone. These changes "
    "should make the review read as a fabricated account rather than a genuine one. "
    "Preserve the original meaning exactly. "
    "Only return the rewritten review, nothing else."
)


def get_strategic_prompt(label: int) -> str:
    """Select the strategic prompt based on the true label of the review.
    Label 0 = truthful, Label 1 = deceptive."""
    return strategic_deceptive_prompt if label == 1 else strategic_truthful_prompt


## LLaMA 3 8B (Ollama)
def generate_paraphrase_ollama(review: str, prompt: str) -> str:
    temperature = round(random.uniform(0.1, 1.0), 2)
    full_prompt = f"{prompt}\n\nReview: {review}"

    response = requests.post(
        ollama_url,
        json={
            "model":  ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        print(f"Ollama error: {response.status_code} - {response.text}")
        return ""


## LLaMA 3.3 70B (HuggingFace)
def generate_paraphrase_hf(review: str, prompt: str) -> str:
    temperature = round(random.uniform(0.1, 1.0), 2)
    full_prompt = f"{prompt}\n\nReview: {review}"

    response = hf_client.chat.completions.create(
        model=hf_model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


## BERTScore
scorer = BERTScorer(lang="en", device="mps")


def compute_bertscore(original: str, paraphrase: str) -> float:
    _, _, F1 = scorer.score([paraphrase], [original])
    return F1.item()


# Generate paraphrases with quality checks
def generate_valid_paraphrase(review: str, prompt: str, condition: str, use_hf: bool = False):
    for attempt in range(1, max_attempts + 1):
        if use_hf:
            paraphrase = generate_paraphrase_hf(review, prompt)
        else:
            paraphrase = generate_paraphrase_ollama(review, prompt)

        bs = compute_bertscore(review, paraphrase)
        print(f"  [{condition}] attempt {attempt} : bertscore: {bs:.4f}", end="")

        if bs >= bert_threshold:
            print(" within threshold")
            return paraphrase, bs, attempt
        else:
            print(f" below {bert_threshold}: regenerating.")

    print(f"  [{condition}]: all attempts below threshold, keeping last result.")
    return paraphrase, bs, max_attempts


# Load the test set
test_df = pd.read_csv("data/test_set.csv")

## Load or Initialize Checkpoint (incase training stops mid run)
if os.path.exists(checkpoint_path):
    checkpoint_df = pd.read_csv(checkpoint_path)
    completed_indices = set(checkpoint_df["original_index"].tolist())
    print(f"Resuming from checkpoint")
else:
    checkpoint_df = pd.DataFrame()
    completed_indices = set()
    print("Starting from scratch.")


# Generate Paraphrases
rows = []

for i, row in test_df.iterrows():

    # Skip already completed reviews
    if i in completed_indices:
        print(f"Skipping already computed reviews: {i+1}/{len(test_df)}")
        continue

    review = row["text"]
    label  = row["label"]
    strategic_prompt = get_strategic_prompt(label)

    print(f"\nreview {i+1}/{len(test_df)}...")

    result = {"original_index": i}

    try:
        # LLaMA 3 8B non-strategic
        p, bs, att = generate_valid_paraphrase(review, non_strategic_prompt, "8B non-strategic", use_hf=False)
        result["non_strategic_8b"] = p
        result["non_strategic_8b_bertscore"] = bs
        result["non_strategic_8b_attempts"] = att

        # LLaMA 3 8B strategic
        p, bs, att = generate_valid_paraphrase(review, strategic_prompt, "8B strategic", use_hf=False)
        result["strategic_8b"] = p
        result["strategic_8b_bertscore"] = bs
        result["strategic_8b_attempts"] = att

        # LLaMA 3.3 70B non-strategic
        p, bs, att = generate_valid_paraphrase(review, non_strategic_prompt, "70B non-strategic", use_hf=True)
        result["non_strategic_70b"] = p
        result["non_strategic_70b_bertscore"] = bs
        result["non_strategic_70b_attempts"] = att

        # LLaMA 3.3 70B strategic
        p, bs, att = generate_valid_paraphrase(review, strategic_prompt, "70B strategic", use_hf=True)
        result["strategic_70b"] = p
        result["strategic_70b_bertscore"] = bs
        result["strategic_70b_attempts"] = att

    except Exception as e:
        print(f"\n  Error on review {i+1}: {e}")
        print(f"Saved checkpoint until review {i}")

        if rows:
            new_rows_df = pd.DataFrame(rows)
            checkpoint_df = pd.concat([checkpoint_df, new_rows_df], ignore_index=True)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved")
        break

    rows.append(result)

    # Save checkpoint every 10 reviews
    if len(rows) % 10 == 0:
        new_rows_df = pd.DataFrame(rows)
        checkpoint_df = pd.concat([checkpoint_df, new_rows_df], ignore_index=True)
        checkpoint_df.to_csv(checkpoint_path, index=False)
        rows = []

    time.sleep(1)

# Save any remaining rows
if rows:
    new_rows_df = pd.DataFrame(rows)
    checkpoint_df = pd.concat([checkpoint_df, new_rows_df], ignore_index=True)
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"\nFinal checkpoint")

# Merge checkpoint with test set if all complete

if len(checkpoint_df) == len(test_df):
    checkpoint_df = checkpoint_df.sort_values("original_index").reset_index(drop=True)
    final_df = test_df.copy()
    final_df["non_strategic_8b"] = checkpoint_df["non_strategic_8b"].values
    final_df["non_strategic_8b_bertscore"] = checkpoint_df["non_strategic_8b_bertscore"].values
    final_df["non_strategic_8b_attempts"] = checkpoint_df["non_strategic_8b_attempts"].values
    final_df["strategic_8b"] = checkpoint_df["strategic_8b"].values
    final_df["strategic_8b_bertscore"] = checkpoint_df["strategic_8b_bertscore"].values
    final_df["strategic_8b_attempts"] = checkpoint_df["strategic_8b_attempts"].values
    final_df["non_strategic_70b"] = checkpoint_df["non_strategic_70b"].values
    final_df["non_strategic_70b_bertscore"] = checkpoint_df["non_strategic_70b_bertscore"].values
    final_df["non_strategic_70b_attempts"] = checkpoint_df["non_strategic_70b_attempts"].values
    final_df["strategic_70b"] = checkpoint_df["strategic_70b"].values
    final_df["strategic_70b_bertscore"] = checkpoint_df["strategic_70b_bertscore"].values
    final_df["strategic_70b_attempts"] = checkpoint_df["strategic_70b_attempts"].values
    final_df.to_csv("data/paraphrases.csv", index=False)

    # Quality summary
    print(f"\n Quality Summary ")
    print(f"  8B  non-strategic avg bertscore: {checkpoint_df['non_strategic_8b_bertscore'].mean():.4f}")
    print(f"  8B  strategic avg bertscore:     {checkpoint_df['strategic_8b_bertscore'].mean():.4f}")
    print(f"  70B non-strategic avg bertscore: {checkpoint_df['non_strategic_70b_bertscore'].mean():.4f}")
    print(f"  70B strategic avg bertscore:     {checkpoint_df['strategic_70b_bertscore'].mean():.4f}")
else:
    print(f"\n{len(checkpoint_df)}/{len(test_df)} reviews completed. Run again to continue.")