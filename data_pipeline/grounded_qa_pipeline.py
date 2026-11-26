# extract_bio_qna.py
"""
Creates a grounded JSONL Q&A file from a source .docx document
Uses llama-cpp-python to call a local GGUF model, enforces grounding filters:
1. Forbidden-patterns filter (no invented places/years/people).
2. Token overlap check (basic grounding).
3. Semantic similarity check (allow synonyms, block drift).
"""

import os
import re
import json
import string
import docx
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = os.getenv("GGUF_MODEL", "models/mistral-q5_k_m.gguf")
DOCX_PATH = os.getenv("SOURCE_DOCX", "data/source.docx")
OUTPUT_PATH = os.getenv("QA_OUTPUT", "data/grounded_qa.jsonl")

# thresholds
MIN_TOKEN_OVERLAP = 0.35   # relaxed since we now use embeddings
MIN_SEMANTIC_SIM = 0.65    # cosine similarity between passage and answer

# llama params
MAX_TOKENS = 256
TEMPERATURE = 0.15
N_THREADS = 8

# Forbidden patterns (years, human origins, locations, etc.)
FORBIDDEN_PATTERNS = [
    r"\b\d{3,4}\s?AD\b",
    r"\b\d{3,4}\s?BC\b",
    r"\b\d{3,4}\b",   # standalone years
    r"\bfather\b", r"\bmother\b", r"\bborn\b", r"\bvillage\b",
    r"\bIndia\b", r"\bChina\b", r"\bUSA\b", r"\bEurope\b", r"\bcountry\b",
]

# -------------------------------
# Helpers
# -------------------------------
def read_docx(file_path):
    doc = docx.Document(file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def normalize_text(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_list(s):
    return normalize_text(s).split()

def contains_forbidden(s):
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False

def token_overlap_fraction(answer, passage):
    a_tokens = [t for t in token_list(answer) if len(t) > 1]
    p_tokens = set(token_list(passage))
    if not a_tokens:
        return 0.0
    match = sum(1 for t in a_tokens if t in p_tokens)
    return match / len(a_tokens)

# -------------------------------
# Load models
# -------------------------------
print("🔹 Loading LLaMA model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=N_THREADS,
    verbose=False
)

print("🔹 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast

# -------------------------------
# Prompt template
# -------------------------------
PROMPT_TEMPLATE = """You are given a short passage. 
Task:
  1) Produce ONE meaningful question about ONLY this passage.
  2) Answer in first person as "Ramdas" by REPHRASING ONLY content that appears in the passage.
  3) Do NOT invent new facts, names, places, years, or events.
  4) If the passage lacks enough information to answer, respond with exactly:
     <s> [INST] Question? [/INST] I do not see that in the text. </s>

Format EXACTLY as:
<s> [INST] <question> [/INST] <answer> </s>

Passage:
{chunk}
"""

# -------------------------------
# Main loop
# -------------------------------
def generate_qna_from_bio():
    paras = read_docx(DOCX_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, para in enumerate(paras, start=1):
            if not para.strip():
                continue

            prompt = PROMPT_TEMPLATE.format(chunk=para)
            try:
                resp = llm(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stop=["</s>"]
                )
                raw_text = resp["choices"][0]["text"].strip()
            except Exception as e:
                print(f"[{idx}] Model error: {e}")
                raw_text = "<s> [INST] Question? [/INST] I do not see that in the text. </s>"

            # Ensure format
            if not raw_text.startswith("<s>"):
                raw_text = "<s> [INST] Question? [/INST] I do not see that in the text. </s>"

            # Extract answer part
            m = re.search(r"\[/INST\](.*)</s>", raw_text, flags=re.IGNORECASE | re.DOTALL)
            answer_part = m.group(1).strip() if m else raw_text

            # Filtering step 1: forbidden patterns
            if contains_forbidden(raw_text):
                print(f"[{idx}] ❌ Forbidden entity detected -> fallback")
                raw_text = "<s> [INST] Question? [/INST] I do not see that in the text. </s>"

            else:
                # Filtering step 2: token overlap
                overlap = token_overlap_fraction(answer_part, para)

                # Filtering step 3: semantic similarity
                emb_passage = embedder.encode(para, convert_to_tensor=True)
                emb_answer = embedder.encode(answer_part, convert_to_tensor=True)
                sim = float(util.cos_sim(emb_passage, emb_answer).item())

                if overlap < MIN_TOKEN_OVERLAP and sim < MIN_SEMANTIC_SIM:
                    print(f"[{idx}] ❌ Low grounding (overlap={overlap:.2f}, sim={sim:.2f}) -> fallback")
                    raw_text = "<s> [INST] Question? [/INST] I do not see that in the text. </s>"
                else:
                    print(f"[{idx}] ✅ Accepted (overlap={overlap:.2f}, sim={sim:.2f})")

            fout.write(json.dumps({"text": raw_text}, ensure_ascii=False) + "\n")

    print(f"\n✅ Completed. Output saved to {OUTPUT_PATH}")

# -------------------------------
if __name__ == "__main__":
    generate_qna_from_bio()
