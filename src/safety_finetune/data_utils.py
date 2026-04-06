"""Dataset loading and formatting for KTO training.

Training data (bio/chem focused):
  - ClearHarm (AlignmentResearch/ClearHarm): CBRN prompts + harmful targets
  - LLM-LAT/harmful-dataset: refusal/harmful pairs, keyword-filtered to bio/chem
  - BeaverTails (PKU-Alignment/BeaverTails): volume filler, keyword-filtered to bio/chem
  - SlimOrca (Open-Orca/SlimOrca): safe prompts to prevent over-refusal

Eval data (never trained on):
  - WMDP (cais/wmdp): expert-written bio/chem MCQ — primary eval
  - HarmBench (walledai/HarmBench): free-form bio/chem behaviors — secondary eval
"""

import random

from datasets import Dataset, load_dataset

# Keywords to filter general safety datasets down to bio/chem domain
BIOCHEM_KEYWORDS = [
    "synth", "chemical", "biological", "pathogen", "toxin", "virus", "bacteria",
    "nerve agent", "sarin", "vx", "ricin", "anthrax", "botulinum", "cyanide",
    "mustard gas", "phosgene", "weaponiz", "bioweapon", "bio-weapon",
    "chemical weapon", "poison", "explosive", "precursor", "reagent",
    "fentanyl", "methamphetamine", "cultivation", "ferment", "culture",
    "dangerous substance", "hazardous", "lethal", "toxic", "drug",
    "bomb", "detonate", "incendiary", "radiological", "nuclear",
]

# Varied refusal templates so the model doesn't learn a single brittle pattern
REFUSAL_TEMPLATES = [
    "I'm not able to help with that. This request involves potentially dangerous "
    "information that could cause serious harm. I'd be happy to help with safe, "
    "legal alternatives or answer general science questions instead.",

    "I can't provide that information. Requests related to dangerous substances "
    "or weapons fall outside what I can assist with. Let me know if there's "
    "something else I can help you with.",

    "Sorry, I won't help with this request. It asks for information that could "
    "be used to cause harm. I'm here to help with safe and constructive topics.",

    "I'm not going to assist with that. This type of request involves serious "
    "safety concerns. If you have questions about chemistry or biology for "
    "educational purposes, I'm happy to help in a safe way.",

    "I cannot fulfill this request. Providing information on dangerous substances, "
    "weapons, or harmful activities isn't something I can do. Please ask me about "
    "something safe and I'll do my best to help.",
]


def _matches_biochem(text: str) -> bool:
    """Check if text contains bio/chem related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in BIOCHEM_KEYWORDS)


def build_kto_dataset(max_samples: int | None = None, eval_split: float = 0.2) -> dict:
    """Build a KTO dataset from multiple bio/chem safety sources.

    Returns dict with 'train' and 'eval' Dataset splits, each having columns:
        - prompt: the user message
        - completion: the assistant response
        - label: True (desirable) or False (undesirable)
    """
    examples = []

    # 1. ClearHarm: CBRN prompts with harmful targets (label=False) + refusals (label=True)
    clearharm = _load_clearharm()
    examples.extend(clearharm)

    # 2. LLM-LAT: already has refusal+harmful pairs, filtered to bio/chem
    llm_lat = _load_llm_lat_biochem()
    examples.extend(llm_lat)

    # 3. BeaverTails: keyword-filtered bio/chem for volume
    beavertails = _load_beavertails_biochem()
    examples.extend(beavertails)

    # 4. Safe examples from SlimOrca to balance
    n_unsafe = len(examples)
    safe = _load_slimorca_safe(n=n_unsafe // 2)
    examples.extend(safe)

    random.seed(42)
    random.shuffle(examples)

    if max_samples:
        examples = examples[:max_samples]

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=eval_split, seed=42)
    return {"train": split["train"], "eval": split["test"]}


def build_eval_dataset(n_safe: int = 150) -> dict:
    """Build held-out eval sets. NEVER used in training.

    - WMDP bio+chem: expert-written MCQ converted to open-ended prompts
    - HarmBench chemical_biological: free-form behaviors
    - SlimOrca: safe prompts to check for over-refusal
    """
    wmdp = _load_wmdp_eval()
    harmbench = _load_harmbench_biochem()
    safe = _load_slimorca_safe(n=n_safe, offset=10000)
    return {"wmdp": wmdp, "harmbench": harmbench, "safe": safe}


# ---------------------------------------------------------------------------
# ClearHarm: CBRN prompts
# ---------------------------------------------------------------------------

def _load_clearharm() -> list[dict]:
    """Load CBRN prompts from ClearHarm.

    Each prompt yields two examples:
      - prompt + harmful target → label=False
      - prompt + refusal → label=True
    """
    ds = load_dataset("AlignmentResearch/ClearHarm", "default", split="train")

    examples = []
    rng = random.Random(42)

    for row in ds:
        # content is a list; join into a single prompt
        prompt = " ".join(row["content"]) if isinstance(row["content"], list) else row["content"]

        if not prompt.strip():
            continue

        # Only keep bio/chem related (ClearHarm is mostly CBRN but filter to be safe)
        if not _matches_biochem(prompt):
            continue

        # Harmful completion (undesirable)
        harmful_target = row.get("proxy_gen_target", "")
        if harmful_target and harmful_target.strip():
            examples.append({
                "prompt": prompt,
                "completion": harmful_target,
                "label": False,
            })

        # Refusal (desirable)
        examples.append({
            "prompt": prompt,
            "completion": rng.choice(REFUSAL_TEMPLATES),
            "label": True,
        })

    return examples


# ---------------------------------------------------------------------------
# LLM-LAT/harmful-dataset: refusal/harmful pairs
# ---------------------------------------------------------------------------

def _load_llm_lat_biochem() -> list[dict]:
    """Load bio/chem filtered examples from LLM-LAT/harmful-dataset.

    Has 'chosen' (refusal) and 'rejected' (harmful) columns — perfect for KTO.
    """
    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")

    examples = []
    for row in ds:
        prompt = row["prompt"]
        if not _matches_biochem(prompt):
            continue

        # Harmful response (undesirable)
        if row.get("rejected"):
            examples.append({
                "prompt": prompt,
                "completion": row["rejected"],
                "label": False,
            })

        # Refusal response (desirable)
        if row.get("chosen"):
            examples.append({
                "prompt": prompt,
                "completion": row["chosen"],
                "label": True,
            })

    return examples


# ---------------------------------------------------------------------------
# BeaverTails: keyword-filtered bio/chem
# ---------------------------------------------------------------------------

def _load_beavertails_biochem(max_examples: int = 1000) -> list[dict]:
    """Load bio/chem examples from BeaverTails using keyword filtering.

    Only keeps unsafe rows whose prompt matches bio/chem keywords.
    Each yields a harmful completion (label=False) + a refusal (label=True).
    """
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

    examples = []
    rng = random.Random(42)
    count = 0

    for row in ds:
        if row["is_safe"]:
            continue
        if not _matches_biochem(row["prompt"]):
            continue

        # Harmful response (undesirable)
        examples.append({
            "prompt": row["prompt"],
            "completion": row["response"],
            "label": False,
        })

        # Refusal (desirable)
        examples.append({
            "prompt": row["prompt"],
            "completion": rng.choice(REFUSAL_TEMPLATES),
            "label": True,
        })

        count += 1
        if count >= max_examples:
            break

    return examples


# ---------------------------------------------------------------------------
# SlimOrca: safe helpful examples
# ---------------------------------------------------------------------------

def _load_slimorca_safe(n: int = 2000, offset: int = 0) -> list[dict]:
    """Load safe instruction-following examples from SlimOrca (label=True).

    Args:
        n: Number of examples to load.
        offset: Starting index (use different offsets for train vs eval to avoid overlap).
    """
    ds = load_dataset("Open-Orca/SlimOrca", split="train")

    examples = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        convs = row["conversations"]

        human_msg = next((c["value"] for c in convs if c["from"] == "human"), None)
        gpt_msg = next((c["value"] for c in convs if c["from"] == "gpt"), None)
        if not human_msg or not gpt_msg:
            continue

        if len(human_msg) > 1000 or len(gpt_msg) > 2000:
            continue

        examples.append({
            "prompt": human_msg,
            "completion": gpt_msg,
            "label": True,
        })
        if len(examples) >= n:
            break

    return examples


# ---------------------------------------------------------------------------
# WMDP: held-out eval (NEVER train on this)
# ---------------------------------------------------------------------------

def _load_wmdp_eval() -> list[dict]:
    """Load WMDP bio+chem as eval prompts.

    Converts MCQ format to open-ended prompts. The model should refuse to
    answer rather than selecting the correct choice.
    """
    prompts = []

    for subset in ["wmdp-bio", "wmdp-chem"]:
        ds = load_dataset("cais/wmdp", subset, split="test")
        for row in ds:
            question = row["question"]
            choices = row["choices"]
            # Format as a natural MCQ prompt
            choices_text = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
            prompt = f"{question}\n{choices_text}"
            prompts.append({"prompt": prompt, "source": subset})

    return prompts


# ---------------------------------------------------------------------------
# HarmBench: secondary eval
# ---------------------------------------------------------------------------

def _load_harmbench_biochem() -> list[dict]:
    """Load bio/chem behaviors from HarmBench for eval.

    HarmBench is a gated dataset — returns empty list if not authenticated.
    """
    try:
        ds = load_dataset("walledai/HarmBench", split="train")
    except Exception:
        print("WARNING: Could not load HarmBench (gated dataset). Skipping.")
        return []

    prompts = []
    for row in ds:
        if row.get("SemanticCategory") != "chemical_biological":
            continue
        if row.get("FunctionalCategory") != "standard":
            continue
        prompts.append({"prompt": row["Behavior"], "source": "harmbench"})

    return prompts


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_for_kto(example: dict, tokenizer) -> dict:
    """Format a single example into the chat template expected by TRL's KTOTrainer."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        "prompt": example["prompt"],
        "completion": example["completion"],
        "label": example["label"],
        "text": text,
    }
