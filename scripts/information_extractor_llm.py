"""
Extraction of organ + cancer_type from whole-slide-image (WSI) reports
constrained to seven organs: stomach, prostate, lung, colon, cervix,
bladder, breast.

Usage
-----
$ pip install "transformers>=4.40" accelerate bitsandbytes jsonschema torch
$ python extract_wsi_entities.py   # runs built-in demo
"""

import json, re, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from jsonschema import validate, ValidationError

# ---------------------------------------------------------------------
# 1. Model & tokenizer
# ---------------------------------------------------------------------
MODEL_NAME = "batsResearch/BioMistral-7B"       # change to Hippo-7B etc.
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    load_in_4bit=True                          # remove for full precision
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_length=512,
    temperature=0.0,
)

# ---------------------------------------------------------------------
# 2. JSON schema with enumerated organs
# ---------------------------------------------------------------------
ORGANS = [
    "stomach", "prostate", "lung",
    "colon", "cervix", "bladder", "breast"
]

SCHEMA = {
    "type": "object",
    "properties": {
        "organ":  {"type": "string", "enum": ORGANS + ["unknown"]},
        "cancer_type": {"type": "string"}
    },
    "required": ["organ", "cancer_type"]
}

# ---------------------------------------------------------------------
# 3. Prompt components
# ---------------------------------------------------------------------
ALLOWED = (
    "Allowed organs: stomach, prostate, lung, colon, cervix, bladder, breast. "
    'If none are mentioned, output {"organ": "unknown", "cancer_type": "unknown"}.'
)

SYSTEM_INST = (
    "You are a medical information-extraction assistant. "
    "Return ONLY a JSON matching this schema: "
    '{"organ": string, "cancer_type": string}. ' + ALLOWED
)

FEW_SHOT = """
Report: "Segment shows invasive carcinoma of the breast..."
JSON: {"organ": "breast", "cancer_type": "invasive ductal carcinoma"}

Report: "Histology confirms colonic adenocarcinoma."
JSON: {"organ": "colon", "cancer_type": "adenocarcinoma"}
""".strip()

# ---------------------------------------------------------------------
# 4. Synonym → canonical mapping (extend as needed)
# ---------------------------------------------------------------------
CANONICAL = {
    "gastric": "stomach",
    "gastric mucosa": "stomach",
    "pulmonary": "lung",
    "bronchus": "lung",
    "uterine cervix": "cervix",
    "carcinoma of prostate": "prostate",
}

def _normalise_organ(name: str) -> str:
    name_lc = name.lower()
    return CANONICAL.get(name_lc, name_lc)

# ---------------------------------------------------------------------
# 5. Core extraction function
# ---------------------------------------------------------------------
def extract_entities(report: str) -> dict:
    prompt = f"{SYSTEM_INST}\n\n{FEW_SHOT}\n\nReport: \"{report.strip()}\"\nJSON:"
    raw = generator(prompt, do_sample=False)[0]["generated_text"]

    # text after last "JSON:"
    json_fragment = re.split(r"JSON:", raw)[-1].strip()
    try:
        data = json.loads(json_fragment)
        data["organ"] = _normalise_organ(data.get("organ", ""))
        validate(data, SCHEMA)
    except (json.JSONDecodeError, ValidationError, TypeError, KeyError):
        data = {"organ": "unknown", "cancer_type": "unknown"}

    return data

# ---------------------------------------------------------------------
# 6. Convenience CLI / demo
# ---------------------------------------------------------------------
def _demo():
    demo_reports = [
        "Whole-slide model notes: 'High-grade serous carcinoma involving ovary and tube.'",
        "Biopsy: 'Poorly differentiated adenocarcinoma of the colon with mucinous features.'",
        "Specimen shows gastric signet-ring cell carcinoma.",
    ]
    for txt in demo_reports:
        print(f"\nReport: {txt}")
        print("Extracted:", extract_entities(txt))

if __name__ == "__main__":
    # If file path(s) passed as arguments, process each line in those files.
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            path = Path(f)
            with path.open() as fh:
                for line in fh:
                    print(json.dumps(extract_entities(line), ensure_ascii=False))
    else:
        _demo()
