#!/usr/bin/env python
# extract_pathology.py
# pip install transformers accelerate bitsandbytes optimum
'''
import json, re, time, gc, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ───────────────────────── USER-EDITABLE SETTINGS ──────────────────────────
IN_FILE  = Path("../running_dir/competition_data/batch_reports.json")           # input list [{"id":..,"report":..}, ...]
OUT_FILE = Path("../running_dir/competitoin_data/llm_processed_batch_reports.json")   # where results are written
CUDA_ID   = 2

MODELS = {                                          # add / remove models here
    #"medalpaca":   "medalpaca/medalpaca-7b",
    #"mistral_gen": "mistralai/Mistral-7B-Instruct-v0.2",
    "biomistral":    "BioMistral/BioMistral-7B",
    "biogpt":        "microsoft/biogpt" 

}

PROMPT = """You are a clinical NLP assistant.

Extract from the histopathology report:
1. organ
2. cancer_type (basal cell carcinoma, melanoma, none, etc.)
3. procedure (punch biopsy, excision, etc.)

Return a single-line JSON: {{"organ":"...","cancer_type":"...","procedure":"..."}}

Report:
{report}
"""


# ────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": f"cuda:{cuda_id}"},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        temperature=0.0,
        max_new_tokens=256,
        batch_size=1
    )

JSON_RE = re.compile(r"\{.*\}", re.S)

def safe_parse(text: str):
    m = JSON_RE.search(text[text.rfind("{"):])
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    reports = json.loads(IN_FILE.read_text(encoding="utf-8"))
    results = {name: [] for name in MODELS}

    for name, model_id in MODELS.items():
        print(f"\nLoading {name} …")
        gen = load_model(model_id)

        for rec in reports:
            t0 = time.perf_counter()
            out_text = gen(PROMPT.format(report=rec["report"].strip()))[0]["generated_text"]
            latency = round(time.perf_counter() - t0, 3)

            data = safe_parse(out_text)
            data.update(id=rec.get("id"), latency_s=latency)
            results[name].append(data)

        del gen; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll done → {OUT_FILE}")

if __name__ == "__main__":
    main()

'''

#!/usr/bin/env python
# extract_to_txt.py
# pip install transformers accelerate bitsandbytes optimum

from pathlib import Path
import json, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── SETTINGS ───────────────────────────────────────────────────────────────
IN_FILE   = Path("../running_dir/competition_data/batch_reports.json")      # list of {"id":.., "report":..}
OUT_TXT   = Path("../running_dir/competitoin_data/llm_processed_batch_reports.json")         # plain-text destination
MODEL_ID  = "BioMistral/BioMistral-7B"
CUDA_ID   = 2                                   # ← run on GPU 2 (cuda:2)
# ───────────────────────────────────────────────────────────────────────────

PROMPT_HEADER = """You are a clinical NLP assistant.

Extract from the histopathology report:
1. organ
2. cancer_type (basal cell carcinoma, melanoma, none, etc.)
3. procedure (punch biopsy, excision, etc.)

"""

def load_model(model_id: str, cuda_id: int):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{cuda_id}"},   # place all weights on cuda:2
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,                      # pipeline will call cuda:2
        temperature=0.0,
        max_new_tokens=256,
        batch_size=1
    )

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    reports = json.loads(IN_FILE.read_text(encoding="utf-8"))
    gen = load_model(MODEL_ID, CUDA_ID)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with OUT_TXT.open("w", encoding="utf-8") as fh:
        for rec in reports:
            prompt = PROMPT_HEADER + rec["report"].strip()
            answer = gen(prompt)[0]["generated_text"].split(prompt)[-1].strip()
            fh.write(f"{rec['id']} :: {answer}\n\n")

    del gen; gc.collect()
    torch.cuda.empty_cache()

    print(f"Done. Outputs saved to {OUT_TXT}")

if __name__ == "__main__":
    main()


