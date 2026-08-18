"""EXAONE-3.5 기반 반도체 산업 감성 점수 추론.

세 가지 척도(binary / trinary / score-4)를 하나의 인터페이스로 묶었습니다.
원래는 척도별로 스크립트가 3개 분리되어 있었는데, 실제로 다른 부분은
프롬프트의 SCORING SCALE 문구와 결과 파싱 로직뿐이라 함수로 정리했습니다.
"""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

SCALE_PROMPTS = {
    "binary": {
        "instruction": (
            "### SCORING SCALE (Binary Scale: -1, 1)\n"
            "You MUST choose ONLY one of the following two values:\n"
            "-1 : Negative\n"
            "+1 : Positive\n\n"
            "0 is NOT allowed."
        ),
        "pattern": r"(-1|1)",
        "valid": {-1, 1},
    },
    "trinary": {
        "instruction": (
            "### SCORING SCALE (Trinary Scale: -1, 0, 1)\n"
            "Classify the sentiment into one of the following three categories:\n"
            "-1 : Negative (e.g., Decline, restrictions, supply chain disruption, loss)\n"
            " 0 : Neutral (e.g., No significant impact, purely factual report, status quo)\n"
            "+1 : Positive (e.g., Growth, recovery, investment expansion, earnings surprise)"
        ),
        "pattern": r"(-?\d)",
        "valid": {-1, 0, 1},
    },
    "score4": {
        "instruction": (
            "### SCORING SCALE (-4 to 4)\n"
            "Score the overall sentiment on an integer scale from -4 (extremely negative) "
            "to +4 (extremely positive), where 0 is strictly neutral."
        ),
        "pattern": r"(-?\d)",
        "valid": set(range(-4, 5)),
    },
}


def load_model(hf_token: str | None = None):
    """토큰은 인자로 넘기거나 HF_TOKEN 환경변수에서 읽습니다. 코드에 직접 쓰지 않습니다."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Hugging Face 토큰이 없습니다. 환경변수 HF_TOKEN을 설정하거나 "
            "load_model(hf_token=...)으로 직접 넘겨주세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    return tokenizer, model


def build_prompt(article_text: str, scale: str) -> str:
    scale_block = SCALE_PROMPTS[scale]["instruction"]
    return f"""### ROLE
You are an expert Economic Analyst specializing in the global Semiconductor Industry.

### TASK
Read the provided news article and return ONE sentiment score assessing the overall performance of the semiconductor industry.
In your assessment, consider factors such as Export, Production, and Investment.

{scale_block}

### INPUT ARTICLE
{article_text}

### OUTPUT FORMAT
Return ONLY a valid JSON object with a single integer score.
Please refrain from providing further comments.

Example:
{{ "semiconductor_activity_score": -1 }}
"""


def parse_score(response_text: str, scale: str):
    pattern = SCALE_PROMPTS[scale]["pattern"]
    valid = SCALE_PROMPTS[scale]["valid"]
    m = re.search(pattern, response_text)
    if not m:
        return None
    val = int(m.group(1))
    return val if val in valid else None


def infer_score(tokenizer, model, article_text: str, scale: str, max_retry: int = 2):
    base_prompt = build_prompt(article_text, scale)

    response = ""
    for attempt in range(max_retry + 1):
        prompt = base_prompt
        if attempt > 0:
            prompt += "\nOutput ONLY JSON with a single integer score. No other tokens."

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        score_val = parse_score(response, scale)
        if score_val is not None:
            return score_val, response, None

    return None, response, "parse_error"
