"""사용 예:

    python -m src.run_experiment --scale binary
    python -m src.run_experiment --scale trinary
    python -m src.run_experiment --scale score4

HF_TOKEN은 환경변수로 설정하고 실행하세요.
    export HF_TOKEN=hf_xxx   (macOS/Linux)
    set HF_TOKEN=hf_xxx      (Windows)
"""

import argparse

import pandas as pd
from tqdm import tqdm

from .data_utils import prepare_data
from .inference import infer_score, load_model


def category_label(score_val, scale):
    if score_val is None:
        return "parse_error"
    if scale in ("binary", "trinary"):
        if score_val > 0:
            return "positive"
        if score_val < 0:
            return "negative"
        return "neutral"
    # score4: 부호로만 대분류하고 원점수는 별도 컬럼에 남김
    if score_val > 0:
        return "positive"
    if score_val < 0:
        return "negative"
    return "neutral"


def main():
    parser = argparse.ArgumentParser(description="EXAONE-3.5 기반 반도체 뉴스 감성 스코어링")
    parser.add_argument("--scale", choices=["binary", "trinary", "score4"], required=True)
    parser.add_argument("--label-path", default="LABELING.xlsx")
    parser.add_argument("--corpus-path", default="2010_2026_EcoTech.csv")
    parser.add_argument("--out", default=None, help="결과 CSV 경로 (기본: exaone_3.5_{scale}_results.csv)")
    args = parser.parse_args()

    out_path = args.out or f"exaone_3.5_{args.scale}_results.csv"

    samples = prepare_data(args.label_path, args.corpus_path)
    if not samples:
        print("분석 대상이 없습니다.")
        return

    tokenizer, model = load_model()

    results = []
    print(f"분석 시작 (대상: {len(samples)}건, scale={args.scale})")
    for item in tqdm(samples):
        score_val, raw_output, err = infer_score(tokenizer, model, item["content"], args.scale)
        results.append(
            {
                "article_id": item["id"],
                "score": score_val if score_val is not None else "",
                "category": category_label(score_val, args.scale),
                "llm_output": raw_output,
            }
        )

    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"분석 완료: {out_path}")


if __name__ == "__main__":
    main()
