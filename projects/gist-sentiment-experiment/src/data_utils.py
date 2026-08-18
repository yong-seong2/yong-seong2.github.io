"""라벨링된 기사 샘플을 불러오는 공통 유틸리티.

세 실험(binary / trinary / score-4) 스크립트가 전부 동일한 방식으로
LABELING.xlsx + 원문 CSV를 결합해 샘플을 만들었기 때문에, 중복을 없애고
이 모듈 하나로 통일했습니다.
"""

import pandas as pd


def prepare_data(label_path: str = "LABELING.xlsx", corpus_path: str = "2010_2026_EcoTech.csv"):
    """라벨이 있는 기사 번호만 골라, 원문 기사와 매칭해 리스트로 반환합니다.

    Returns
    -------
    list[dict]  # [{"id": 기사번호, "content": 본문}, ...]
    """
    try:
        label_df = pd.read_excel(label_path)

        try:
            main_df = pd.read_csv(corpus_path, encoding="cp949")
        except UnicodeDecodeError:
            main_df = pd.read_csv(corpus_path, encoding="utf-8-sig")

        test_target = label_df.dropna(subset=["label"]).copy()
        test_target["article number"] = pd.to_numeric(test_target["article number"]).astype(int)

        samples = []
        for num in test_target["article number"]:
            idx = num - 1
            if idx < len(main_df):
                content = (
                    str(main_df.iloc[idx]["body"])
                    if "body" in main_df.columns
                    else str(main_df.iloc[idx].iloc[1])
                )
                samples.append({"id": num, "content": content})
        return samples

    except Exception as e:
        print(f"데이터 로드 에러: {e}")
        return []
