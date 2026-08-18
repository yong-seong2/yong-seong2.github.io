# EXAONE-3.5 기반 반도체 뉴스 감성 스코어링 실험

GIST(광주과학기술원) 문체부 과제 연구 인턴십 중 진행한 실험 코드입니다.
외신 경제/기술 기사 중 반도체 산업 관련 기사를 대상으로, LLM(EXAONE-3.5-2.4B-Instruct)에게
직접 감성 점수를 매기게 하는 zero-shot 프롬프트 기반 스코어링을 세 가지 척도로 비교했습니다.

- **binary**: -1(부정) / +1(긍정) 2단 분류
- **trinary**: -1 / 0(중립) / +1 3단 분류
- **score4**: -4 ~ +4의 9단계 세분화 점수

세 스크립트로 나뉘어 있던 원본 코드를 데이터 로딩(`data_utils.py`)과
프롬프트/추론 로직(`inference.py`)으로 정리하고, 척도만 인자로 바꿔 실행할 수 있게
`run_experiment.py` 하나로 합쳤습니다.

## 실행 방법

```bash
pip install -r requirements.txt

# Hugging Face 토큰은 코드에 넣지 않고 환경변수로 설정합니다
export HF_TOKEN=hf_xxx   # Windows는 set HF_TOKEN=hf_xxx

python -m src.run_experiment --scale binary
python -m src.run_experiment --scale trinary
python -m src.run_experiment --scale score4
```

## 입력 데이터

`LABELING.xlsx`(라벨링된 기사 번호)와 `2010_2026_EcoTech.csv`(원문 기사)는
연구용 원본 데이터라 리포지토리에는 포함하지 않았습니다. 같은 형식의 파일을
프로젝트 루트에 두면 바로 실행됩니다.

## 출력

`exaone_3.5_{scale}_results.csv` 형태로 저장되며, 각 행은 다음 컬럼을 가집니다.

| 컬럼 | 설명 |
| --- | --- |
| `article_id` | 원문 기사 번호 |
| `score` | 파싱된 정수 점수 |
| `category` | positive / negative / neutral / parse_error |
| `llm_output` | 모델의 원본 응답 텍스트 |

## 배경

이 실험은 [포트폴리오](https://yong-seong2.github.io)의 GIST 연구 인턴십 프로젝트에서
"직접 설계한 필터링·감성분석 파이프라인"과 별개로, 실제 채택된 방향(토픽모델링·자동 산업분류)
구현 과정에서 LLM 기반 감성 스코어링이 어느 정도 안정적으로 동작하는지 확인하기 위해
비교 실험한 코드입니다.
