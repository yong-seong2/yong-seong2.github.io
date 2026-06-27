고객 이탈 예측 및 LTV 분석

제2회 데이터 분석 경진대회 · 한국데이터정보과학회 × SAS KOREA
팀 제타바이트 (4인) · 2026.04 ~ 05 · 전국 4위 · 본선 진출 · 장려상


프로젝트 개요

항목내용과제고객 이탈 확률 예측(Churn, AUC) + 향후 1년 LTV 예측(RMSE)데이터고객정보·거래이력·금융프로필 3개 테이블, Train 60,000명 / Test 40,000명이탈률9.88% (관측 기간 6개월)최종 성적Churn AUC 0.7993 / LTV RMSE 1,347,538 / Total Score 0.46978팀 구성4인 (팀 리더 · 본선 발표 담당)


4인 중 2명은 머신러닝 대회 참가가 처음이었습니다. 대회 초반에는 분석 흐름과 코드 작성 방식을 맞추는 데 시간을 들였고, 리더보드 제출 횟수가 제한된 대회 특성상 팀 차원에서 멀티시드 검증으로 안정성을 먼저 확보하는 전략을 택했습니다.



나의 역할


3개 이종 테이블 통합 파이프라인 설계 및 전체 코드 구현
핵심 파생변수(short_term_debt_ratio) 직접 설계
모델 선정(LightGBM 단독) 및 Multi-seed 검증 체계 구축
30회 이상의 실험을 기록하고 채택/기각 기준을 데이터로 결정
본선 발표 담당
churn_resistance_score(RFM Rank 변수)는 팀원이 기여


핵심 파생변수

short_term_debt_ratio = (현금서비스 + 카드론) / 예적금 잔액
전체 부채가 아닌 단기 고위험 부채만 분리해 "지금 당장 급전이 필요한 정도"를 직접 측정 → Feature Importance 1위, AUC +0.0013

churn_resistance_score = (F_rank + M_rank + (1 − R_rank)) / 3 (팀원 기여)
RFM을 절대값이 아닌 백분위(rank)로 변환해 극단값 영향 제거 → seed 간 표준편차 20% 감소

모델링 전략


2-Stage Cascading: Stage 1(Churn, LightGBM Binary) → Churn 예측값을 Stage 2(LTV, LightGBM Tweedie)의 피처로 활용
단일 모델 선택 근거: LightGBM(AUC 0.8004, 15분) vs XGBoost/CatBoost/앙상블 — 성능 대비 연산 비용에서 압도적 우위, CatBoost는 seed별 변동성 (std 0.02~0.04)이 커서 제외
검증: 5-Fold Stratified CV × Multi-seed(5~10개) 평균 → 단일 seed 과적합 방지


실험 기록 (30회 이상)

효과 있었던 시도결과STR 파생변수AUC +0.0013Optuna 하이퍼파라미터 튜닝AUC +0.0008LightGBM 단독 (CatBoost/XGBoost 제외)AUC +0.0009RFM Rank 통합std 20% 감소전처리·모델 파라미터 동시 튜닝AUC +0.0025

효과 없었던 시도기각 근거Finance 교차 피처 17개5-seed 전패유전 프로그래밍 수식 탐색 (907개)3-fold↑, 5-fold↓ (fold 수 과적합)GA 피처 선택 (43→29개)local 유지, LB 하락 (LB 과적합 확인)Finance 클러스터링, rank 변환 다수local 하락

주차별 진행

주차내용리더보드1주베이스라인 구축, EDAAUC 0.7970 (2위)2주STR 변수 개발, SVD 임베딩—3주모델 비교·확정, 2-Stage 완성AUC 0.7984 (3위)4주GA·유전 프로그래밍 시도 후 실패, 정보 천장 인식AUC 0.7978 (5위)5주RFM Rank 반영, 파라미터 동시 튜닝AUC 0.7993 (4위, 본선 진출)

비즈니스 시사점


카드론 사용 고객의 이탈률 19.4% (미사용 대비 2.6배) → 금리 우대·상환 컨설팅 등 선제 대응 가능
성별·지역 등 인구통계보다 행동 패턴(구매 빈도 변화, 최근 소비 감소)이 핵심 변수
유지 고객 LTV가 이탈 고객 대비 약 20배 → 소액 리텐션 마케팅으로도 높은 ROI


교훈


도메인 지식 기반 파생변수가 자동화 기법(유전 프로그래밍, GA 피처선택)보다 효과적이었다.
피처는 양보다 질이 중요했다 — 줄이면 과적합, 늘리면 noise.
단일 seed 성능은 신뢰할 수 없다. Multi-seed 검증 없이는 의사결정이 어렵다.
Local-LB 성능 차이를 항상 확인해야 한다.


기술 스택

Python LightGBM Optuna scikit-learn TF-IDF / TruncatedSVD IsolationForest Stratified K-Fold CV

파일 구성

├── 최종코드파일.ipynb        # 전체 파이프라인 코드

├── 제타바이트_발표자료.pdf    # 본선 발표 자료

├── 제타바이트_submission_5.csv  # 제출 결과 샘플

└── 5주차_results.pdf  # 결과 순위 파일
