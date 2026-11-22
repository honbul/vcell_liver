# V0+ 최종 외부 평가

후속 감사에서 이 결과의 CTRPv2 학습 코어에 단일 약물 두 개로 잘못 등록된 혼합 처리 1,394행이 확인됐다. 표와 동결 영수증은 당시 이력이다. [자료 오류와 정정](SOURCE_PANEL_EXTENSION.md)

**성인 HCC 2D에서 같은 약물에 대한 계통별 감수성 신호를 확인했다. V0+의 2D·3D 두 축 동시 우위는 미충족이다.** Ridge는 계통별 감수성, CellHit는 세포별 약물 순위에서 더 높은 점수를 냈다. 후속 LICOB의 29명·53약물에서도 V0+의 환자별 전체 약물 순위가 약물 평균보다 높았고 NDCG@5는 낮았다. 같은 약물의 환자 순위에서는 CODE-AE가 가장 높았다. [LICOB 성적](LICOB_AUTHOR_RESULTS.md) · [Wong 3D 성적](WONG_3D_EXTERNAL_RESULTS.md)

## LIMORE 50개 독립 계통

아래는 최초 평가 이력이다. 같은 50계통에서 단일 약물 정정 학습표와 정정 배포판으로 동결한 후속 성적은 별도 기술 평가에 있다. [LIMORE 정정 기술 평가](LIMORE_CORRECTED_RESULTS.md)

GDSC2·CTRPv2·PRISM과 세포·환자 계통 매핑 중복이 없는 50개 성인 HCC 계통의 68개 약물·3,400개 AA 관측이다. 21개 방법의 같은 71,400개 예측을 봉인하고 1회 별도 채점했다. 0단계 연구에서 LIMORE를 이미 분석한 이력은 기록했다. V0+의 fit·설정·출력 배율에는 LIMORE 반응을 사용하지 않았다.

| 방법 | 약물 내 Spearman | 약물 내 Pearson | 세포 내 NDCG@5 | Precision@5 | 세포 내 Spearman |
|---|---:|---:|---:|---:|---:|
| 약물 평균 | .000 | .000 | .761 | .292 | .531 |
| 약물+세포 가산 | .048 | .044 | .761 | .292 | .531 |
| 약물별 Ridge | .240 | .268 | .748 | .288 | .517 |
| 약물별 ElasticNet | .137 | .181 | .762 | .276 | .550 |
| LightGBM | .094 | .167 | .763 | .284 | .549 |
| CatBoost | .083 | .166 | .762 | .288 | .556 |
| CellHit | .101 | .150 | .774 | .296 | .557 |
| Velodrome inductive | .206 | .239 | .745 | .296 | .522 |
| CODE-AE inductive | .093 | .129 | .749 | .276 | .540 |
| V0+ inductive | .154 | .191 | .756 | .308 | .548 |
| V0+ transductive | .086 | .115 | .756 | .300 | .542 |
| Ridge+V0+ 고정 평균 | .234 | .275 | .752 | .300 | .543 |

약물 평균 대비 V0+의 약물 내 Spearman은 +.154 [95% .067, .238], Pearson은 +.191 [.094, .272]였다. 세포 내 Spearman은 +.017 [.005, .029]였다. NDCG@5는 −.005 [−.022, .014]로 설정한 동시 개선 조건을 넘지 못했다. 약물 평균은 각 약물에서 모든 세포에 같은 값을 주므로 약물 내 상관을 0으로 기록했다.

V0+의 약물 내 Spearman은 Ridge 대비 −.086 [−.179, .008], CellHit 대비 +.052 [−.046, .147]이었다. 세포 내 Spearman은 Ridge보다 +.031 [.013, .049], CellHit보다 −.009 [−.017, −.001]이었다. CellHit 대비 NDCG@5도 −.018 [−.033, −.003]이었다. 고정 평균은 Ridge의 개인화 성능에 가까웠고 세포 내 Spearman은 +.027 [.017, .036] 높았다. 고정 평균의 NDCG@5는 약물 평균보다 −.009 [−.029, .014]였다. 약물 내 구간은 계통과 약물을 함께, 세포 내 구간은 계통을 단위로 짝지어 재표집했다.

## 약물군과 실패 모드

| 약물군 | 약물 수 | 약물 내 Spearman V0+ | Ridge | CellHit | 고정 평균 |
|---|---:|---:|---:|---:|---:|
| 키나아제·HDAC | 37 | .142 | .207 | .068 | .201 |
| 비특이 세포독성 | 12 | .223 | .310 | .125 | .323 |
| HCC TKI | 4 | .067 | .084 | .192 | .098 |
| EGFR·HER2 | 6 | .117 | .191 | .166 | .181 |
| MET | 4 | .075 | .136 | .094 | .113 |
| FGFR | 3 | .188 | .145 | .165 | .186 |
| mTOR | 2 | .264 | .295 | −.091 | .349 |
| HDAC | 3 | .002 | .297 | −.031 | .185 |

V0+의 sorafenib·cabozantinib 약물 내 Spearman은 각각 −.076·−.056이었다. Regorafenib과 lenvatinib은 .281·.120이었다. 주요 HCC TKI를 개별 샘플에서 순위화할 때 성능 편차가 있었다. 약물 수가 작은 군의 평균은 기술 통계다.

외부 RNA를 합쳐 정렬한 V0+는 inductive의 약물 내 Spearman .154에서 .086으로 낮아졌다. 짝지은 차이는 −.068 [−.156, .019]였다. 고정 아키텍처에서 경로 제외는 .165, 전체 특징은 .137이었다. 같은 고정 설정의 표적 attention 제외는 .123, rank 정규화 제외는 .114였다. 이 절제에서 특정 특징의 독립적인 유효성을 확정하지 않았다. [GPU 내부 결과](STEP3_MOLECULAR_CV.md) · [전체 지표, 약물별 결과 및 재표집](limore_final_score.json)

## 독립 3D

Wong의 환자 유래 HCC 오가노이드 10개와 27약물에서는 V0+의 약물 내 Spearman .061, 오가노이드 내 Spearman .495, NDCG@5 .780을 관측했다. 약물 평균 대비 오가노이드 내 Spearman 차이는 +.044 [95% .021, .070]이었다. NDCG@5와 약물 내 Spearman 개선 구간은 0을 포함했다. Broutier의 독립 오가노이드 2개·7약물 기술 평가에서 V0+의 약물 순위 Spearman은 .554였다. Hu의 독립 PDX 16개에서 sorafenib 감수성 순위 Spearman은 −.324였다. LICOB NODE 배포본은 8 MiB가 손상됐다. 후속 저자 패키지 RNA로 공식 S6와 34모델을 연결해 29명·53약물을 채점했다. V0+의 약물 내 Spearman은 .097, 환자 내 Spearman은 .407, NDCG@5는 .642였다. [LICOB 성적](LICOB_AUTHOR_RESULTS.md) [Wong 3D 성적](WONG_3D_EXTERNAL_RESULTS.md) · [Broutier 기술 채점](STEP4_3D_TECHNICAL.md) · [Hu PDX 및 LICOB 재감사](STEP4_3D_PUBLIC_ACCESS_FOLLOWUP.md)

예측 병합·동결 `32466`, 별도 채점 `32467`, 패키지·CPU 재현 `32468→32469`. 동결 SHA-256 `894d5dc7a5fb7efdca49f261ba87aa58cf7b520595993a681fdb336175c17dba`. [동결 영수증](limore_forecast_receipt.json) · [오프라인 모델 카드](V0PLUS_MODEL_CARD.md)

이 외부 회차 이후의 구조 커널 실험은 세 학습 스크린의 계통·assay 보류에서 평가했다. [소스 전이 개발 성적](STRUCTURED_CV_RESULTS.md)

정정 코어를 사용한 별도 Yang 3D 평가에서 Ridge·구조 커널 결합의 74명·5약물 주 기준은 미통과였다. 세포주 예측의 기저 RNA 전이와 같은 조건에서 13개 모델을 비교했다. [Yang 동결 성적](YANG_HCC_EXTERNAL_RESULTS.md)

LICOB NODE RNA의 복구 축으로 이미 채점한 29공여자와 겹치지 않는 성인 HCC 공여자 9명·53약물을 새로 확인했다. 구조 커널은 사전 공동 개선 조건을 충족하지 못했다. [새 공여자 동결 성적](LICOB_NODE_RECOVERY_RESULTS.md)

넓은 공개 RNA 축과 정확한 단일 약물 두 개를 추가해 LICOB 38명·55약물을 평가했다. 연구를 통째로 보류하고 다른 두 3D 연구에서 학습한 잔차도 비교했다. 두 기술 평가에서 구조 커널 또는 3D 보정의 공동 개선 기준은 미통과였다. [넓은 RNA 및 약물 지원](LICOB_WIDE_RNA_RESULTS.md) · [세 연구의 보류 평가](HCC_3D_STUDY_POOL_RESULTS.md)

GDSC2·CTRPv2·PRISM에서 전반적인 약물 민감도와 약물별 상호작용을 분리해 LIMORE 50계통과 LICOB 38공여자에서 기존 예측과 비교했다. 공통 민감도의 외부 전이 방향이 두 코호트에서 달랐고, 분리 후보의 공동 개선 기준은 미통과였다. [반응 성분의 분리 평가](RESPONSE_COMPONENT_RESULTS.md)

Wong의 기존 27약물에서 62약물로 평가를 넓혔다. 같은 10명의 원발 종양 RNA와 오가노이드 RNA를 각각 입력해 같은 3D 실측에 비교했다. 신규 35약물의 상위 순위는 두 입력에서 약물 평균보다 나아졌고 약물 내 환자 감수성은 공동 승격 기준에 미달했다. [공개 약물 확장과 원발 종양 RNA 결과](WONG_EXPANDED_RESULTS.md)

같은 환자의 2D 세포주와 3D 오가노이드가 모두 측정된 Wong 8명·100약물에서 형식 간 직접 전달을 평가했다. 환자와 약물의 공통 민감도를 분리하면 두 형식에서 공유되는 약물별 상호작용이 관측됐으며, 원 반응의 약물별 환자 순위는 재현되지 않았다. [짝지은 실측 및 공여자 보류 보정](WONG_ASSAY_BRIDGE_RESULTS.md)

LICOB 화합물의 입체 구조를 바로잡아 기존 훈련 연결의 오류와 누락을 확인했다. 43개 HCC 오가노이드의 76약물 전수 예측을 동결하고 성인 38명에서 채점했다. 미관측 구조의 주 평가에서 화학·RNA 상호작용 모형은 사전 기준을 충족하지 못했다. [화학 재등록 및 76약물 점수](LICOB_COLD_RESULTS.md)

LIMORE 핵심 공개 2D 검증에는 ECFP·기저 RNA·경로, 표적 유전자 발현, 실측 HCC 변이·절대 CNA를 함께 넣은 비교가 있다. 실측 결합은 약물 평균 대비 공동 지표를 통과했고 DNA 추가 효과는 불확실했다. LICOB의 물리 농도를 맞춘 Emax·IC50 모델은 모두 사전 공동 개선 기준에 미달했다. [LIMORE 분자 입력 비교](MEASURED_DRIVER_RESULTS.md) · [LICOB 종말점 비교](LICOB_DOSE_ENDPOINT_RESULTS.md)

LIMORE의 구조 커널과 같은 학습표의 Ridge·약물 평균을 68약물 오프라인 패키지로 제공한다. LIMORE 3,400개 동결 예측을 CPU에서 모두 재현했고 Wong 원발 조직 RNA의 270개 등록 요청을 처리했다. LICOB 가공 발현은 이 배포판의 값 범위·유전자 커버리지 규약에 미달했다. [구조 커널 배포와 입력 검증](STRUCTURED_RELEASE_MODEL_CARD.md)

Wong의 10명에서 다른 환자의 원발 종양·오가노이드 RNA 쌍으로 원발 RNA를 교정했다. 분자 수준의 발현 순위는 실제 오가노이드에 가까워졌고, 62약물에서 약물 내 환자 순위와 상위 약물 순위의 공동 개선 기준은 미통과였다. [공여자 보류 RNA 교정](WONG_PAIRED_RNA_ADAPTER_RESULTS.md)

LICOB 38약물의 공통 물리 농도에서 2D 곡선을 학습했다. IC50의 약물 내 환자 감수성은 약물 평균보다 높았고 상위 5개 선별은 낮았다. 보조 Emax에서는 약물 평균 대비 두 지표가 올랐으며 같은 지지의 Emax 직접 적합 대비 추가 효과는 불확실했다. [농도별 곡선 전이](LICOB_CURVE_TRANSFER_RESULTS.md)

동일한 38약물에서 2D 반응 도달 확률의 IC50 공동 기준도 미통과였다. 공개 S7의 IC50은 전부 최고 농도 이하였고, 동일 표본의 S8 Emax에서 생존율 .5를 넘은 경우가 487개였다. [도달 확률과 공개 종말점 감사](LICOB_CROSSING_RESULTS.md)

Oslo 대장암 간전이 자료의 91개 PDO·52명에서 기저 RNA와 웰별 농도 반응을 연결했다. 20개 동일 제품의 두 라이브러리 간 공여자 보류에서 RNA+경로 커널의 공동 기준은 미통과였다. LICOB와 이름이 같은 14개 중 전체 농도 범위를 덮는 후보는 1개였다. [공개 원자료 감사 및 보류 성적](OSLO_3D_CURVE_RESULTS.md)

후속 화학 확인으로 Oslo–LICOB의 입체 구조를 지정한 13약물에서 3D RNA 반응을 기존 2D 모델에 추가했다. LICOB 38명의 AUC에서 추가 편차는 공동 개선 기준을 통과하지 못했다. 최고 물리 농도를 맞춘 Emax는 2약물에 비교 지지가 있었고 ceralasertib은 Oslo 단독 기술값을 보고했다. [Oslo–LICOB 전이](OSLO_LICOB_TRANSFER_RESULTS.md)

새로 연결한 Singapore 치료 전 PDX RNA에서 sorafenib 구조 커널의 약물 내 두 상관은 양수였지만 10환자군의 사전 기준은 미통과였다. Vinorelbine 실측의 2023–2024 공통 11군에서 Spearman .782 [+.183, +.972]로 사전 재현 기준을 통과했다. PRISM의 같은 parent 반복 표기 처리도 바로잡아 vinorelbine과 everolimus의 학습 지지를 복원했다. [PDX 결과와 PRISM 감사](SHUEN_PDX_RESULTS.md)

복원한 PRISM 데이터를 사용해 vinorelbine의 2D 실측으로 학습하고 Singapore PDX의 공통 6환자군을 별도 평가했다. 기전 커널의 약물 내 순위는 양수였으며 두 출판물 모두에서 기준 미통과였다. LICOB에서는 기존 학습에 없던 7약물의 정확한 PRISM 실측을 추가해 같은 38명에서 평가했다. 상위 5약물 점 추정치와 약물별 환자 순위가 서로 다른 방향으로 움직여 공동 기준에 미달했다. [Vinorelbine PDX](VINORELBINE_TRANSFER_RESULTS.md) · [LICOB PRISM 7약물](LICOB_PRISM_RESTORED_RESULTS.md)

Vinorelbine은 286개 동일 2D 계통에서 실제 농도를 일치시키면 두 스크린의 약물 내 Spearman이 .329에서 .104로 감소했다. 같은 구간으로 적합한 모델의 PDX 공동 기준도 미통과였다. LICOB에서 최고 두 농도가 지원되는 ispinesib 한 종은 별도 동결했다. PRISM 기술 반복의 Emax 순위 상관은 .874였으나 LICOB 38명의 같은 약물 내 예측은 .033으로 미통과였다. [Vinorelbine 원자료](VINORELBINE_RAW_DOSE_RESULTS.md) · [Ispinesib 최고 농도](LICOB_ISPINESIB_EMAX_RESULTS.md)

PRISM 검출 반복이 높은 농도를 더 반영하도록 24약물·9,681개 곡선의 학습 표적을 바꿨다. 같은 지지에서 발행 AUC와 균등 원자료 두 조건을 대조했다. 2D 두 스크린 보류, LIMORE 독립 50계통, LICOB 성인 38명 모두의 공동 기준은 미통과였다. LIMORE에서 해당 14약물의 약물 내 상관은 소폭 높았고, LICOB 24약물에서는 세 개인화 모델 모두 0에 가까웠다. [반복 가중과 두 외부 코호트](PRISM_RELIABLE_DOSE_RESULTS.md)

유전자별 RNA 분포를 평가 코호트에 맞추면 참조 분포의 극단 순위가 LIMORE 14.8%→8.0%, LICOB 33.1%→10.6%로 줄었다. 같은 약물에 대한 개인화의 공동 개선은 두 코호트 모두 사전 기준에 미달했다. [유전자별 보정의 전이 평가](GENE_CALIBRATION_RESULTS.md)

Wong의 원발 종양–오가노이드 RNA 10쌍으로 구한 방향을 2D 학습과 외부 평가 양쪽에서 제거했다. LICOB NDCG@5는 .624→.633으로 올랐다. 약물별 환자 순위는 공여자 짝 순열보다 낮았고 두 코호트의 사전 공동 기준은 미통과였다. [짝지은 RNA 사영의 전이 평가](RNA_STATE_PROJECTION_RESULTS.md)

LIMORE에서 점 추정 .363을 기록한 별도 ECFP4·경로 ElasticNet의 계통 순위는 68개 약물 모두에서 같았다. 새 가산 대조에서도 약물 내 Spearman은 LIMORE .334, LICOB −.091로 방향이 달랐다. 약물별 경로 효과를 분리한 희소 회귀는 두 코호트 공동 승격 기준에 미달했다. [공통 감수성과 약물별 효과의 전이](SPARSE_PATHWAY_TRANSFER_RESULTS.md)

약물별 상호작용을 직접 학습한 신경망을 구조 커널에 더하면 LICOB NDCG@5는 .624→.661로 올랐다. 공여자 내 전체 약물 순위 Spearman은 .339→.277로 낮아졌다. 실측을 쓰지 않은 보정 분해에서 LICOB 보정 제곱합의 77.7%는 시료가 공유하는 약물별 편차였다. [상호작용 신경망과 사후 진단](NEURAL_INTERACTION_TRANSFER_RESULTS.md)

동결 보정을 세 성분의 여덟 조합으로 나누면 LICOB 전체 순위 하락에 대한 약물 공통 성분의 기여는 −.0583이었다. 상호작용 성분의 상위 5개 선별 기여는 양수였으며, 상호작용 단독은 구조 커널 대비 공동 개선 기준을 충족하지 못했다. [성분별 짝지은 재채점](NEURAL_OFFSET_FACTORIAL_RESULTS.md)
