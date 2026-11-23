# H2-v2 scaffold-cold 개발 모델 카드

## 예측 계약

모델은 등록 basal RNA와 exact chemical parent로 스크린 내 상대적 단일제 감수성 순위를 예측한다. 모든 단계는 CTRPv2·GDSC2·PRISM20Q2 670,830행과 v1의 scaffold 분할, 소스별 감수성 방향, 가중치, RNA 커널을 공유한다. 약물 절편에는 fit-only 화학 및 선택적 LINCS 이웃 또는 KRR을 사용한다. 환자 약물 선택용으로 검증된 결과는 없다. [데이터와 노출](DATA_SUPPORT_AND_EXPOSURE.md)

- **Assay-assisted drug-cold** 입력은 exact parent의 구조, 가용 표적 및 독립 assay에서 측정한 서명이다. 실제 LINCS 관측 724개와 자연 결측까지 포함한 혼합 배포 점수를 분리한다. DepMap 실측이 모두 없는 세포는 같은 약물 구성의 RNA 기반 모델로 정확히 복귀한다.
- **Structure-only drug-cold**는 질의 단계에서 약물의 실측 LINCS, Cell Painting, Tahoe를 전부 가린다. C1 대조는 고정 화학·표적과 기존 세포 RNA 기반이다. 구조→가상 LINCS 후보는 source training 약물에서만 generator를 적합하고, 반응 적합 약물의 입력 서명도 scaffold 교차 적합으로 생성한다. 가상 서명으로 얻은 임상 의미를 추정하지 않는다.
- LINCS와 Tahoe의 참조 세포는 비간계열로 제한한다. LIMORE·LICOB는 이전 결과 노출 자료이며 DepMap 직접 측정 융합에 필요한 데이터가 없다. GDSC1 신규 골격은 기존 반응과 세포를 공유하는 외부 스크린 기술 검증이다. [잠금](PLAN_LOCK.md)

## 추론과 재생

GBDT는 G1 전용 deps에 고정한 LightGBM 4.6.0을 사용한다. 설치된 lib_lightgbm.so의 SHA-256은 6a12e1a7b13824d14c252d2a4d2e36af467383331a114eb0e7448eecbf02d7b1이다. GPU 잔차와 구조 커널은 G1의 기존 PyTorch 환경에서 적합한다.

봉인된 전체 H2-v2 모델의 네 예측 묶음은 G1의 `nextgen_v2/h2_v2/study/forecasts`에 있다. 각 묶음은 저장된 계수와 입력 표현으로 학습 없이 같은 예측을 다시 계산한다. 선형·RBF·Tanimoto Gram은 PSD이며, 정수 거듭제곱은 Schur 곱으로 구성한다. 가용성 마스크 `MKM`, 단일 지시자 외적, 비음수 가중합, 양의 대각 정규화와 약물·세포 분리 커널의 곱이 PSD를 보존한다. [수치와 재생 영수증](psd_and_replay_receipt.json)

임의 신규 SMILES의 CLI 프로토타입은 등록 기저 RNA에서 **H2-R1 커널 단계**를 실행한다. 이 인터페이스는 SMILES와 선택적 LINCS만 받으며, 물리적으로 입력한 해당 모델의 LINCS 축이 필요하다. LINCS를 생략하면 동일 패키지의 구조 전용 기준선을 실행한다. 등록 표적이 없는 신규 화합물의 표적 벡터는 0이다. TSV는 `feature`, `value` 열을 사용한다. 검증된 G1 파일에서 SHA-256을 확인한 뒤 다음 형식으로 실행한다.

```bash
python -m nextgen_h2_v2.portable G1_PACKAGE --sha256 PACKAGE_SHA256 \
  --smiles CCO --cell-id ACH-000681 --lincs-tsv OPTIONAL_SIGNATURE_TSV
```

별도의 CPU 계산 경로 검사에서 단일 질의 교차 커널과 전체 배치 계산의 최대 차이는 1.14e−5였다. 서로 다른 행렬 크기에서 누적 순서가 달라지는 검사를 fit-only 내부 역할에서 정답 없이 수행했다. 오프라인 재생의 판정은 봉인과 동일한 연산 경로에서 1e−6을 적용한다. [독립 교차식 검사](portable_inner_parity.json)
