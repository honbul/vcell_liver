# DILI v2 모델 카드

DILI v2는 SMILES에서 FDA DILIrank의 이진 concern 라벨을 예측한다. ECFP 및 RDKit 속성, 공개 분자 임베딩, 화합물별 방향성 그래프의 후보를 내부 scaffold OOF 검증에서 비교해 비음수 앙상블을 선택했다. 기전 입력은 BSEP, CYP3A4/2C9/2D6, MMP, ARE, ER 스트레스 등 공개 어세이로 학습된 예측값이다. 임상 노출량, 일일 용량과 임상 라벨 단서는 입력 축에서 제외했다.

DILIrank 모호 판정은 채점 분모에서 제외했다. Murcko와 Butina 개발군은 신규 FDA 155개를 전역 제외하여 732개로 구성했다. 최종 Murcko 및 Butina의 전체 887개를 채점할 때도 학습 세트에서 이 155개를 제외했다. 출시 증분 모델의 학습 라벨은 이전 출시에서 확보한 702개다. 공식 점수는 155개 신규 구조를 조회용으로만 제공하고 모든 설정을 동결한 뒤 계산한다. 공개 기전 어세이 적합에서는 전체 FDA 카탈로그와 connectivity 또는 scaffold가 일치하는 구조를 제외했다.

```bash
python -m liver_dili_v2.predict 'CCO'
```

G1의 동결 checkpoint와 두 FM 경로를 기본값으로 사용한다. 다른 오프라인 장비에서는 모델 디렉터리를 `--model`, 검증된 FM 디렉터리를 `--fm`, ChemBERTa 테이블을 `--chemberta`로 지정한다. 알려진 exact parent에 FM을 조회하고, 새 구조는 구조·그래프 특징과 누락 처리를 사용한다. nearest Tanimoto와 지원 범위 표지를 함께 반환한다. 단일 SMILES는 학습 유병률 기준 보정값을 사용하며 코호트 크기가 충분할 때만 입력 점수 분포의 이동을 추정한다. 이 점수는 FDA 분류 라벨을 지도학습한 구조 기반 연구 점수다.

[방향성 MPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) · [NCATS Tox21](https://tripod.nih.gov/tox21/challenge/data.jsp) · [EPA ToxCast](https://www.epa.gov/comptox-tools/exploring-toxcast-data) · [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)

## 등록 평가와 사용 범위

Murcko와 Butina의 AUROC는 .7484와 .7466이었다. 신규 FDA 증분 155개에서 AUROC .6651, AUPRC .4751, ECE .3867을 기록했다. 비교군 v1은 각각 .6767, .4910, .3606이었다. 신규 증분에서는 최근접 Tanimoto가 큰 80%와 70%만 남겨도 AUROC가 각각 .6484와 .6356이었다. 이 모델의 임상 평가나 DILI v1 대체는 보류하며, 오프라인 구조 기반 연구에 사용한다. [실험 보고서](RESULTS.md) · [동결 채점 원표](evidence/final_score.json)
