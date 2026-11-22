# CLAMP DILI 모델 카드

이 모델은 **약물 구조 기반 FDA 간손상 범주의 1차 선별용 연구 도구**다. 환자별·용량별·병용 임상 간독성 안전성을 보장하지 않는다. FDA DILIrank 2.0의 Most·Less concern을 양성, No concern을 음성으로 학습했다. Ambiguous, 임상 Cmax, 일일 투여량은 제외했다. 공식 CLAMP 분자 인코더와 LightGBM 단일 헤드, Beta 보정기를 역사 v1 702행의 nested scaffold CV로 동결했다. 순위용, 소수 선별용, 통합 배포용에서 같은 모델이 선택됐다. [역사 고정 영수증](evidence/dili_historical_freeze.json) · [사전학습 노출 범위](DILI_PRETRAIN_EXPOSURE.md)

## 오프라인 사용

```bash
python -m confirm_dili.predict --smiles 'CCO'
python -m confirm_dili.predict --batch inputs.tsv
```

배치 파일은 `smiles` 열이 있는 CSV/TSV 또는 줄당 SMILES 하나의 텍스트다. 로컬 복사본을 쓸 때 `--model`과 `--encoder`로 디렉터리를 지정한다. 모델과 공식 인코더를 함께 담은 G1 패키지는 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/confirm_20261123/worker2_dc/task1-01/package/`에 있다. 패키지의 `predict.sh`는 두 로컬 디렉터리를 자동 지정한다.

90% 구간은 역사 connectivity 재표집에 따른 보정 계수의 변동을 나타낸다. 개별 약물의 실제 간손상 위험 예측구간으로 사용할 수 없다. ECFP4 최근접 학습 Tanimoto가 역사 내부에서 동결한 기준 아래이면 확률과 함께 경고·보류 플래그를 반환한다. 별도 환자 자료와 병용 독성 평가는 필요하다. [CLI 검증 영수증](dili_cli_replay_receipt.json)
