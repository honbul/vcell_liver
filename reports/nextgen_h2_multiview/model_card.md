# H2 개발 모델 카드

## 입력과 용도

출력은 세포 기저 RNA와 등록 약물에 대한 상대적 단일제 감수성이다. 주요 검증은 동일 공개 반응 패널의 신규 화학 골격이다. 세포별 치료 선택이나 임상 용량 추론에 적용할 근거는 없다. GDSC2·CTRPv2·PRISM20Q2의 원래 AUC/AA 방향을 감수성 증가 방향으로 정렬했고 학습 배치 간 양의 변환을 사용한다. 출력은 플랫폼별 보정 확률이나 생존율 단위가 아니다. [등록·학습 소스](PLAN_LOCK.md)

- **Structure-only drug-cold**는 등록 exact parent의 ECFP4, RDKit 네 물성, 사용 가능한 ChEMBL 37 표적과 질의 세포의 치료 전 RNA를 사용한다. 알려진 표적이 없는 부모의 표적 벡터는 0이다. 화학적 exact parent 키와 표적 어휘는 적합에 제한하고, 신규 약물은 fitting 어휘에 존재하는 표적을 활성화한다.
- **Assay-assisted drug-cold**는 해당 부모의 독립 참조 assay에서 관측한 Cell Painting 또는 LINCS 서명을 추가한다. LINCS 조직은 알려진 비간계열에 제한되고 기저 RNA와 직접 비교하는 경로 지문에는 일부 best-inferred 유전자가 있다. Test 약물의 AUC는 모델 입력에 없다. 보조 프로파일이 없으면 예측은 같은 파일에 고정한 C1과 완전히 일치한다. [지원·노출 기록](DATA_SUPPORT_AND_EXPOSURE.md)
- 2D 등록 반응과 동일한 RNA rank/경로/TF 및 RNA에서 예측한 Chronos/driver cell basis를 모든 arm에서 쓴다. 관측 행, 응답 타깃과 방향, RNA 처리는 공통이다. 관측 매트릭스에 칸막이 형태의 커널을 쓰며 대형 cell-drug pair Gram은 생성하지 않는다. [선행 실험 재사용](PRIOR_EXPERIMENT_REUSE.md)

## 선택과 해석

동일한 nested 선택 규칙으로 outer-fit마다 하나의 챔피언을 선택한다. 전체 소스에서 적합하는 LIMORE·LICOB·GDSC1 기술 모델은 사전에 잠근 inner 결과 다수결을 따른다. 결측 비교에서 PSD 합의 합성 항은 가용성으로 양쪽을 마스킹하고, 대각 정규화는 fit과 query의 각 자기 커널만 사용한다. 잔차 게이트는 scaffold OOF 기준선 잔차로 적합한다. 모든 fit·query 변환은 적합 분산과 척도를 재사용한다. [잠금](PLAN_LOCK.md) · [동결 선택값](champion.json)

LIMORE는 50 성인 계통, LICOB는 43 PDO를 38 성인 공여자 단위로 합쳤다. LICOB18은 소스에 없는 완전 지정 구조다. GDSC1 신규 골격 진단은 학습 세포를 공유하며, LIMORE/LICOB에는 연구실의 이전 결과 노출이 있다. Tissue 결과의 support는 각 fold와 screen에서 충분한 약물이 관측된 경우에 한정된다. [최종 점수](RESULTS.md)

## 오프라인 재생

G1의 `nextgen_h123/h2_multiview/study-v2/forecasts/{0,1,2,3}/`에 정확 부모 쿼리 패널용 신뢰된 내부 패키지와 예측이 봉인돼 있다. 패키지는 fit RNA basis와 중앙값, 고정 계수, 각 부모의 보조 자료 가용성과 C1 fallback을 보존한다. 원본 약물 명부 밖에 있는 신규 SMILES의 추론은 이 재생 패키지의 인터페이스 범위 밖이다. 신뢰 출처와 SHA-256을 확인한 패키지를 사용한다. 사용 명령은 등록 G1 Python 환경에서 다음과 같다.

```bash
python -m nextgen_h2.replay /mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_h123/h2_multiview/study-v2 /path/to/new/replay.json
```

네 봉인 파티션의 RNA를 입력해 예측을 다시 계산하며, 네트워크 연결을 차단한다. 허용 최대 오차는 `1e-6`이다. 마스크 공란에서 C1과의 허용 차이는 0이다. 원본 반응자료·개별 예측·적합 계수의 저장 경계는 G1이다. [수치 검증과 재생](psd_and_fallback_receipt.json)
