# P2 Tahoe의 외부 LINCS 반응 사전 지식

P2는 외부 LINCS L1000에서 먼저 측정한 약물의 반응을 입력으로 사용한다. Tahoe의 같은 약물 발현은 전체 용량의 외부 적합과 모델 선택에서 제외했다. 외부 실측이 없는 별도 조건에는 구조로 예측한 LINCS 잠재 좌표 또는 구조만 사용한다. [사전 명세](PROTOCOL.md)

## 이전 실험과 적용 범위

R6은 지원 조건과 평가 조건의 약물·배치 일치가 반복 기반 판정에 미치는 영향을 다뤘다. R6의 외부 확인 후보 조사에서는 다른 실험법·세포 계통에서 얻은 LINCS를 같은 조건의 반복으로 채택하지 않았다. R8은 CMap LINCS 2020 Level 3 약물–세포주 반응을 참조로 옮기는 조건과 같은 약물의 원본 참조를 가린 조건을 비교했다. B1은 LINCS 2020의 비표적 161개 세포주에서 TranSiGen을 사전학습한 뒤 HCC 세포주 세 계열에서 적응했다. 이 모델 추론은 구조 지문과 대조군 발현을 입력으로 받았다. R13의 LINCS 개발 실험은 Tahoe 카탈로그와 약물이 겹치지 않도록 하여 출력 공간 또는 학습 prior를 만들었다. 이번에는 **같은 약물의 LINCS Level 5 서명**을 외부에서 주어진 조회 입력으로 명시하고, 여러 Tahoe 모델에 동일한 위치 추정값을 사용했다. [R6 후보 조사](../r6_sensitivity_correction_v3/r6_pause_and_resume.md) · [R8 관측 환경 전이](../r8_transfer_v2/RESULTS.md) · [B1 TranSiGen](../transigen_adaptation_b1/RESULTS.md) · [R13 결과](../r13_optimization/RESULTS.md)

## 서명과 모델

- GSE92742와 GSE70138의 원본 Level 5 MODZ에서 24 h, 정량 µM가 있는 단독 약물의 공통 측정 landmark 978개를 사용했다. 처리 이름·화학 식별 정보는 해당 GEO release의 perturbagen 메타데이터를 조회했다. 탈염한 InChIKey·구조 일치를 우선하며 이름 또는 CID는 고유하고 확인 가능한 구조와 충돌이 없는 경우에 허용했다. [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) · [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138)
- 약물별로 각 벡터를 landmark 축에서 평균 중심화하고 L2 정규화했다. 같은 연구·세포주·용량을 먼저 블록 평균하고, 용량·세포주·연구를 순서대로 균등 가중해 방향을 다시 정규화했다. 사전 지식 후보는 24 h 전체 세포주의 10 µM 근처(5–20 µM에서 로그 거리가 가장 가까운 용량), 전체 세포주의 모든 정량 용량, HepG2·PHH 세포의 10 µM 근처다. 보유·미보유 등록은 전체 세포주·모든 정량 용량의 실측 지원으로 고정했다.
- 학습 Tahoe 정답에서 공통 trim 위치를 빼고 pool 대비를 계산한 다음, parent의 반응을 행 가중으로 평균했다. 구조는 parent마다 이름순 첫 화학 엔티티의 ECFP4를 사용했다. 다섯 가지 고정 모델 계열은 Tanimoto Ridge, LINCS cosine Ridge, 두 커널의 선형 혼합, LINCS PCA16/32 직접 Ridge, 그리고 구조에서 PCA16 LINCS 좌표를 예측해 미보유 약물을 메우는 혼합 Ridge다. 마지막 후보는 구조→서명→반응의 두 단계 경로를 별도로 검증하며, 공개 VAE 가중치를 가져오지 않았다. PCA와 구조→서명 사상은 fit parent만 사용했다.
- 모든 약물 모델은 P1의 동일한 trim 위치 예측에 더한다. Alpha와 배율, 실측 LINCS preset은 내부 두 fold의 평균 SSE/SST로 골랐다. 바깥 예측의 절대 수준에는 동결한 T1 B_abs를 사용했다. 동결 지표는 전체 및 고정 실측 보유·미보유의 같은 행에서 계산한다.
