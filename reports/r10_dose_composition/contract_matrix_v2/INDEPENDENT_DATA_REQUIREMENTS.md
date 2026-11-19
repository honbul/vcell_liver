# R10-E3 독립 자료 요구 명세 (미실행 — 자료 대기)

상태: **미실행.** 현재 저장소·서버에는 Sci-Plex3 24 h(A549·K562·MCF7, 두 replicate) 외에 용량별 약물 처리 단일세포 자료가 없다. 근거: private plan 2026-11-19 §7.3. 이 문서는 자료가 확보되면 바로 입력 계약을 고정할 수 있도록 요구 조건만 정리한다. 새 자료를 내려받거나 등록·유료 서비스를 사용하지 않았다.

## 1. 목표를 둘로 나눈다

| 목표 | 질문 | 바꾸는 것 | 같이 바꾸면 안 되는 것 |
|---|---|---|---|
| (a) 새 배양/실험 반복 | 기존 세포주의 새 batch로 E1의 선택 기준선·후보가 이동하는가 | culture batch, plate, sequencing run | 세포주 |
| (b) 간세포암 모델 | HCC 유래 세포/환자 모델로 이동하는가 | 세포 계통 | batch 설계를 (a)와 분리 |

새 세포주와 새 batch를 동시에 바꾼 단일 결과로 둘의 기여를 분리하지 않는다.

## 2. 최소 데이터 계약

- culture 날짜·batch provenance, well/plate/sequencing 배치 ID.
- drug ID(구조 포함)·dose·24 h 조건·독립 생물학적 replicate 표시.
- 세포 QC 전 원 count 또는 동등한 원자료, hash/demultiplexing 기준.
- 각 plate의 vehicle well(현재 계약은 endpoint replicate의 vehicle만 허용; query batch의 별도 vehicle은 추가 관측 비용으로 다른 계약).
- 저·중간·고 용량의 대응(현재 episode: 10–100–1000 nM, 100–1000–10000 nM).
- 같은 약물의 용량을 plate와 완전히 결합시키지 않는 균형·무작위 배치.
- 가능하면 3개 이상의 독립 배양 batch(between-batch 변동 추정용; 검정력 보장 숫자가 아니며, 규모는 pilot의 약물별 효과·반복 변동으로 산정).
- HCC는 진단·유래가 확인된 간세포암 모델일 것(이름 접두만으로 판단하지 않음). R9 환자 RNA 주석 자료는 용량별 처리 정답이 없으므로 대체 자료가 아니다.

## 3. 평가 전에 고정할 것

- representation/atlas: 기존 train-only PCA/GMM의 transfer인지, 신규 line 적응(재적합)인지.
- 관측 허용: 양 endpoint·endpoint vehicle은 입력, 중간 용량 RNA·구성비·회수 세포 수는 채점까지 보류.
- batch/replicate ID 사용 여부(사용한다면 D00·t*·W에도 같은 정보 제공).
- 기준선과 후보: v2 E1a에서 이미 정한 arm(D00, t*_same/t*_cross, D10/D01/D11, W)을 그대로 쓰고 새 자료로 다시 고르지 않는다.
- primary metric(재판독 actual 구성비 TV)·대비·실용 기준(0.005)·다중성 처리.
- 과제 명칭: 양 endpoint를 관측한 새 약물의 **부분 관측 보간**이다. drug zero-shot이라 부르지 않는다.

## 4. 실행을 여는 조건

위 2의 자료가 확보되고 3의 명세를 commit(freeze)한 뒤에만 실행한다. 그 전까지 E3는 이 문서 상태로 둔다.
