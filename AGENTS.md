# Agent instructions

먼저 `docs/AGENT_EXPERIMENT_PLAN.md`와 `docs/MODEL_COMPARISON.md`를 읽고 실제 저장소 상태를 확인한다. 이 파일보다 사용자의 후속 명시 지시가 우선한다.

## 목표

공개 HCC scRNA-seq에서 scGPT와 Geneformer-V2-104M을 설치·실행·비교하고, KG 추가 실험과 사용자가 직접 따라 할 수 있는 한국어 튜토리얼을 완성한다. 계획/설치만으로 완료하지 않는다.

## 현재 문서의 상태

2026-10-26 초기 문서는 문헌·공식 자산 조사와 실험 기획이다. 모델 실행 결과나 측정된 GPU 요구량은 아직 없다. 장비를 먼저 조사하고 실제 결과를 보고한다.

## 핵심 규칙

- scGPT부터 end-to-end로 연결하고, Geneformer를 같은 patient split에서 비교한다.
- GSE149614의 공개 count/metadata를 우선 사용하며 원본 열 구조는 직접 확인한다.
- 환자 단위 split, train-only preprocessing, PCA baseline을 유지한다.
- Geneformer embedding shift와 실측 perturbation expression prediction을 구분한다.
- GEARS는 별도 실측 genetic perturbation 데이터가 필요한 조건부 후속이다.
- 기존 KG가 없어도 Reactome으로 최소 연계를 진행하되 provenance를 기록한다.
- 자원·실패·unknown을 숨기지 않고, mock 결과를 실측값으로 표시하지 않는다.
- 공개 Git에 raw count, weights, 비밀값, 미출판 내부 문서를 올리지 않는다. source/version/hash와 작은 결과 요약을 남긴다.
- 기존 환경과 진행 중 GPU 작업을 파괴하지 않는다. paid service를 자동 사용하지 않는다.
- 필요한 데이터 정합/누수 검증은 수행하되 불필요한 보안 감사·전면 재설계·무제한 HPO로 범위를 확대하지 않는다.
- 단계별 commit/push 후 실제 원격 반영을 확인한다. force push 금지.

현재 문서의 스크립트·노트북 이름은 구현할 목표 인터페이스다. 존재 여부를 확인하지 않고 실행 가능한 것으로 가정하지 않는다.

## 연구 확장 V2

후속 연구는 `docs/RESEARCH_EXPERIMENT_PLAN_V2.md`를 우선한다. 위 초기 튜토리얼 지침과 기존 결과는 이력으로 보존한다. HCC02/HCC09는 legacy 회귀용이며 새 blind test로 취급하지 않는다. 새 실험은 research branch와 별도 immutable run ID를 사용한다.
