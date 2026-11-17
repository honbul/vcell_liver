# R7 결정 기록 (DECISIONS)

각 결정은 실행 **전에** 적는다. 형식: 관측 → 가능한 원인 → 예측 → 변경 하나 → 판정 기준. 실행 후 결과는 같은 항목 아래 "결과"로 덧붙인다(앞부분은 고치지 않는다).

## 0. 시작 전 확인 (2026-11-18)

- 기획서 수신 확인(비공개, Git 미포함): `00_OVERVIEW_AND_GIT_REVIEW.md` 16097 B sha256 `eb565983…be4f9`, `02_R7_EVIDENCE_GATED_MECHANISM.md` 20081 B `51a86118…`; 전체 해시는 `manifests/evidence_gated_r7/plan_receipt.json`.
- R4 브랜치·서버 자산은 수정하지 않는다. R4 unit 파일은 읽기 전용으로 쓴다. R4 test는 이미 한 번 열렸다(job 25374). 따라서 R7 test는 노출된 공개 benchmark의 사전 등록 후속 평가다.
- 계산은 모두 Slurm에서 하고 job 이름은 `r7_`로 시작한다.

## D-0. chemistry backbone 정의 (실행 전, R4 validation 기록 근거)

- **관측(R4 validation 기록, test 아님):** R4 delta-MLP의 validation source ΔRMSE는 0.566(3 seed)으로 train_cell_mean 0.5538, ECFP+X0 Ridge(α=1e4) 0.5469보다 나빴다(`manifests/response_geometry_r4/baselines_validation.tsv`, `runs/mlp_full_*`). best checkpoint는 3000–7000 update에서 나왔다.
- **가능한 원인(가설):** 원시 Level 3 X0 입력의 척도와 세포 평균 이동을 MLP가 직접 맞춰야 하는 출력 계약. 확인하지 않았다.
- **R7에 주는 위험:** 기전 보정 r은 세포 상태를 입력으로 받는다. 따라서 backbone이 세포 평균을 못 맞추면, r이 "지식"과 무관하게 세포 평균 차이를 메워 점수가 오를 수 있다. 그러면 정보 가치가 과대평가된다.
- **변경(모든 arm 공통, 2×2 수정 횟수에 포함하지 않는 설계 정의):** backbone = `m_c + MLP([ECFP, 표준화 X0])`. R4 MLP 폭(512-512)은 그대로 두고, 출력층은 0으로 초기화한다. m_c와 X0 표준화는 training unit만으로 적합한다.
- **판정:** 이것은 성능 주장이 아니다. pilot의 chem_only가 train_cell_mean보다 나쁘면 그대로 보고하고, 이 정의를 바꾸지 않는다.

## D-1. XPert 비교군 실행 방식 (실행 전)

- 저자 가중치는 L1000 전체로 학습되어 R7 test 약물 반응이 들어 있을 수 있다. 그래서 쓰지 않고, R4 split으로 처음부터 재학습한다.
- 공개 Zenodo 자료(CC-BY-4.0)에서 UniMol 원자 feature와 HG drug embedding을 쓴다. 구조(InChIKey)로 XPert 약물 색인에 연결하고, 연결되지 않는 약물은 UniMol을 생성하고 HG는 0으로 둔다(flag).
- HG embedding 사전학습에 반응 유래 관계가 쓰였는지는 저장소에서 확인할 수 없다(기본 edge PPI·DTI·DDS). **미확인**으로 기록한다.
- `flash_attn`은 같은 수식의 torch SDPA shim으로 바꾼다. 원 코드의 flash 경로는 원자 padding mask를 쓰지 않으며, 그 동작은 그대로 둔다.
- 원 학습 스크립트는 마지막에 `test` split을 평가한다. 그래서 학습 h5ad의 `test` 칸에는 validation 복사본만 넣는다. R7 test unit은 학습 job에 들어가지 않는다.
