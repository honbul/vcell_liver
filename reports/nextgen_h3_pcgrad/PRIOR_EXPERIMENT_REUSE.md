# 이전 자산 재사용

| 근거 | 이번 적용 |
|---|---|
| `5ec8ed231ee88c6ae578d0e1cb30d7591c792392` | 새 H3 worktree의 기준 커밋. 역사적 blocked lineage × held screen 행, assay 지원과 약물 anchor 규약을 따른다. |
| `9ed87a1a50e301dd4c90078e3d1bd32103f27468`의 `v0plus_v2/lean.py` | RNA rank-z·ECFP4 RNA-only bilinear 형태 및 β=.45 SmoothL1, 약물당 최소 관측 8개의 Pearson, 온도 .17 ListNet을 재사용한다. H3a의 독립변수는 같은 H3 목적의 기울기 결합 방법이다. |
| `1912b5d100a36778dfef3650b96707178adb6885` | 정정 68약물 학습 SHA 및 단일 행 등록부. 순위 이득 NDCG, 동률 기대 Precision, 약물 내 Spearman 산식을 고정해서 가져왔다. LIMORE의 정정 구조 커널과 Pairwise Top-5는 외부 기술 확인의 등록 비교군이다. |
| 기존 `NEURAL_INTERACTION_TRANSFER` | 관측된 지지에서 가산 성분을 사영하고 적합 계통 기준으로 상호작용을 중심화하는 실험 근거다. H3b는 해당 사영 정답과 공통 anchor를 모델 내부의 α·δ 지도학습에 쓴다. |
| 기존 확장 패널 동결본 `expanded-panel-81119c8` | 9개 같은 행에 대한 정정 구조 커널(벌점 4), 약물별 Ridge, 약물 평균을 SHA 및 정확 키 대조 후 사용한다. Inner 구조 커널 대조는 각 내부 적합 행으로 다시 학습했다. |

H3 모델의 발현은 적합 세포의 변동성 상위 512개와 fit-only 정규화를 사용한다. 화학은 정확 등록 구조의 반경 2·512비트 지문이다. H3는 두 원래 입력만 공통으로 사용한다. 기존 전체 구조 커널은 고정 비교군으로 RNA·경로/TF·분자 보조 참조를 원래 설정대로 사용한다. R1의 무 dropout·무 ID embedding bilinear 구현과 G0/G1 적합 재현성은 별도 불변식 테스트로 확인했다. [사전 잠금](PLAN_LOCK.md) · [이전 신경망 상호작용 명세](../vc_build_v1/v0plus/NEURAL_INTERACTION_TRANSFER_PROTOCOL.md) · [H3 목적함수](../../src/h3_pcgrad/train.py)

PCGrad는 음의 내적에만 원 태스크 기울기에 대한 투영을 적용한다. Normalized PCGrad는 투영 전 단위 크기로 바꾼다. GradNorm은 학습률 균형 기준으로 가중치를 갱신하고, CAGrad는 세 목적의 최소 국소 개선을 고려한 simplex 해를 근사한다. [PCGrad 원 논문](https://arxiv.org/abs/2001.06782) · [GradNorm 원 논문](https://proceedings.mlr.press/v80/chen18a.html) · [CAGrad 원 논문](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
