# 공개 비교 모델 실행 계약과 출처

| 비교군 | 코드 및 라이선스 | 입력·적합·평가 적격성 |
|---|---|---|
| PRnet | [공식 저장소](https://github.com/Perturbation-Response-Prediction/PRnet/tree/f19174bde3ed2633f54c7831799cc38c4ffc7a0d), Apache 2.0 | Sci-Plex 구현의 PGM, FCFP4 1,024비트와 `log10(1+µM)` 배율, Gaussian NLL. Tahoe의 같은 역할 DMSO 웰 평균에서 처리 웰 평균을 예측하는 적응이다. 2,000 유전자에 새로 적합하며 100 epochs, Adam 0.001, seed 42, 32 latent-noise 예측 평균을 고정한다. 원 논문 단일세포 분포보다 평가 출력이 제한된다. |
| State ST-Tahoe 공개 가중치 | [공식 X_hvg 체크포인트 카드](https://huggingface.co/arcinstitute/ST-HVG-Tahoe), [State Model 비상업적 라이선스](https://huggingface.co/arcinstitute/ST-HVG-Tahoe/blob/main/MODEL_LICENSE.md) | 공개 split은 세포 맥락 일반화 중심이며, 평가 parent를 모든 Tahoe 세포주와 용량에서 제외했다는 증명이 없다. 모델 입력은 학습 drug×dose 범주 코드다. 공개 체크포인트를 이 연구의 처음 보는 parent 판정에 선별하거나 재사용하지 않는다. 카드의 CellEval은 별도 계약이므로 대응 RMSE로 변환하지 않는다. |
| State Morgan 적응 | [State 공식 코드](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add), [코드 CC BY-NC-SA 4.0](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/LICENSE) | 공개 State Transition GPT-2 set 모델에 고정 ECFP4 구조 입력과 같은 역할 DMSO에서 만든 8개 소집단 토큰을 넣고, 모델 출력의 평균을 처리 웰 평균에 fit한다. 실측 세포분포 대신 웰 평균 MSE, 256 hidden, 4 transformer 층, seed 42, AdamW 0.0003, 40 epochs를 사용한다. 참조 VCI decoder import를 해당 decoder 사용 때까지 미루는 패치는 경로 의존성만 바꾼다. 가중치는 무작위 초기화이며 공식 ST-Tahoe 점수와 독립된 적응 행이다. |
| CPA 기본형 | [공식 CPA 구현](https://github.com/theislab/cpa/tree/fbd7c0250edc23eff003a10c99655579c53afd63), BSD 3-Clause | 범주형 약물 embedding, dose, 세포주 등 공변량, 발현 입력을 사용한다. 해당 약물의 효과 벡터를 반응 fit에서 얻는 기본형은 새 parent의 학습된 효과를 제공할 수 없다. 외부 구조 embedding을 쓰는 형식은 MoE 이전 chemCPA 경로에서 비교한다. 기본형은 적격 경쟁군에서 제외한다. [공식 State 재현 CPA](https://github.com/ArcInstitute/state-reproduce/tree/main/baselines) |

PRnet과 State Morgan 적응은 공개 처리 반응 checkpoint를 쓰지 않는다. 각 Tahoe dose와 외부 fold에서 fit parent 웰만 학습한다. 채점 가능 여부 및 실행 영수증은 T2 결과와 연결한다. 위 설정은 외부 결과를 확인하기 전에 기록했다.
