# Tahoe 기존 VC 비교 세부 실행

R12/R13 화합물 3-fold를 0.05, 0.5, 5 µM와 최소 세포 수 20, 200의 각 조합에 적용한다. 각 외부 fold에서 약물 반응 학습 및 내부 모델 선택은 다른 두 fold로 제한한다. 훈련 품질 표와 예측 영수증은 모델별로 보존한다. 출력은 공통 2,000 유전자에 맞춘 웰·세포주 평균 반응이다.

| Arm | 기존 실행 및 고정 입력 | 이번 적합 |
|---|---|---|
| XPert_morgan | R7의 Morgan/zero-HG arm, [공식 XPert](https://github.com/GSanShui/XPert/tree/d53ff497e465692c80fac7b899e3cc5cc8e7bfed) MIT | XPertNet을 Tahoe 공통 유전자 축으로 맞췄다. R7 PPI landmark 임베딩을 Ensembl로 매칭하고 나머지 유전자의 prior를 0으로 초기화한다. 1024-bit Morgan과 own-role DMSO 입력, R7 L0의 초기 복합 손실, Adam 0.004, AMP, batch 24를 쓴다. 3/8/16 epoch 중 두 내부 fold 평균 대비 RMSE 최저를 고른다. |
| chemCPA_P0 | MoE 이전 P0의 [공식 chemCPA ComPert](https://github.com/theislab/chemCPA/tree/43e830eb0958c54e4aa64442c17ec0fed19b3f15) MIT | Gaussian NLL 재구성, 약물·세포주 adversary, amortized doser 및 공개 P0 기본값을 쓴다. 기존 단일세포 입력 대신 공통 웰 평균과 fit DMSO 웰 평균을 사용한다. 원 구현의 RDKit2D 기술자에 fit-only 분산 필터·표준화를 적용한다. 5/15/30 epoch 내부 선택이다. |
| TranSiGen_Tahoe | B1의 [공식 TranSiGen ECFP4 구조](https://github.com/myzheng-SIMM/TranSiGen/tree/8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a) MIT | 978 landmark 대신 2,000개 공통 유전자를 쓰도록 입력·출력 크기를 변경해 Tahoe fit에서 초기화부터 적합한다. 원 VAE 복합 손실 및 posterior-mean 추론, Adam 0.0003, batch 128이다. 1/5/15/30 epoch 내부 선택이다. CMap 및 B1 가중치는 이 실행의 fit에 사용하지 않았다. |
| ChemBERTa_R11 | R11의 [공개 ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/tree/66b895cab8adebea0cb59a8effa66b2020f204ca) CLS 벡터, checkpoint 카드 라이선스 미표기 | 동일 CLS32 × control PCA16 interaction Ridge를 fold마다 α=1000으로 적합한다. 체크포인트는 SMILES에서 분자 기술자를 예측하는 학습을 거쳤다. Tahoe 발현 반응은 현재 fit에서만 사용한다. |
| FiLM_R12, Concat_R12 | R12 A5/A4의 동일 ECFP PCA32 및 control PCA16 입력 | R12 residual MSE와 모델 구조를 재사용한다. 학습률 0.0003/0.001 및 1/5/10/20/40 epoch 내부 검증을 거친다. |
| scGPT_C7 | R12 C7 [공식 scGPT](https://github.com/bowang-lab/scGPT/tree/cebd6fae655b9c585a4807daa3ac31bb764f06b4) MIT, whole-human checkpoint | DMSO 세포만 임베딩한 16차원 fold-fit context를 R12 concat에 넣는다. 0.5 µM 자료에 쓰인 8웰의 임베딩을 재사용하고, 나머지 필요한 18 DMSO 웰을 같은 역할·세포 표본 계약으로 추가한다. |
| UCE_C8 | R12 C8 [공식 UCE](https://github.com/snap-stanford/UCE/tree/9c416007be15ad6753dc84af4468c1dc10421ab9) MIT, 공개 4-layer checkpoint | 같은 물리 DMSO 웰을 UCE로 임베딩한다. PCA16 context에 같은 R12 concat과 선택 절차를 적용한다. |

R7 XPert 및 B1 TranSiGen이 쓰던 원래 입력 유전자와 Tahoe 공통 유전자 축이 다르다. 과거 chemCPA의 단일세포 예측은 웰·분할·패널이 현재 실험과 다르다. 새 비교에서 세 방법은 공통 웰 표적에 대한 구조 적응이다. scGPT와 UCE는 DMSO 맥락으로 사용하는 비교군이다. State, PRnet, CPA의 공식 코드와 조건 지원은 [별도 공개 모델 기록](PUBLIC_COMPARATORS.md)에 있다.

모든 비교는 [공통 사전 명세](PROTOCOL.md)의 동결·채점 방식을 적용한다. 가장 좋은 기존 비교군은 99% 대응 재표집마다 새로 정해 최저 RMSE를 사용한다. 출처별 scGPT와 UCE 체크포인트, R11 ChemBERTa 임베딩, XPert landmark PPI, 훈련 설정 및 예측 파일의 SHA-256을 보존한다.
