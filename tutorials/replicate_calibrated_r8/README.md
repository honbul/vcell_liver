# 연구 R8 튜토리얼

`01_common_vs_per_drug_correction.ipynb` — source 세포주 하나(HCC515, 확인 family)를 "새 세포주"로 보고, 약물 16개의 측정에서

1. 공통 보정(support 잔차의 평균을 줄여서 모든 query에 더하는 것)과
2. 약물별 보정(닮은 support의 잔차를 빌려 오는 것: 닫힌 식 C1, 학습한 B00)

이 각각 예측을 얼마나 바꾸고 오차를 얼마나 줄이는지를 숫자로 나눠 보여 준다. 동일 가중과 관측 공분산 가중의 차이, 확인 family 전체의 요약 표도 읽는다.

- HCC 발현값은 읽지 않는다. 아무것도 학습하지 않는다(커밋된 선택 manifest와 서버의 최종 checkpoint를 읽는다).
- 실행: 서버 작업 위치에서 `sbatch experiments/replicate_calibrated_r8/notebooks.sbatch` (CPU, 약 30초). 결과 노트북은 `assets/public/tutorials/`에 쓰이고 `experiments/replicate_calibrated_r8/sync_public.sh`로 저장소에 복사한다.
- 노트북 원본은 `build_notebook.py`가 만든다. 저장소에 있는 `.ipynb`는 job 25689에서 실행한 출력이 들어 있는 사본이다.
- 발현 배열과 checkpoint는 저장소에 없으므로 서버 밖에서는 출력만 읽을 수 있다.
