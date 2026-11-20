# 연구 R2 튜토리얼

두 노트북 모두 **아무것도 학습하지 않는다.** 저장된 출력은 서버에서 실제로 실행한 결과다(job 25288).

| 노트북 | 내용 | 필요한 것 |
|---|---|---|
| `01_read_results.ipynb` | source 가상 target과 HCC 결과 표를 읽고 그림으로 본다 | 이 저장소의 `manifests/episodic_residual_r2/`, `predictions/episodic_residual_r2/`의 작은 표, pandas·matplotlib |
| `02_worked_example_16_supports.ipynb` | support 16개에서 query 약물 하나의 ΔRNA를 예측하는 과정을 숫자로 따라간다(f0 → offset → 축소 → 제안법) | 서버의 source 단위 저장소와 동결 checkpoint. HCC 자료는 열지 않고 source 가상 target(HCC515)을 쓴다 |

다시 만들고 실행하려면(서버):

```bash
sbatch experiments/episodic_residual_r2/notebooks.sbatch      # 결과는 assets/public/tutorials/episodic_residual_r2/ 에 생긴다
bash experiments/episodic_residual_r2/sync_public.sh          # 로컬에서 저장소로 복사
```

노트북 02의 예제는 세포주 하나·support 추출 한 번의 결과다. 방법 간 순서는 `reports/episodic_residual_r2/RESULTS.md`의 평균 표로 판단한다.
