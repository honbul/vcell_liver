# 연구 R1 튜토리얼

세 노트북 모두 실행 결과가 들어 있어, 실행하지 않고 읽기만 해도 됩니다. 연구 모델을 다시 학습하지 않습니다.

| 노트북 | 내용 | 실행에 필요한 것 |
|---|---|---|
| [01_inspect_results.ipynb](01_inspect_results.ipynb) | 저장소에 있는 결과표만 읽어 모든 모델의 점수, 쌍 차이, 실제 대 무작위 prior, 선택된 설정, 외부(LCCL) 결과를 확인 | pandas, matplotlib. 자료·서버 불필요 |
| [02_pathway_and_rest_one_cell.ipynb](02_pathway_and_rest_one_cell.ipynb) | 실제 세포 하나의 RNA를 경로 부분과 나머지로 나누기, 같은 약물에서 hard / soft / prior 없음의 예측 비교, 규제 강도에 따른 변화 | 서버의 입력 캐시(`experiments/softprior_multitask_r1/r0_build.sbatch`가 만듦). 수 초 |
| [03_penalty_demo_synthetic.ipynb](03_penalty_demo_synthetic.ipynb) | 작은 가상 자료에서 prior 강도(η)와 약물 공유 강도(τ)를 바꿔 보기. 신호가 경로 안에 있을 때와 밖에 있을 때, 진짜 경로와 섞은 경로 | numpy, matplotlib. 자료 불필요. 1분 이내 |

## 읽는 순서

1. 처음이면 [BEGINNER_SUMMARY.md](../../reports/softprior_multitask_r1/BEGINNER_SUMMARY.md)를 먼저 읽습니다.
2. 03으로 "규제를 방향마다 다르게 준다"는 것이 무슨 뜻인지 가상 자료에서 봅니다.
3. 02로 실제 세포에서 경로 부분이 얼마나 작은지 봅니다.
4. 01로 최종 결과표를 확인합니다.

## 다시 만들기

노트북은 `_build_notebooks.py`가 만들고, `experiments/softprior_multitask_r1/finalize.sbatch`가 서버에서 실행해 결과를 넣었습니다.

```bash
python tutorials/softprior_multitask_r1/_build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace tutorials/softprior_multitask_r1/03_penalty_demo_synthetic.ipynb
```

## 주의

- 02의 한 약물·한 fold 숫자는 예시입니다. 결론은 01의 전체 표에서만 읽습니다.
- 02는 예시 약물 하나에 대해 보류 세포 16개의 관측 AA와 예측을 나란히 출력합니다. RNA 값은 출력하지 않습니다.
- 03의 가상 자료는 방법이 어떻게 움직이는지 보여 줄 뿐, 간암 자료의 결과를 대신하지 않습니다.
