# T1–T3 동일 assay 전이 보정 DEV 명세

GSE92742의 X 벤치마크에서 고정한 493개 parent, 실측 10 µM·6 h, 978 L1000 landmark를 사용한다. 최종 평가 세포주는 HEPG2며 비간계열 A375·A549·HT29·MCF7의 같은 parent 반응을 donor로 쓴다. R18 보호 행·별칭·donor는 감사한 X 명부에서 제외되어 있다. 기존 3개 parent fold와 물리 fit/select/score DMSO의 세 역할을 유지한다. Tahoe 외부 T2·T3의 물리 웰과 assay 지원 판정은 유지한다.

- `T0`은 각 비간계열 세포주의 fit DMSO 대비 반복 웰 평균을 세포주 균등 가중한 전체 delta다. 이 gain 1 정의는 X3에서 배율을 선택한 `B_cell`과 다르다. `B0`은 0 대비다.
- `T1`은 개발 source OOF의 delta 예측 및 select DMSO 대비 실제 처리 웰에 에너지 최소제곱 비음수 gain을 적합한다. 범위는 [0,2], 에너지 0의 gain은 0이다. `T2`는 fit DMSO에서 추정한 유전자별 분산비의 shrink·floor·clip으로 T0를 변환한다. `T3`는 같은 T2 규칙에 source OOF를 적용하고 그 좌표에서 T1의 gain 공식으로 다시 적합한다.
- source OOF는 비간계열 하나씩을 pseudo-target으로 하고 남은 세 비간계열을 donor로 삼는다. 각 source 검증 fold의 pseudo-target domain과 해당 parent를 통째로 뺀 OOF 예측으로 gain을 내부 적합한다. 즉 내부 적합에서는 두 비간계열 donor가 남는다. 3개 내부 CRC32 parent fold를 사용한다. 외부 held parent의 정답과 HEPG2의 모든 처리값은 보정 적합·설정 선택에서 제외한다. 최종 gain만 외부 허용 source OOF 4개 domain 전체에서 다시 적합한다. 외부 held parent의 비간계열 처리값은 B 추론 입력이다.
- 분산은 같은 Level 3 물리 웰 좌표의 fit DMSO만 사용하고, cell 평균을 빼고 여러 source cell의 웰을 pooled sample variance로 합친다. target이거나 내부 pseudo-target인 각 세포주에 독립 fit control 12웰 및 3plate 이상을 요구한다. 이 기준이 안 되면 규정상 T2/T3의 지원을 보류한다. 분산비의 shrink `0.25,0.5`, floor `0.1,0.5` × source 유전자 분산의 중앙값(최소 1e−8), clip `log(1.25),log(1.5)`에서 고른다. 내부 검증에서 T3의 R17 공동 손실을 source domain×parent fold별로 계산해 균등 평균하고, 소수 9자리 손실과 작은 shrink·floor·clip 순으로 결정한다. T2에도 이 규칙을 공유한다.
- 별도의 Slurm 예측 → 동결 → 채점으로 진행한다. HEPG2 score DMSO로 뺀 각 실제 물리 처리 웰에서 parent-macro RMSE와 pooled R²를 구한다. 짝 parent 10,000회 재표집의 99% 구간을 쓴다. `T1−T0`, `T2−T0/T1`, `T3−T1/T2`와 B0 대비 점수를 보고한다. 주 비교 T3−T1은 RMSE 개선 하한 >0 및 R² 개선 하한 >0이면 통과한다. T2와 T3가 모두 T1 대비 상대 RMSE 개선 상한 <0.5%와 R² 개선 상한 <0.5%p라면 이 동일 assay diagonal 규약을 종료한다. T2만 같은 지표 기준에서 개선이 확인되면 보조 결과로 기록한다. 나머지는 `INCONCLUSIVE`다.

외부 HEPG2 query는 B donor 네 세포주를 사용하며 source pseudo-target의 donor 수는 다르다. 동일한 전이 종류에서 보정 규칙을 평가하며 새로운 코호트의 독립 확인으로 해석하지 않는다.
