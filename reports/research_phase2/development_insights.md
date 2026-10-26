# 개발 데이터에서 확인한 통찰과 해석 범위

전체 개발63036cells의 실제 OOF 예측과 raw counts를 연결했다. Source prediction hash는 `development_diagnostics/provenance.json`, 세부 표는 같은 디렉터리에 있다. 아래 분석은 이미 고정된 D1/partial confirmation 후보나 외부 평가 규칙을 바꾸지 않는다.

Geneformer의 pooled macroF1 .94708은 PCA .93483보다 높지만 모든 lineage에서 recall이 높은 것은 아니다. B-cell recall은 Geneformer .87737, PCA .90889, scGPT .88817이다. 세 모델 모두 틀린 비율은 B에서 .07038, 전체 UMI 최상위 사분위에서 .05438이다. UMI 최하위 사분위의 같은 공통 오류율은 .01580이다. 따라서 단순히 저UMI 세포를 제거하면 오류가 해결된다고 결론 낼 수 없다. Class·donor composition이 이 연관을 설명할 수 있다.

Official Geneformer Ensembl mapping과 token vocabulary를 적용해 입력 coverage를 진단한다. B lineage에서 IGKC, IGHG3, IGHG1 등 풍부한 immunoglobulin gene이 vocabulary 밖에 있음이 확인된다. 이는 원 ID를 임의의 다른 gene token에 대응시킬 근거가 아니다. 실제로 B 내부의 vocabulary count-fraction 최하위 사분위 Geneformer recall은 .99482, 최상위는 .81865로 단순한 coverage 감소→오류 증가 관계가 아니다. 각 사분위는579cells다. 이 유전자들의 누락이 모델 간 성능 차이를 일으켰다는 인과 결론은 내리지 않는다. 상위 gene별 raw-count 비율과 B 내부 coverage 사분위의 모델 오류를 별도 표로 제공한다.

Stage는 donor 내부에서 일정하며 단계별 donor 수가1–2명뿐이다. Virus도 donor와 얽히고 일부 환자는 미보고다. 이 개발 데이터만으로 etiology·stage generalization을 입증하지 않는다. Platform·doublet score는 미제공이다. 조직 위치는 **원본 site가 canonical tissue_site로 변환되어 제공된다**. 이전 진단의 site/tissue literal column 탐색은 이 변환을 놓쳤으며, V3에서는 실제 원본71915행 중 개발63036행의 값이 정확히 일치함을 확인했다. 기존 진단 provenance는 이력으로 보존하고 보완 분석을 `tissue_error_analysis/`에 둔다.

조직별 세포 수는 Tumor27497, Normal26725, PVTT5971, Lymph2843이다. Geneformer pooled F1은 각각 .93323/.95300/.90401/.94201이며, PVTT는2명, Lymph는1명뿐이다. Site 효과와 donor·lineage 구성 차이를 분리한 결과가 아니다. Donor×site×lineage support와 TP/FN/FP를 제공하므로 낮은 recall을 전체 pooled 수치 뒤에 숨기지 않는다. 예를 들어 HCC03 Tumor T/NK203cells의 Geneformer recall은 .24631, HCC08 Tumor B77cells는 .40260이다. 개발 결과의 기술적 사례이며 새 selection 근거로 사용하지 않는다.

scGPT1200과 Geneformer4096의 token 제한에 걸릴 가능성은 lineage별로 다르다. 표는 관측 vocabulary gene 수에 근거한 capacity overflow이며 실제 각 random sample의 marker 포함 로그가 아니다. Truncation 때문에 특정 오분류가 발생했다고 단정하지 않는다.

Legacy KG의 선택된128개 pathway는4166개 membership edge이며 일부 pathway의 관측 member 집합이 완전히 같다(max pairwise Jaccard1). 중복되는 요약 특징은 concat 이득이 제한될 수 있다는 후보 설명이다. 이를 원인으로 증명한 실험은 아니다. Size·gene degree를 보존한 random control과 비교한 기존 음성 결과를 보존한다.

실측 genetic response는 train-only response magnitude와 cell support별 평가 및 direct-target 제외 민감도를 제공한다. Full HepG2에서 GEARS DeltaRMSE .13091은 no-change .12373 및 GO/PCA/ridge .10584보다 크다. 이 결과는 해당 control 처리까지 포함한 pipeline 비교다. V3 공통191target 감도에서는 GO/ridge에 full-control 평균 대신 기록된 sampled-control 평균을 넣으면 RMSE .11073→.13050으로 바뀌었다. 단순히 같은 control을 truth와 fixed prediction 양쪽에서 빼는 평가 기준 변경은 RMSE를 바꾸지 않지만 Delta Pearson을 바꾼다. 후자의 높은 Pearson을 새로운 예측력 향상으로 해석하지 않는다. Native pretrained/random의 원 비교는 동일한 sampled-control 입력을 사용하므로 별도로 유지한다. Mouse TCDD는 다른 population·종·조건이다.
