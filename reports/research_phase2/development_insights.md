# 개발 데이터에서 확인한 통찰과 해석 범위

전체 개발63036cells의 실제 OOF 예측과 raw counts를 연결했다. Source prediction hash는 `development_diagnostics/provenance.json`, 세부 표는 같은 디렉터리에 있다. 아래 분석은 이미 고정된 D1/partial confirmation 후보나 외부 평가 규칙을 바꾸지 않는다.

Geneformer의 pooled macroF1 .94708은 PCA .93483보다 높지만 모든 lineage에서 recall이 높은 것은 아니다. B-cell recall은 Geneformer .87737, PCA .90889, scGPT .88817이다. 세 모델 모두 틀린 비율은 B에서 .07038, 전체 UMI 최상위 사분위에서 .05438이다. UMI 최하위 사분위의 같은 공통 오류율은 .01580이다. 따라서 단순히 저UMI 세포를 제거하면 오류가 해결된다고 결론 낼 수 없다. Class·donor composition이 이 연관을 설명할 수 있다.

Official Geneformer Ensembl mapping과 token vocabulary를 적용해 입력 coverage를 진단한다. B lineage에서 IGKC, IGHG3, IGHG1 등 풍부한 immunoglobulin gene이 vocabulary 밖에 있음이 확인된다. 이는 원 ID를 임의의 다른 gene token에 대응시킬 근거가 아니다. Coverage와 오류의 관련성은 descriptive이며, 이 유전자들의 누락이 모델 간 성능 차이를 일으켰다는 인과 결론은 내리지 않는다. 상위 gene별 raw-count 비율과 B 내부 coverage 사분위의 모델 오류를 별도 표로 제공한다.

Stage는 donor 내부에서 일정하며 단계별 donor 수가1–2명뿐이다. Virus도 donor와 얽히고 일부 환자는 미보고다. 이 개발 데이터만으로 etiology·stage generalization을 입증하지 않는다. 명시적 platform·doublet score·site/tissue 필드가 없으면 unknown으로 남기고 sample 이름에서 임의로 만들지 않았다.

scGPT1200과 Geneformer4096의 token 제한에 걸릴 가능성은 lineage별로 다르다. 표는 관측 vocabulary gene 수에 근거한 capacity overflow이며 실제 각 random sample의 marker 포함 로그가 아니다. Truncation 때문에 특정 오분류가 발생했다고 단정하지 않는다.

Legacy KG의 선택된128개 pathway는4166개 membership edge이며 일부 pathway의 관측 member 집합이 완전히 같다(max pairwise Jaccard1). 중복되는 요약 특징은 concat 이득이 제한될 수 있다는 후보 설명이다. 이를 원인으로 증명한 실험은 아니다. Size·gene degree를 보존한 random control과 비교한 기존 음성 결과를 보존한다.

실측 genetic response는 train-only response magnitude와 cell support별 평가 및 direct-target 제외 민감도를 제공한다. Full HepG2에서 GEARS DeltaRMSE .13091은 no-change .12373 및 GO/PCA/ridge .10584보다 크다. 이 결과는 복잡한 모델을 추가했다는 이유만으로 intervention 성능이 좋아진다고 볼 수 없음을 보여준다. Native foundation 비교와 mouse TCDD 결과는 각각 다른 population·종·조건으로 분리한다.
