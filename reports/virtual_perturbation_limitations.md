# 가상 삭제 결과의 한계

**실측 섭동 검증 전 탐색.** 출력은 Geneformer V2의 rank token을 하나씩 삭제한 전후 CLS embedding cosine distance(1−cosine)다. raw expression을 0으로 만든 실측 knockout이 아니며 post-perturbation expression, logFC, 생존율, 약효를 예측하지 않는다.

Train hepatocyte만 최대 100세포 선택, ALB(계열 marker)와 HNF4A(hepatic TF)를 사전 지정. 실제 발현·vocabulary를 확인하고 각 조작에서 token 존재를 기록한다. 미존재 token은 변화가 없으며 실제 삭제된 결과와 구분한다. 동일 세포 no-op과 train expression 수준이 가까운 무작위 gene 대조(seed42)를 포함한다. 다유전자 공동 삭제를 하지 않음. 공식 delete_indices와 공식 CLS extraction을 사용하되 변화 크기만 보고하며 전체 InSilicoPerturber 상태전이 protocol을 재현한 결과는 아니다.

신뢰할 정상/질환 상태 라벨과 목표 reference가 없어 방향성, 정상화, HCC 치료 표적 우열을 주장하지 않는다. cell type head confidence를 정상화로 해석하지 않는다. embedding 변화가 크다고 생물학적 영향이 크다는 뜻이 아니다. 실제 genetic perturbation 데이터와 독립 실험으로 확인하기 전 가설 생성용이다.
