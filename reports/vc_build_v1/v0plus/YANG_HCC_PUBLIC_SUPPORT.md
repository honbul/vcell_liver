# Yang 공개 성인 HCC 오가노이드 외부 평가 지원

저자 발현 자료 432개와 발행사 S6의 약물 스크린 PDO 376개를 대조했다. 기저 종양 PDO의 RNA·반응 ID는 90명·255개에서 일치했다. S1 병리와 나이로 제한한 성인 HCC는 **76명·223개 PDO**, 원천 RNA와 공통 유전자는 17,332개다. 여러 종양 지역은 `CaseID`로 묶었다. 기존 VC 학습 코어에 Yang 연구의 PDO는 들어 있지 않다. 다른 외부 코호트와 동일인을 대조할 수 있는 공통 환자 ID는 없다. [등록 명세](YANG_HCC_REGISTRATION_PROTOCOL.md) · [지원과 해시](yang_hcc_registration_receipt.json)

| S6 약물 | 학습에 등록된 정확한 parent 지원 | 이번 등록의 판단 |
|---|---|---|
| Lenvatinib | CTRPv2 690계통 모델, PRISM20Q2 621모델 | 정정 68약물 코어에 포함 |
| Sorafenib | CTRPv2 688모델, GDSC2 636모델 | 정정 코어에 포함 |
| Regorafenib | CTRPv2 666모델, PRISM20Q2 452모델 | 정정 코어에 포함 |
| Ivosidenib | PRISM20Q2 557모델 | 1,129약물 확장에 포함 |
| Apatinib | 기존 등록에는 없고, PRISM20Q2 원본에서 443모델·성인 HCC 14모델 확인 | 정확한 추가 학습 후보. 기존 동결 코어에 미포함 |
| Pemigatinib | PRISM24Q2 `REP300` 2.5 µM, 공개 처리 메타데이터 6웰 | 단일 농도 후보. 확정 AUC 학습 지원 및 제제 구조는 미확인 |
| Bevacizumab | 항체 | 소분자 모델 지원 범위 밖 |

Apatinib의 PRISM20Q2 AUC 454행에서 출판 SMILES 열에는 같은 구조가 쉼표로 두 번 기록됐다. 모든 구조를 각각 정규화하면 한 parent `WPEWQEMJFLWMLV-UHFFFAOYSA-N`이고, Broad 전체 로트 ID `BRD-K25244359-066-03-4`의 카탈로그와 PubChem 45139106의 탈염 parent에 일치한다. 원천 범위에서 STR이 통과하고 결측이 없는 유효 AUC는 `HTS002`의 443개 학습 모델에 있었다. 원천 곡선 파일 SHA-256은 `2ac69a21f1d681fe7447689262b82ca6e3dc90bfef0bd96eb5479b96f424e43d`, Broad 카탈로그는 `3324dfac93d19ceaeafbe7d9e466127709328777cb2f6c5a465740cd327088e9`다. [Broad 공개 자료](https://depmap.org/repurposing/) · [PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/45139106) · [PRISM24Q2 배포](https://doi.org/10.6084/m9.figshare.25917643)

현재 네 약물의 892개 PDO·약물 키를 등록했다. 성인의 약물 내 환자 감수성은 계산 가능한 설계이며, 4약물의 Top-5 적중률은 분별력이 없다. 추가 학습 후보를 포함하는 회차는 앵커·공통 비교 범위를 먼저 동결해야 한다. 현재 등록에서는 연속 `Normalized AUC`를 채점하지 않았다. 초기 스키마 출력에 5개 PDO의 이진 `Sensitive` 플래그가 노출된 이력을 명세에 기록했다. 발현 배포는 CC BY 4.0이다. 발행사 표는 파일별 재배포 라이선스가 확인되지 않아 원본을 G1에 둔다. [저자 발현 배포](https://doi.org/10.17632/rv2w3dv9rs.2) · [원논문](https://doi.org/10.1016/j.ccell.2024.03.004) · [공개 다운로드 영수증](yang_hcc_asset_receipt.json) · [PRISM 메타데이터 영수증](yang_hcc_prism_metadata_receipt.json)

Slurm `33216`의 첫 등록은 S6에서 지역 숫자가 생략된 두 PDO 키를 감지하고 중단했다. 이 원본 형식을 허용한 `33217`에서 등록을 완료했으며 두 키는 기저 RNA와 매칭되지 않았다. [작업 원장](RUN_LEDGER.tsv)
