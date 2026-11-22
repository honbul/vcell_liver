# Yang 학습의 LICOB·Wong 연구 간 전이 성적

**Yang에서 학습한 3D RNA Ridge의 환자별 감수성 신호는 LICOB와 Wong으로 전이되지 않았다.** 두 연구에서 약물 내 평균 Spearman이 음수였다. 같은 17,209개 유전자와 정정 70약물 학습표에서 다시 적합한 2D Ridge가 두 연구 모두 더 높았다. LICOB 29명에서는 3D Ridge와 2D Ridge의 Spearman 차이가 −.621 [95% −1.191, −.020]이었고 Wong 10명에서는 −.323 [−.785, +.152]였다. 두 자료의 기존 외부 점수는 별도로 보존했다. [사전 명세](HCC_3D_STUDY_TRANSFER_PROTOCOL.md) · [동결 채점 원표](hcc_3d_study_transfer_score.json)

Yang 74명·217 PDO·5약물만 3D 모델 적합에 사용했다. 평가의 세 약물은 lenvatinib, regorafenib, sorafenib이다. LICOB는 34 PDO에서 모은 성인 29명·87환자-약물, Wong은 10 PDO·10명·30환자-약물로 모든 환자의 세 약물 수치가 있었다. 낮은 LICOB 생존 AUC와 높은 Wong 성장억제 AUC를 각각 감수성 방향으로 맞췄다. [공통 입력과 학습 해시](hcc_3d_study_transfer_inputs_receipt.json)

| 모델 | LICOB 약물 내 Spearman | Pearson | 환자 내 Spearman | NDCG@3 | Wong 약물 내 Spearman | Pearson | 환자 내 Spearman | NDCG@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D 약물 평균 | .000 | .000 | .121 | .863 | .000 | .000 | .150 | .879 |
| 2D 약물별 Ridge | .312 | .314 | .121 | .863 | −.042 | −.051 | .200 | .884 |
| 2D 구조 커널 | .233 | .209 | .121 | .863 | −.038 | −.171 | .150 | .879 |
| 2D Ridge+구조 커널 | .309 | .289 | .121 | .863 | .018 | −.094 | .200 | .884 |
| 2D CellHit | .103 | .103 | .121 | .863 | −.127 | −.220 | .150 | .879 |
| Yang 3D 약물 평균 | .000 | .000 | −.552 | .696 | .000 | .000 | −.750 | .654 |
| Yang 3D 평균+2D 선형 보정 | −.042 | −.065 | −.500 | .704 | .018 | .016 | −.700 | .659 |
| **Yang 3D 약물별 Ridge** | **−.309** | **−.295** | **−.517** | **.699** | **−.366** | **−.373** | **−.750** | **.654** |
| Yang 3D 구조 커널 | −.310 | −.288 | −.517 | .699 | −.386 | −.342 | −.750 | .654 |
| Yang 3D 구조 커널+2D 사전값 | −.200 | −.226 | −.534 | .698 | −.329 | −.250 | −.750 | .654 |
| Yang 3D Ridge+보류 RNA CORAL | −.238 | −.254 | −.414 | .725 | −.378 | −.305 | −.350 | .739 |

## 환자별 감수성과 약물 평균의 이동

| 약물 | LICOB 2D Ridge | LICOB Yang 3D Ridge | Wong 2D Ridge | Wong Yang 3D Ridge |
|---|---:|---:|---:|---:|
| Lenvatinib | .242 | .198 | .067 | −.236 |
| Regorafenib | .407 | −.702 | .042 | −.370 |
| Sorafenib | .286 | −.424 | −.236 | −.491 |

위 표는 각 약물의 환자 내 감수성 순위 Spearman이다. LICOB에서 3D Ridge와 2D Ridge의 Pearson 차이는 −.609 [−1.175, −.049]였다. Yang 내부 보류에서 3D Ridge의 sorafenib Spearman은 .611이었지만 다른 두 연구에서 음수였다. 이 자료로는 농도, 약물 처리 기간, 배양 방식, 환자 구성과 RNA 배치의 영향을 각각 가를 수 없다. [Yang 내부 결과](YANG_HCC_ADAPTATION_RESULTS.md)

동결된 2D 약물 평균의 감수성 순위는 sorafenib, lenvatinib, regorafenib 순이었다. Yang 3D 약물 평균은 lenvatinib, regorafenib, sorafenib 순이었다. 2D 평균만 썼을 때 가장 민감한 약물 적중률은 LICOB .552, Wong .600이었다. Yang 3D 평균에서는 .103과 .000이었다. 이 고정 3약물에서 3D Ridge를 더해도 적중률은 그대로였다. CORAL은 Wong의 NDCG@3을 .654→.739로 올렸고 95% 짝지은 차이 구간은 [+.010, +.164]였다. 2D Ridge .884에 미달했고 약물 내 Spearman은 −.366→−.378이었다. CORAL은 보류 연구의 집단 발현을 사용하는 전이형 비교다.

등록부 교차 감사에서 LICOB와 Wong은 이전에 예측한 exact parent 21개를 공유하며, 전체 구조 카탈로그의 교집합은 22개였다. 이 수는 반응표 완결성 감사 전의 구조 지지다. 이번 기술 평가는 Yang 적합에 있는 세 약물까지만 포함한다.

`33264`에서 입력을 만들고 `33265`·`33266`에서 2D 비교군을, `33267`에서 3D 예측을 생성했다. `33268`이 11방법의 1,452개 예측을 병합해 동결했다. `33269`에서 동결을 검증한 후 두 연구의 AUC를 별도로 채점했다. 예측 동결 SHA-256은 `f391b3ebea87ee1a438305397220924b4c391ca7e6f75efc18816ae0327022f6`이다. [동결 영수증](hcc_3d_study_transfer_forecast_receipt.json) · [작업 원장](RUN_LEDGER.tsv)
