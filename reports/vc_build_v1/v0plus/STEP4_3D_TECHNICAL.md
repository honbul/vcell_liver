# 독립 3D 기술 평가

Broutier GSE84073의 HCC-1·HCC-3은 세 학습 스크린과 별도 연구에서 확립한 환자 유래 HCC 오가노이드다. 출판사 보충 Dataset 1의 기저 RPKM과 Dataset 5의 3D 약물 AUC를 연결했다. GDSC 화학 등록과 일치한 7약물의 4개씩 반복을 중앙값으로 합쳐 14개 관측을 평가했다. RNA는 공통 유전자 12,273개에서 소스 데이터에만 재적합했다. 설정은 2D 소스 CV에서 이미 잠근 값을 사용했다. [원논문·보충자료](https://www.nature.com/articles/nm.4438?error=cookies_not_supported) · [표본 등록](broutier_receipt.json)

| 방법 | 두 오가노이드 평균 약물 순위 Spearman | NDCG@5 | Precision@5 |
|---|---:|---:|---:|
| 약물 평균 | .411 | .784 | .800 |
| 약물별 Ridge | .446 | .774 | .700 |
| V0+ inductive | .554 | .860 | .800 |

V0+의 Spearman은 HCC-1에서 .643, HCC-3에서 .464였다. 같은 두 표본의 Ridge는 .679, .214였다. 오가노이드 두 개가 각 환자를 대표하므로 약물별 개인화 Spearman·Pearson의 최소 5계통을 충족하는 약물은 0개다. 이 비교는 약물 순위의 기술 평가로 해석한다.

주 3D 코호트 LICOB에서는 43개 모델·76약물의 반응과 일치하는 RNA 헤더를 확인했다. 발행 RNA 아카이브가 MD5·CRC 검사를 통과하지 못했으므로 주 3D 확인은 미완료다. [LICOB 무결성·지원 감사](STEP1_SCALE_AND_LICOB.md) · [결과 원표](broutier_score.json) · [동결 영수증](broutier_forecast_receipt.json)
