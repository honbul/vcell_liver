# R18 예측 경로 합성 입력 검증

R18의 실제 예측 진입점에 만든 입력을 넣어 R17 모델을 직접 적합하고 결과를 병합했다. 두 경우를 시험했다. `C2/B/f1-B`에는 두 release와 같은 화합물의 비간계열 RNA 입력을 넣었다. `C3_24h_HEPG2_only/A/f2-C`에서는 HEPG2로만 적합하고 HUH7을 조회했다. 두 경우 모두 978개 landmark의 순서를 유지하고 적합·선택·평가용 ID를 서로소로 만들었다. 화합물과 RNA 배열 및 ChemBERTa 임베딩은 테스트용으로 구성했다.

CPU 경로와 chemCPA, TranSiGen, XPert, PRnet의 GPU 경로가 두 경우 모두 완료됐다. 병합 파일에서 등록된 예측 arm 전체, 행·유전자 축, 유한 배열과 입력 해시를 검증했다. 빌드부터 검증까지 Slurm 작업 14개가 모두 종료 코드 0이었으며, GPU 작업은 동시에 최대 두 개를 실행했다. 관련 로컬 테스트는 136개가 통과했다. [작업 원장](data/preflight/r18/native/jobs.tsv) · [실행 내역](data/preflight/r18/native/resource_usage.tsv) · [검증 영수증](data/preflight/r18/native/verified.json)

실행 커밋은 `13275dc`, G1 실행 ID는 `r18-native-smoke.J9gcv3pc`이다. 이 검증의 범위는 두 합성 입력의 실제 적합·병합·예측 검사까지다. R18 예약 발현은 미개봉 상태이며, 실제 108개 조합의 예측 동결과 별도 채점은 미실행이다.
