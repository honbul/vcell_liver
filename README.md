# vcell_liver

간암(HCC) Virtual Cell 모델링을 위한 공개 모델 재현·비교·튜토리얼 프로젝트.

## 현재 상태

2026-10-26: 문헌·공식 코드·공개 데이터 기반의 실행 계획과 모델 비교 자료를 작성했다. **이 저장소 초기 문서의 성능·VRAM 예산은 실제 모델 실행 결과가 아니다.** GPU/OS/RAM은 작업 agent가 먼저 조사한다.

## 시작 문서

- [AI agent 실험 요청서](docs/AGENT_EXPERIMENT_PLAN.md): 환경 세팅 → 데이터 준비 → 실제 smoke/pilot → 비교 → KG → 사용자 튜토리얼 → 단계별 Git 반영.
- [공개 모델 비교와 권고](docs/MODEL_COMPARISON.md): scGPT, Geneformer, scFoundation, GEARS의 입력·task·GPU 근거·공개 자산·라이선스 비교.

## 기본 경로

1. **scGPT**로 GSE149614의 환자 단위 세포 유형 분류 파이프라인을 만든다.
2. **Geneformer-V2-104M**으로 같은 환자 분할에서 비교한다.
3. Reactome 기반 경로 특징을 결합해 KG의 추가 이득을 측정한다.
4. Geneformer 가상 삭제는 임베딩 변화 탐색으로 보고한다. 실측 발현량·치료 효과 예측과 구분한다.

GEARS는 실제 유전적 섭동 데이터가 준비된 후의 조건부 경로다. HepG2의 유래와 환자 HCC의 차이를 명시한다.

## 데이터 및 공개 범위

우선 데이터: [GSE149614](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614). 공개 processed count와 metadata를 사용한다. Raw FASTQ 접근은 첫 튜토리얼의 선행조건이 아니다.

원시 데이터·대형 체크포인트·비밀값·미출판 내부 문서는 저장소에 넣지 않는다. 다운로드 manifest, hash, 환경, 코드, 작은 결과 요약과 실행된 튜토리얼을 남긴다. 모델과 데이터의 개별 라이선스를 따른다.

## 완료 기준

실제 실행된 end-to-end 결과, 환자 분리 평가와 PCA 기준선, 실측 자원 사용량, 재실행 가능한 한국어 튜토리얼을 확보한다. 설치만으로 완료하거나 임의의 지표를 성공 결과로 기록하지 않는다.
