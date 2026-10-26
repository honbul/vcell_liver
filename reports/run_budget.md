# 실행 예산과 현재 상태

서버 G1, NVIDIA A30 24 GiB ×4, driver 560.28.03 (CUDA driver API 12.6), RAM 약 1 TiB, CPU 80 logical threads. 실제 원문은 environment.json. 다른 사용자의 gmx 프로세스가 GPU 4개에 각각 약 250–322 MiB 사용 중이며 종료하지 않는다. 본 작업은 CUDA_VISIBLE_DEVICES=0 하나와 CPU thread 8개로 제한한다. GPU VRAM을 합산하지 않는다.

scGPT whole-human와 Geneformer-V2-104M 두 모델만 frozen inference. 64-cell smoke batch1, 2000-cell pilot와 10000-cell comparison은 scGPT batch8 / sequence1200, Geneformer batch4 / sequence4096. scGPT는 upstream 지원 non-flash 경로. 추론 출력만 저장하며 새 사전학습/미세조정/전체 Tahoe 다운로드는 수행하지 않는다. OOM이면 batch를 줄이고 변경을 run config에 기록한다. gene/sequence 변경은 별도 실험이다.

환자 6/2/2 split을 모델 성능 확인 전에 seed42로 고정. 동일 분할/평가 ID에 PCA·frozen embedding·Reactome feature concatenation 비교. C={0.01,0.1,1}, class_weight={None,balanced}만 validation 선택. 결과가 낮아도 test 기반 변경을 하지 않는다.

모델별 .venvs 격리 환경을 지정 작업 루트에 생성. scGPT inference에 사용하지 않는 scvi/GEARS/legacy orbax 학습 extras는 제외하고 upstream 코드를 --no-deps로 설치; 실행에 실제 필요한 의존성만 별도 고정한다. 이는 upstream 전체 training 환경 설치가 아니며 import/embedding 경로를 실제 실행해 검증한다. 기존 base 환경을 수정하지 않는다.

가상 삭제는 train hepatocyte 최대100세포, ALB/HNF4A와 발현량 유사 random gene 및 no-op, 독립 단일-gene deletion. 실측 섭동 검증 전 탐색이며 발현 예측/약효가 아니다.

대형 산출물은 /mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver. 코드/환경은 /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver. 원 count는 공유 데이터 루트 GEO/GSE149614. 기존 데이터/KG는 수정하지 않는다.
