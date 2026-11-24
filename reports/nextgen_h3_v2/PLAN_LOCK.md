# H3-v2 사전 실행 잠금

기준은 H2-v1 커밋 `e90c8e2cf684b6c332e34a678239f8770cd485a7`이다. H3-v2 전용 worktree와 G1 `nextgen_v2/h3_v2`를 사용한다. H2-v1과 H3-v1 자산은 각 원래 SHA를 검증해 읽는다.

`ng2-h3-*`의 초기 GPU 상한은 세 장이었다. 2026-11-24 09:37 KST부터 사용자 승인에 따라 학습을 A100 여섯 장과 CPU 배열 작업 35개까지 병렬 실행한다. CPU 파티션의 노드당 상한은 48코어다. 이미 배정된 CPU 학습은 12코어를 사용했고, 이후 대기 및 신규 학습은 작업당 8코어를 요청한다. 단계별 실행 한도 이력은 G1의 `logs/runtime-adjustments.tsv`와 최종 Slurm 원장에 기록한다.

2026-11-24 10:29 KST 이후 관리자 요청을 반영해 C4의 H3-v2 사용량을 48코어에서 24코어로 낮췄다. C4를 대기 및 신규 CPU 작업의 배정에서 제외하고, CPU 학습의 동시 실행 한도를 24개로 낮췄다. 실행 중인 다른 노드의 작업은 종료와 함께 새 한도로 수렴한다.

2026-11-24 10:55 KST 이후 CPU는 C5 한 대, GPU는 G4 한 대에 고정한다. 이관 직후 C4의 할당은 0코어였다. C5에는 기존 12코어와 8코어 학습, 신규 8코어 학습, 2코어 모니터링을 배정했다. G4의 동시 A100 할당 상한은 6장이다.

같은 날 13:53 KST에 기존 12코어 작업 `36482_38`이 완료돼 C5 신규 배열 `36678`의 동시 실행 상한을 1에서 2로 올렸다. 13:54 KST에 두 번째 신규 작업이 시작됐고, C5 총 할당은 학습 24코어와 모니터링 2코어로 줄었다.

16:28 KST부터 사용자 요청에 따라 C5 상한을 48코어로 올렸다. CPU 학습은 8코어 작업 최대 5개와 6코어 작업 최대 1개를 병렬 실행하고, 모니터링에 2코어를 배정한다. 대기 작업 재등록은 이미 완료된 학습 기록과 같은 내부 분할, 설정 및 입력을 재사용한다. 각 후속 CPU 배열은 기존 작업이 종료된 뒤 자리를 이어받는다. 최종 CPU 학습과 기준선 예측은 순차 배정한다.

G4의 A100 학습은 2026-11-24 21:39:53 KST에 끝났다. 이어진 GBM 비교와 신경망 병행 탐색은 G3 한 대에서 최대 A40 네 장으로 진행한다. CUDA LightGBM 4.6.0은 G1의 별도 경로에 설치했고 CUDA 12.1 도구로 A40용 `sm_86`을 빌드했다. CUDA와 OpenCL float64 비교에서는 이미 완료된 Huber 후보 네 개를 같은 내부 분할과 seed로 재학습한다. 시간, 예측, 내부 지표 및 실행 라이브러리의 SHA를 각각 G1의 `cuda-validation`, `opencl-validation` 영수증에 보관한다. 후보 선택은 등록된 내부 학습 영수증을 사용한다.

같은 날 22:13 KST부터 FiLM, Transformer, GNN의 다음 탐색을 C5의 GBM 학습과 병행한다. 이 세 패밀리의 다음 후보는 각 패밀리에서 완료된 3개 내부 폴드로 결정된다. 다음 라운드에 등록할 때 전체 라운드의 선택을 다시 계산해 먼저 등록한 후보 및 순서가 같은지 확인한다. 이전 라운드의 결과가 모두 확인되면 다음 GBM 학습을 제출한다. 진행 중인 신경망 학습은 기존 배열에 유지하고, 후속 GPU 배열은 기존 배열이 종료된 뒤 G3의 A40 최대 네 장을 이어받는다. 각 라운드의 후보를 비교하기 전에 두 배열의 영수증을 함께 확인한다. 최종 Slurm 원장은 제출 내역 및 관측된 배정·상태를 기록한다.

같은 날 22:30 KST부터 C5는 모니터링 포함 32코어, C4·G3는 GBM용 각각 최대 48코어, G4는 GBM용 최대 128코어를 요청한다. G4에서는 GPU 자원 요청 없이 CPU를 배정한다. C4의 실제 배정은 다른 사용자의 작업을 포함한 파티션 상한을 따른다. 옮긴 C5 작업 두 건은 완료된 영수증에서 이어서 실행한다. 공식 GBM 후보의 학습 장치는 CPU로 유지한다.

C4의 두 CPU 파티션에 이전 배열과 대체 배열이 겹쳐 23:01 KST부터 약 3분간 H3-v2 CPU가 72코어 배정됐다. 3개 작업을 재대기시켜 48코어로 되돌렸고 완료 영수증을 유지했다. 각 대기열은 자리를 넘겨주는 기존 작업이 끝난 뒤 8코어씩 이어받는다. 파티션과 상관없이 C4의 동시 실행은 최대 6개다. C4에서 이관한 GBM 묶음은 23:03 KST에 G3에서 시작했다.

같은 날 23:34 KST까지 G3의 대기 작업 `37960_76`과 `37960_78`을 C4에서 먼저 끝날 것으로 예상되는 두 개의 8코어 대기열에 하나씩 추가했다. 새 작업 `38082`와 `38083`은 각각 `38036`과 `38037`의 종료 후 시작한다. 내부 선택 작업 `38014`는 두 새 작업을 포함한 모든 GBM 학습이 종료된 뒤 진행한다. 이관 당시 두 묶음의 학습은 시작 전이었다. 학습 설정과 분할의 SHA는 G1의 `locks/boundary-1-g3-c4-tail-rebalance.json`에 기록했다.

23:48 KST에는 신규 골격 3개 outer의 완료된 GBM 내부 결과로 다음 경계 후보 33개를 고정했다. 세 inner fold에서 총 99건이다. CPU 배열 `38102`는 G3에서 현재 경계 단계의 CPU 작업과 동시 실행 상한 복원 작업이 끝난 뒤 최대 48코어로 시작한다. 단계 선택 작업 `38103`은 `38014`의 선행 작업을 그대로 확인한 뒤 전체 다음 후보를 다시 계산하고 선행 등록된 후보와 순서까지 대조한다. GPU나 다른 서버의 다음 단계 작업은 각자의 선행 작업이 끝나면 시작한다. G3의 후속 GBM 배열은 선행 CPU 배열이 종료된 뒤 시작하며 해당 배열이 실행 중일 때는 같은 후보를 다시 제출하지 않는다.

2026-11-25 01:11 KST부터 사용자 요청에 따라 GBM 상한을 G3 144코어와 G4 176코어로 올렸다. CPU 전용 배열 `38290`과 `38291`은 1차 확장의 대기 묶음 10개를 이어받았다. G3의 신규 골격 2차 확장은 `38292`가 같은 99건을 CPU 40코어로 먼저 실행한다. 실행 중이던 학습은 계속 진행했다. G4의 기존 배열 한 묶음과 32GiB로 등록한 대체 배열 한 묶음은 실제 시작을 확인했다. 52GiB로 등록한 나머지 묶음 네 개는 G4의 메모리 배정으로 대기했다. 실행 중인 두 L2 묶음의 관측 최대 RSS가 22GiB 미만이고 대기 묶음의 내부 행 수가 비슷해, 대기 묶음의 GPU 없는 CPU 요청을 각각 32GiB와 3시간으로 조정했다. 새 선택 작업 `38308`은 1차 확장 결과를 모두 받은 뒤 후속 배열을 제출한다. G3에서는 선행 배열과 CPU 상한을 분담하고 선행 배열이 끝나면 남은 슬롯을 연다. 실제 Slurm 배정과 재등록 이력은 G1의 `locks/boundary-1-cpu-g3-g4-expanded-v23.json`과 `logs/runtime-adjustments.tsv`에 기록했다.

2026-11-25 02:45 KST부터 현재 서버 `daea`에서 CPU 8코어씩 세 작업을 실행한다. 허용 CPU 중 다섯 개는 남겨 두었다. G4의 Cell-Cold 내부 작업 두 묶음과 C5의 한 묶음에서 아직 시작하지 않은 Huber 설정을 각각 네 개씩 선택했다. 첫 로컬 시도에서 재계산한 내부 행의 해시가 G1과 달라 학습 전에 종료됐다. G1에서 검증한 내부 행 마스크를 옮겨 세 작업을 재시작했다. CPU LightGBM 바이너리와 atlas·schema·내부 행 해시는 G1과 대조했다. 완료 결과는 G1에서 설정, seed, 예측 행과 학습곡선을 검증한 뒤 기존 Slurm 작업이 재사용한다. 실행 계획은 G1의 `locks/boundary-1-daea-huber.json`에 기록했다.

02:51 KST에는 G3의 1차 L2 작업이 반환한 자원을 사용해 신규 골격 2차 GBM 아홉 묶음을 모두 실행했다. CPU 72코어를 배정했다. 후속 선택 작업은 `38393`이다. 2차 확장 이후에는 Huber·LambdaRank 등 계산량이 큰 후보를 같은 내부 분할과 패밀리 안에서 최대 세 개씩 제출한다. 실제 후보 목록으로 모든 미완료 후보가 한 묶음에만 들어가는지 확인했다.

02:56 KST에는 G4에 48코어, G3에 8코어가 확보돼, 실행 중인 G4의 긴 Huber 묶음 일곱 개에서 아직 시작 전인 마지막 후보를 세 개씩 병렬 실행했다. Slurm 배열은 G4의 `38405`와 G3의 `38406`이다. 완료된 결과는 검증을 거쳐 G1에 추가되고 기존 묶음이 재사용한다. 원래 작업의 마지막 후보를 앞당겨 학습하는 경우에도 등록된 설정, CPU, seed, fit·validation 행과 예측 검증은 동일하다. G1의 `locks/boundary-1-slurm-huber-tails.json`에 배정과 후보를 기록했다.

04:04 KST에 `daea`의 첫 Huber 결과 `788ea49e8292`를 G1의 `cell-1/inner-2`에 반영했다. 이 후보는 기존 G4 작업 `37959_28`에 등록돼 있었다. CPU 학습 시간은 77.3분이었다. G1에서 149,088개 검증 행의 ID, 예측과 학습 곡선 SHA, 유한한 train·validation Huber 값 160쌍을 확인했다. 승인 기록은 G1의 `logs/daea-cpu-imports.tsv`, 영수증 SHA-256은 `021c28f03ff9c17349dc95fc45b6e382b33c02c1c9d55f472accd87fb6aa042d`다.

04:05 KST에 G4 추가 배열 `38405_5`의 첫 Huber 결과를 G1의 `GDSC2-1/inner-2`에 반영했다. 기존 G4 작업 `37959_64`에 등록된 후보 `788ea49e8292`의 CPU 학습 시간은 68.9분이었다. 129,960개 검증 행의 ID, 예측과 학습 곡선 SHA, 유한한 train·validation Huber 값 120쌍을 확인했다. 승인 기록은 G1의 `logs/slurm-cpu-tail-imports.tsv`, 영수증 SHA-256은 `2f2b00b2fda6c41345ab1a40ad545f1451b75e947783d253d147f1782793e183`다.

04:34 KST까지 `daea` 세 작업의 첫 후보를 각각 G1에 반영했다. `cell-2/inner-0`은 108.1분, `cell-2/inner-2`는 107.7분 걸렸다. 각각 기존 G4 `37959_30`과 C5 `37332_34`의 후보이며, 동일 설정 ID가 서로 다른 내부 분할에서 한 건씩 승인됐다. G1에서 세 건 모두 검증 행과 학습곡선 SHA를 다시 대조했다.

04:50 KST에는 G3의 GBM 작업 하나가 반환한 CPU 8개에 C4에서 아직 시작하지 않은 `PRISM20Q2-2/inner-2` Huber 후보 3개를 배정했다. G3의 `38430_7` 실행과 GBM CPU 144개 배정을 확인했다. 기존 C4 작업 `38041_88`은 계속 실행하고, 검증된 병렬 결과를 받으면 해당 후보를 건너뛴다. 내부 선택 작업 `38393`은 추가 배열 종료도 기다린다. 원래 등록을 포함한 후보와 Slurm 배정은 G1의 `locks/boundary-1-slurm-huber-tails-v2.json` 및 `locks/boundary-1-slurm-huber-tails-jobs-v2.json`에 기록했다.

04:57 KST에는 C4의 GBM 작업이 반환한 CPU 8개에 G3에서 아직 시작하지 않은 `PRISM20Q2-0/inner-2` Huber 후보 3개를 배정했다. C4의 `38431_8`이 시작해 총 GBM CPU 48개를 사용한다. 기존 G3 작업 `38290_76`이 이어서 학습하며 검증된 병렬 결과를 재사용한다. 내부 선택 작업 `38393`은 이 배열 종료도 기다린다. 배정과 후보는 G1의 `locks/boundary-1-slurm-huber-tails-v3.json` 및 `locks/boundary-1-slurm-huber-tails-jobs-v3.json`에 기록했다.

05:04 KST에는 G3 배열 `38406_6`의 첫 결과 `cell-1/inner-1`을 G1에 반영했다. 기존 G4 작업 `37959_26`에 등록된 후보 `788ea49e8292`의 CPU 학습은 127.7분 걸렸다. 승인 원장을 읽은 직후 G1에서 결과 링크가 바로 조회되지 않아 24초 뒤 다시 확인했다. 등록된 검증 행 149,076개와 유한한 Huber 학습 곡선 값 153쌍을 검증했다. 영수증 SHA-256은 `271400004b7193d6b2d6fa393492e0d0b9064a2c2dc097877baa5ed65b9a56e4`다.

05:14 KST에는 정상 종료한 G3 `37960_18`이 반환한 CPU 8개에 G4 `37959_62`의 `GDSC2-1/inner-1` 미시작 Huber 후보 3개를 배정했다. G3 `38434_9`는 기존 G4 추가 배열에 배정된 후보를 제외하고 시작했다. G3의 GBM CPU는 144개이며 선택 작업 `38393`은 추가 배열 종료도 기다린다. G1의 `locks/boundary-1-slurm-huber-tails-v4.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v4.json`에 후보와 배정을 기록했다.

05:31 KST에는 정상 종료한 C4 `37961_40`이 반환한 CPU 8개에 G3 `38290_78`의 `PRISM20Q2-1/inner-0` 미시작 Huber 후보 3개를 배정했다. C4 `38438_10`이 시작해 CPU 48개를 사용한다. 기존 G3 학습과 선택 작업 `38393`의 선행 조건을 확인했다. G1의 `locks/boundary-1-slurm-huber-tails-v5.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v5.json`에 후보와 배정을 기록했다.

05:39 KST에는 정상 종료한 G3 신규 골격 2차 GBM 선행 작업 `38292_4`가 반환한 CPU 8개에 G4 `37959_58`의 `GDSC2-0/inner-2` 미시작 Huber 후보 3개를 배정했다. 이미 병렬 배정한 후보를 제외했다. G3 `38440_11`과 GBM CPU 배정 144개를 확인했다. G1의 `locks/boundary-1-slurm-huber-tails-v6.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v6.json`에 새 배열과 선택 작업 `38393`의 선행 조건을 기록했다.

06:01 KST에는 G3에서 시작한 `38430_7`의 첫 Huber 결과를 C4 `38041_88`의 `PRISM20Q2-2/inner-2` 후보로 반영했다. CPU 학습은 70.9분 걸렸다. G1에서 검증 행 81,718개, 유한한 train·validation Huber 값 181쌍 및 결과 해시를 확인했다. 영수증 SHA-256은 `7950567c00a8cc86b0462f106e5d4aaf0aa5f78a6710fb61cff97db329046a15`다.

06:10 KST에는 세 Huber 후보를 모두 반영하고 정상 종료한 G4 `38405_5`가 CPU 8개를 반환했다. 마지막 후보 `3505919266de`의 검증 행 129,960개와 유한한 train·validation Huber 값 122쌍을 확인했다. 영수증 SHA-256은 `3b6ee6ee2c3b05f9018c7aa2ffaf25d2c620402b2e3b0c31e73a88444e0a7868`이다. 그 자리에 C5 `37334_32`의 `cell-2/inner-1` 미시작 Huber 후보 `788ea49e8292`, `0a851673e3dd`, `3505919266de`를 배정했다. G4 `38448_12`에서 CPU 8개로 실행했고 G4의 H3 CPU 배정은 176개였다. G1의 `locks/boundary-1-slurm-huber-tails-v7.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v7.json`에 새 배열과 선택 작업 `38393`의 선행 조건을 남겼다.

06:18 KST에는 세 Huber 후보를 모두 반영하고 정상 종료한 G4 `38405_0`이 반환한 CPU 8개에 C5 `37326_24`의 `cell-1/inner-0` 마지막 미시작 후보 `788ea49e8292`를 배정했다. G4 `38450_13`의 CPU 전용 실행과 H3 CPU 176개 배정을 확인했다. 2개 선행 후보가 기존 C5 작업에 남아 있다. 추가 배정과 선택 작업 `38393`의 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v8.json` 및 `locks/boundary-1-slurm-huber-tails-jobs-v8.json`에 기록했다.

같은 시각, C4 `38431_8`의 첫 결과 `788ea49e8292`를 G3 `38290_76`의 `PRISM20Q2-0/inner-2`에 반영했다. CPU 학습은 80.6분 걸렸으며 검증 행 82,159개와 유한한 train·validation Huber 값 182쌍을 확인했다. 영수증 SHA-256은 `583be4f5ed23c8a926776b31641e312815612c912c56af261bc7bd33197955d5`다.

06:31 KST에는 G4 `38405_4`가 Huber 세 후보를 반영하고 정상 종료했다. 최종 후보 `3505919266de`의 검증 행 130,694개, 유한한 train·validation Huber 값 122쌍과 영수증 SHA-256 `cf7b39e519573c3e2cac865b65b3a9d79ae77524c79259b8fb415e4dc04b8564`를 확인했다. 반환된 CPU 8개에 C5 `37333_22`의 `cell-0/inner-2` 미시작 Huber 후보 `788ea49e8292`, `0a851673e3dd`를 배정했다. G4 `38454_14`의 CPU 전용 실행과 H3 CPU 배정 176개를 확인했다. 새 배열과 선택 작업 `38393`의 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v9.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v9.json`에 남겼다.

06:42 KST에는 G4 `38405_3`이 세 후보를 반영하고 정상 종료했다. 같은 GDSC2 내부 분할에서 아직 시작하지 않은 다른 Huber 후보 두 건을 반환된 CPU 8개에 배정했다. 후보 `45dad9d5e94b`, `92a020823054`의 기존 G4 작업은 `37959_54`의 `GDSC2-0/inner-0`이며 새 작업 `38457_15`가 CPU로 실행한다. G4의 H3 CPU 배정은 176개였다. 선택 작업 `38393`의 선행 조건을 포함한 새 등록은 G1의 `locks/boundary-1-slurm-huber-tails-v10.json` 및 `locks/boundary-1-slurm-huber-tails-jobs-v10.json`에 남겼다.

06:48 KST에는 G4 `38405_1`이 세 후보를 반영하고 정상 종료해 CPU 8개를 반환했다. 기존 G4 `37959_60`의 `GDSC2-1/inner-0`에서 아직 시작하지 않은 `45dad9d5e94b`를 새 작업 `38461_16`에 배정했다. 같은 분할에서 먼저 완료된 병렬 후보 `3505919266de`의 검증 행 129,908개 및 유한한 train·validation Huber 값 121쌍을 확인했다. 영수증 SHA-256은 `b836fe58315160a1df72a7b4b9730e9b906fde506b11c1b74c203f9436c9eb4a`다. G4 CPU 전용 배정은 176개이며, 선택 작업 `38393`의 선행 조건을 포함한 새 등록은 G1의 `locks/boundary-1-slurm-huber-tails-v11.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v11.json`에 기록했다.

06:53 KST에는 신규 골격 2차 확장의 G3 `38292_8`이 정상 종료했다. `drug-1/inner-1` 11건의 검증 행·예측·학습 곡선을 대조해 유한한 train·validation Huber 값 873쌍을 확인했다. 반환된 CPU 8개에는 C4 `38040_86`의 `PRISM20Q2-2/inner-1` 미시작 후보 `788ea49e8292`, `0a851673e3dd`, `3505919266de`를 배정했다. 새 작업 `38463_17`의 CPU 전용 실행과 G3 GBM CPU 배정 144개를 확인했다. G1의 `locks/boundary-1-slurm-huber-tails-v12.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v12.json`에 새 배열과 선택 작업 `38393`의 선행 조건을 기록했다.

06:58 KST에는 정상 종료한 G4 `37959_50`이 반환한 CPU 8개에 같은 서버의 `37959_30`에서 아직 시작하지 않은 `cell-2/inner-0` 후보 `92a020823054`를 배정했다. `daea`에 이미 등록된 후보는 제외했다. 새 G4 작업 `38467_18`의 CPU 전용 실행, G4 H3 CPU 배정 176개 및 선택 작업 `38393`의 선행 조건을 확인했다. 배정은 G1의 `locks/boundary-1-slurm-huber-tails-v13.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v13.json`에 남겼다.

06:59 KST에는 G4 `38405_2`가 세 후보를 반영하고 정상 종료해 CPU 8개를 반환했다. 같은 시각 영수증 `a9da9be0da43befde14ef403f24db82747a40cfaf7909913527f6730120e3df3`의 검증 행 130,688개와 유한한 train·validation Huber 값 122쌍을 확인했다. 반환된 CPU에는 C4 `38039_84`의 `PRISM20Q2-2/inner-0` 미시작 후보 `788ea49e8292`, `0a851673e3dd`, `3505919266de`를 배정했다. G4 `38468_19`의 CPU 전용 실행과 H3 CPU 배정 176개를 확인했다. 새 목록과 선택 작업 `38393` 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v14.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v14.json`에 기록했다.

07:01 KST에 `daea` 첫 작업의 네 번째 후보 `45dad9d5e94b`를 G4 `37959_28`의 `cell-1/inner-2`에 반영했다. 로컬 CPU 학습은 59.2분 걸렸으며 검증 행 149,088개와 유한한 train·validation Huber 값 121쌍을 확인했다. 영수증 SHA-256은 `ed241a37677cbefa1e573963600b516317fdfc893afd664de781554c4c1392a1`이다. 07:06 KST에는 정상 종료한 이 작업이 반환한 로컬 CPU 8개에 G4 `37959_56`의 `GDSC2-0/inner-1`에서 아직 시작하지 않은 다른 `45dad9d5e94b`를 배정했다. G1 내부 분할의 학습 행 261,378개와 검증 행 130,688개를 기준으로 옮긴 행 마스크와 입력 해시를 대조했다. `ng2-h3-daea-batch-2-cpu-0` 실행을 확인했으며, 세 로컬 작업에 24코어를 배정하고 5코어를 남겼다. 새 등록은 G1의 `locks/boundary-1-daea-huber-batch-2.json`에 기록했다.

07:08 KST에는 G3 `38434_9`의 정상 종료로 반환된 CPU 8개에 C5 `37332_34`의 `cell-2/inner-2` 미시작 Huber 후보 `92a020823054`를 배정했다. 새 G3 작업 `38495_20`의 CPU 전용 실행과 GBM CPU 배정 144개를 확인했다. 이미 로컬 배정한 후보는 제외했다. 새 등록과 선택 작업 `38393` 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v15.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v15.json`에 남겼다.

07:17 KST에는 G3 `38406_6`이 세 Huber 후보를 모두 반영하고 정상 종료해 CPU 8개를 반환했다. `cell-1/inner-1`의 마지막 결과 `3505919266de`는 학습에 66.5분이 걸렸다. 검증 행 149,076개와 유한한 train·validation Huber 값 122쌍을 확인했다. 영수증 SHA-256은 `9714feccec6defcc8855ee91432485cdbb2c280ccec67cfebedc31ce15cbdf07`이다. G3의 새 CPU 전용 작업 `38496_21`에는 G4 `37959_64`의 `GDSC2-1/inner-2` 미시작 후보 `45dad9d5e94b`를 배정했다. G3 GBM CPU 배정은 144개였다. 새 배정과 선택 작업 `38393` 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v16.json` 및 `locks/boundary-1-slurm-huber-tails-jobs-v16.json`에 기록했다.

07:23 KST에는 최종 보고서를 실제 실행 경로에 연결하기 위해 C5의 대기 중인 선택 작업을 `38500`으로 교체했다. `full-lincs-v29`는 기존 v25의 925개 파일을 SHA-256으로 대조했으며 보고서 모듈과 그 테스트만 v28에서 가져왔다. 선택·학습·채점 코드와 미완료 선행 조건 27개를 대조했고, 등록된 Huber 후보 목록의 SHA-256도 일치했다. 첫 대체 작업 `38499`는 Slurm 해제 직후의 상태 확인에서 취소됐고, 다시 등록한 `38500`이 선행 조건을 기다리는 상태임을 확인했다. `38393`과 `38499`의 실행 시간은 모두 0초다. 변경 증거는 G1의 `locks/boundary-1-select-v29.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v17.json`에 보관했다.

07:28 KST에는 G3 `38430_7`이 세 후보를 반영하고 정상 종료했다. C4 `38041_88`의 `PRISM20Q2-2/inner-2`에 반영한 마지막 Huber 후보 `3505919266de`의 검증 행 81,718개와 유한한 train·validation Huber 값 122쌍을 확인했다. 영수증 SHA-256은 `235f90b6f617cf833933bd39be1128f89a9e2568f6d610563522721c4215cc27`이다. 반환된 CPU 8개에는 G3 `37960_72`의 `PRISM20Q2-0/inner-0` 미시작 후보 `788ea49e8292`, `0a851673e3dd`를 배정했다. 새 CPU 작업 `38508_22`의 실행, G3의 GBM CPU 배정 144개, 선택 작업 `38500`의 선행 조건을 확인했다. 배정은 G1의 `locks/boundary-1-slurm-huber-tails-v18.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v18.json`에 기록했다.

07:32 KST에는 G3 `38440_11`의 정상 종료로 반환된 CPU 8개를 사용해 G3 `37960_46`의 `CTRPv2-1/inner-2` 미시작 후보 `0a851673e3dd`, `3505919266de`를 시작했다. 선행 병렬 작업이 반영하지 않은 두 후보는 기존 G4 작업의 결과를 사용한다. G3 `38509_23`의 CPU 전용 실행 및 GBM CPU 배정 144개를 확인했다. 배정과 선택 작업 `38500`의 선행 조건은 G1의 `locks/boundary-1-slurm-huber-tails-v19.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v19.json`에 기록했다.

07:44 KST에는 G4 `38457_15`가 `GDSC2-0/inner-0` Huber 후보 `45dad9d5e94b`를 반영하고 정상 종료했다. G1에서 검증 행 130,684개와 유한한 train·validation Huber 값 122쌍을 확인했다. 학습은 61.6분 걸렸으며 영수증 SHA-256은 `43a90e1a7a08edd6bcf66d2c4236895eb1a2cf0bd82b3c1dd8daa75b343c7438`이다. 반환된 CPU 8개에 C4 `38038_82`의 `PRISM20Q2-1/inner-2` 미시작 후보 `788ea49e8292`를 배정했다. G4 `38525_24`의 CPU 전용 실행과 H3 CPU 배정 176개를 확인했다. 선택 작업 `38500`의 선행 조건과 후보 등록은 G1의 `locks/boundary-1-slurm-huber-tails-jobs-v20.json` 및 `locks/boundary-1-slurm-huber-tails-v20.json`에 남겼다.

07:48 KST에는 G3 선행 신규 골격 GBM `38292_16`이 반환한 8코어에 C4 `38041_88`의 `PRISM20Q2-2/inner-2` 미시작 Huber 후보 `45dad9d5e94b`를 배정했다. 선행 작업의 `drug-2/inner-2` 후보 11건과 유한한 train·validation 지점 546쌍을 검증했다. 마지막 영수증 SHA-256은 `ad3e2773b3ed2cda7693045e79d4ba351596d491527522282a80bc4783a44398`이다. G3 `38541_25`의 CPU 전용 실행, GBM CPU 배정 144개, 선택 작업 `38500`의 선행 조건을 확인했다. 후보 목록과 배정은 G1의 `locks/boundary-1-slurm-huber-tails-v21.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v21.json`에 남겼다.

07:49 KST에는 G4 `38461_16`이 `GDSC2-1/inner-0`에 Huber 후보 `45dad9d5e94b`를 반영하고 정상 종료했다. 학습은 59.7분 걸렸으며 검증 행 129,908개와 유한한 train·validation 값 122쌍을 확인했다. 영수증 SHA-256은 `4d981d5153ceaffc0b6394cbdbac2e28fe2d3d2eb940ab62d26d2a1a4cbe67b7`이다. 반환된 CPU 8개에는 G3 `37960_46`의 `CTRPv2-1/inner-2` 미시작 후보 `45dad9d5e94b`를 배정했다. G4 `38545_26`의 CPU 전용 실행과 선택 작업 `38500`의 선행 조건을 확인했다. 배정은 G1의 `locks/boundary-1-slurm-huber-tails-v22.json`과 `locks/boundary-1-slurm-huber-tails-jobs-v22.json`에 기록했다.

07:55 KST에는 `CTRPv2-1/inner-2`의 Huber 후보 `55272af9a1b2`가 G3 `37960_46`에서 317분 넘게 걸려 G4 CPU 8코어에서 같은 등록 설정을 별도로 실행했다. 같은 설정의 다른 두 내부 폴드는 32.3분과 52.4분이 걸렸다. G3 프로세스는 8초 동안 CPU 시간 49초를 사용하며 계산을 계속했다. G4의 `38448_12`가 정상 종료하며 반환한 코어에 재실행 `38551`을 배정했다. G4의 배정은 H3 CPU 176개다. 재실행은 등록된 fold, 설정, atlas 및 seed를 사용하고 외부 평가 레이블을 읽지 않는다. 07:56 KST까지 격리 결과 경로에서 학습이 시작됐고 기존 결과 경로에는 새 결과를 반영하지 않았다. 계획과 Slurm 배정은 G1의 `locks/boundary-1-ctrp-55272-replay.json` 및 `locks/boundary-1-ctrp-55272-replay-route.json`에 남겼다.

07:56 KST에는 `daea` 추가 배치가 정상 종료하고 `GDSC2-0/inner-1`의 Huber 후보 `45dad9d5e94b`를 반영했다. 학습은 49.5분 걸렸으며 검증 행 130,688개와 유한한 train·validation 값 122쌍을 확인했다. 영수증 SHA-256은 `4717367f5b2e2895807d2a92b014bc93a6132496bfe3ce20e566cc6a4aa43c8c`이다. 다른 로컬 작업 두 건은 16코어로 계속 실행한다.

## 분할과 입력

- Drug-Cold는 H2-v1 1,001골격 연결 성분의 outer 3개 및 각 inner 3개를 그대로 사용한다. Cell-Cold는 확장 패널 3개 전역 계통 차단 분할, 교차스크린은 H3-v1의 9개 고정 평가 지지다. 교차스크린 훈련에는 해당 훈련 계통과 허용 소스의 확장 패널을 사용한다.
- 결과값 없는 query, 각 분할의 train 파일, 별도 held 실측을 떼어 둔다. 모든 모델에는 같은 학습용 관측을 제공한다. 평가 오프셋과 단위는 원 스크린에 따르며 학습용 affine 및 정규화는 각 fit에서만 계산한다.
- 전체 17,871유전자 PCA와 변동/표적 발현을 제공한다. H2의 14,421유전자 RNA 기하를 재구성하는 fit-span 사영과 query norm, 동일 경로/TF 338개와 RNA 기반 보조 추정값을 함께 쓴다. 2,048비트 ECFP4, 4가지 RDKit 물성, ChEMBL 릴리스 37의 인간 표적 유전자 1,451개, LINCS 978 landmark와 44개 경로, 관측 여부 마스크, 직접 표적 발현 요약 5개와 경로 곱을 공통 atlas에 등록한다. 신경망은 LINCS 모듈을 관측 시에만 활성화하며 LightGBM은 결측 NA 분기로 보존한다. [입력 검증 영수증](step0_verification.json)
- H2 PSD의 고정 레시피는 벌점 4, 24 h MODZ cosine, 가중 .50, 결측시 C1이다. Hybrid의 각 내부 훈련 샘플은 그 샘플과 외부 inner validation 그룹 양쪽을 제외해 적합한 H2에서 base를 받는다. 최종 refit용 훈련 base는 세 내부 fit의 OOF 연결이다. GNN 원자·결합 그래프와 CLAMP/Uni-Mol 서브암의 별도 추가 입력은 함께 표기한다.

## 튜닝과 동결

각 outer 분할 선택은 자신의 inner 결과에만 의존한다. 예비 확정한 탐색표는 [그리드](../../src/h3v2/grid.py)에 있다. R1에서 5개 패밀리를 모두 맞추고 R2는 모든 패밀리에 최소 두 후보를 추가한다. 내측 상위 세 패밀리에 공동 손실·명시적 3항 분해·정규화 탐색을 추가한다. 각 축에서 다른 설정을 고정한 관측값이 우승점의 양쪽에 올 때까지 내부 분할에서 확장한다. 대형 어텐션 폭은 메모리에 맞춘 배치 변경을 설정표에 함께 기록한다. 물리적 경계는 실제 검색 범위로 공개한다.

최종 보고서는 실제 내부 학습 기록에서 모델 구조와 탐색 범위를 집계하고, 봉인 평가 영수증에서 패밀리별 승격 조건의 충족 여부를 표시한다. 2026-11-25 07:14 KST에 G1의 R1 990건과 R2 636건으로 구조 범위 표를 검증했다. 이 중 522건은 CPU LightGBM, 1,104건은 CUDA 학습이었다. 모델 구조 및 승격 조건을 추가한 보고서 코드는 G1 `code/full-lincs-v28`에 등록했으며, 모의 평가 자료로 28개 암종 표와 두 곡선 그림의 생성을 확인했다. 최종 산출물에는 이 보고서 버전의 검증 영수증을 사용한다.

세 축 Inner 지표는 약물 내 ρ .45, 상호작용 ρ .30, NDCG@5 .25의 선형 점수로 비교한다. 동일 점수는 차례로 NDCG, 약물 내 ρ, 설정 해시로 정한다. 세 내부 검증을 먼저 균등하게 합산한 뒤 패밀리별 1개와 최대 점수의 단일 패밀리를 고른다. 최종 적합 epoch/트리 수는 inner 최적값 중앙값을 사용한다. 두 단계 트리는 L2와 LambdaRank 각각의 중앙값으로 재적합한다. 독립 seed 11711/11729/11743의 개별 점수와 표본 SD를 보고하고 seed를 표본 수에 더하지 않는다. 외부 설정은 Drug-Cold의 inner 선택 정책을 근거로 동결한다.

## 채점과 판정

3개 Drug-Cold, 확장 Cell-Cold 및 68약물 교차스크린과 GDSC1/LIMORE/LICOB의 모든 예측을 먼저 고정한다. 최종 적합과 기준선 예측의 영수증을 대조하고, 실패한 작업의 누락분은 동일 설정과 seed로 채점 전에 복구한다. 하나의 scorer가 배타적 claim을 먼저 생성하고 평가용 실측을 한 번 읽는다. Drug-Cold 및 외부는 동결 H2의 정확 행과 동일 수식으로 계산하고, 68약물 패널은 H3-v1 순위 공식을 그대로 적용한다. LINCS 724 및 결측 462를 실제 별도 지원에서 채점한다. 두 기준선 대비 전역 골격/계통 단위 짝구간, 28개 암종 지지, 세 seed 안정성을 기록한다. GO-MODEL은 사전 지정 이득과 95% 하한, 두 축 비열등 하한, 2개 스크린 개선 및 GDSC1 비열등의 결합이다. 포화 증거와 외부·암종 지지가 충분할 때만 CEILING-CONFIRMED를 사용한다.
