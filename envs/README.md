# 실행 환경 재현

`install_*.sh`는 설치에 실제 사용한 주요 버전과 inference-only 범위를 설명한다. `*.freeze.txt`는 성공한 환경에서 수집한 수정하지 않은 실제 pip freeze 출력이다. `*.requirements.txt`는 그 출력에서 editable source 행만 뺀 전체 의존성 lock이다. 기준 Python은 서버 conda의 3.10.10이며 기존 base의 패키지는 변경하지 않았다.

Geneformer는 공식 HF revision에서 파일을 선택 다운로드해 editable로 설치했다. 이 디렉토리는 Git clone이 아니므로 pip freeze가 가장 가까운 부모 Git 저장소(vcell_liver)의 URL을 editable 행에 잘못 연결한다. 따라서 freeze의 Geneformer Git URL을 설치 원본으로 사용하지 않는다. 공식 code/dictionary/checkpoint의 정확한 URL/revision/SHA256은 `manifests/assets.json`과 `upstream_versions.tsv`에 있다.

정확히 재현하려면 `scripts/bootstrap_server.sh`의 source 다운로드 및 scGPT 패치를 먼저 준비하고 다음을 실행한다.

```bash
bash envs/rebuild_from_lock.sh scgpt
bash envs/rebuild_from_lock.sh geneformer
```

lock 설치는 `--no-deps`로 전체 고정 패키지 목록을 설치한 뒤, 해당 공식 local source를 별도로 `--no-deps -e` 설치한다. scGPT의 사용하지 않는 training/integration extras(cell-gears, leidenalg, orbax, scib, scvi-tools)는 제외했다. 이 때문에 scGPT의 pip check는 이 5개를 누락으로 보고하며 `reports/scgpt_dependency_check.txt`에 원문을 남겼다. 이는 전체 upstream training 환경이 아니며, 이 튜토리얼의 공식 non-flash embedding 경로는 실제 HCC 데이터에서 실행해 검증했다. Geneformer 환경의 pip check는 통과했다.

GPU 부족 시 scGPT/Geneformer batch_size를 1로 줄인다. Geneformer CPU fallback은 공식 tokenizer와 동일 encoder/CLS(layer11) 의미를 유지하지만 upstream get_embs가 CUDA를 고정하므로 CPU 부분만 별도로 구현했다. 이번 실제 비교는 A30 CUDA에서 수행했다.
