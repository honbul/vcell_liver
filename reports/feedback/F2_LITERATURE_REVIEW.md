# F2. Perturbation 반응 예측 모델·최신 벤치마크 검토 (피드백 2)

**작성일:** 2026-11-10 · **범위:** `docs/FEEDBACK_PLAN_20261110.md` 2절(모델·최신 연구 검토). 실험 결과가 아니라 **문헌·코드·체크포인트 조사**다.
**확인 방법:** 논문 메타데이터는 Nature/PMC 페이지 meta 태그, PubMed E-utilities, Crossref로 확인했다. 코드·체크포인트는 2026-11-10에 GitHub API와 Hugging Face(HF) API로 저장소 존재 여부, 라이선스, 마지막 push/release를 직접 조회했다. 직접 확인하지 못한 내용은 **[미확인]**, 원문 대신 2차 출처(검색 요약, 블로그)에 근거한 내용은 **[2차]**로 표시했다.

---

## 0. 요약

- **유전적 LOCO(4개 CRISPRi 세포주)에서 실제로 돌릴 후보:** ① **PIE**(Arc, 2026 preprint), ② **State ST**(Arc, *Cell* 2026), ③ **TxPert**(*Nat Biotech* 2026, 선택). 세 모델 모두 **Replogle K562/RPE1 + Nadig HepG2/Jurkat의 leave-one-cell-line-out을 원 논문이나 공식 체크포인트에서 이미 수행**했다. PIE는 unseen context, unseen perturbation, double OOD를 하나의 모델에서 공식 split으로 평가하는 유일한 공개 모델이다.
- **화학적 SNU-423 교차 세포주 후보:** ① **PIE**(`tahoe_wdataset` 레시피에 SNU-423을 뺀 사용자 split), ② **State ST**(Tahoe, TOML `[zeroshot]`에서 SNU-423 보류). 두 모델의 **공개 Tahoe 체크포인트는 모두 SNU-423을 학습에 포함**하므로 재학습이 필요하다.
- **필수 기준선(모든 실험 공통):** no-change, context-mean(대상 세포주 control + 학습 세포주 평균 Δ), perturbation-mean, nearest-cell-line, 선형/GO-ridge. 근거: Ahlmann-Eltze 2025, Systema, TxPert의 nearest-cell-line 결과, VCC 2025.
- **계획서 인용 오류:** ③ "Tian et al."의 제1저자는 **Lanxiang Li**이고 Luyi Tian은 교신(마지막) 저자다. 따라서 "Li et al."로 고쳐야 한다. ④ Systema는 2025-08-25 온라인 게재, 2026-06 호 수록이다. ⑥ State 인용 URL은 Arc 도구 페이지라서 DOI로 바꾸기를 권한다. 나머지는 확인되었고, 세부는 §6에 정리했다.

---

## 1. 표 (i) 모델 매트릭스

OOD 표기: **UP**=미관측 perturbation(같은 세포주), **UC**=미관측 세포 맥락(해당 세포주에서 처리 세포 미학습), **DO**=double OOD, **Combo**=미관측 조합.

| 모델 | 해결하려는 문제 | 입력·사전지식 | 출력 | 학습 데이터(원 논문) | 다세포주 학습 / 새 세포주 control만으로 예측 | CRISPRa/i · 화학 | 핵심 한계 |
|---|---|---|---|---|---|---|---|
| **No-change / mean / linear 기준선** (Ahlmann-Eltze 2025) | 복잡 모델이 넘어야 할 하한 | control 평균, 학습 Δ 평균, 유전자·perturbation 임베딩을 쓰는 선형 모델 | 발현 Δ | Norman, Replogle K562/RPE1, Adamson ([논문](https://www.nature.com/articles/s41592-025-02772-6)) | 교차 세포주 사전학습 P(K562→RPE1 등)를 쓴 선형 모델이 가장 좋았다 | a/i(유전적) | 기준선이므로 새 perturbation의 세부 반응은 표현하지 못한다 |
| **GEARS** (Roohani 2024, *Nat Biotech*) | UP, Combo | GO 유전자 그래프 + 공발현 그래프 GNN | 발현 벡터 | Norman, Replogle, Adamson ([DOI](https://doi.org/10.1038/s41587-023-01905-6)) | **세포주 조건화 구조 없음**(데이터셋당 단일 맥락). TxPert가 UC에서 비교군으로 쓰지 않았다([TxPert](https://www.nature.com/articles/s41587-026-03113-4)) | a/i | 단일 perturbation만으로 학습하면 조합 예측 불가([README](https://github.com/snap-stanford/GEARS)). 선형·평균 기준선을 일관되게 넘지 못함([Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6)). TxPert 결과에서 일반 기준선보다 낮음 |
| **scGPT** (Cui 2024, *Nat Methods*) | 사전학습 표현으로 UP·Combo | 33M 정상세포 사전학습 Transformer + GEARS식 perturbation 토큰 | 발현 벡터 | Norman, Adamson, Replogle K562 FT ([DOI](https://doi.org/10.1038/s41592-024-02201-0)) | 세포주 조건화 없음. 직접적인 UC 기능 없음 | a/i | 선형 기준선 미달([Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6)). 성능이 체계적 변동과 강하게 상관(Pearson Δ와 상관 0.91)([Systema](https://www.nature.com/articles/s41587-025-02777-8)) |
| **Geneformer** (Theodoris 2023, *Nature*) | 네트워크 생물학 전이학습 | rank-value 인코딩 Transformer | **임베딩만**. in silico 삭제·과발현 시 임베딩 이동을 출력 | ~30M 단일세포(암세포주 제외) | 해당 없음 | 발현 예측 불가 | **실제 발현 response 예측기가 아니다.** Ahlmann-Eltze는 별도 ridge decoder를 붙여 평가했다([Methods](https://www.nature.com/articles/s41592-025-02772-6)) |
| **scFoundation** (Hao 2024, *Nat Methods*) | 사전학습 임베딩 + GEARS 결합 | 19,264 유전자 고정 패널 | 발현 벡터(GEARS decoder) | Norman 등 ([DOI](https://doi.org/10.1038/s41592-024-02305-7)) | 세포주 조건화 없음 | a/i | 자체 유전자 패널과 정확히 일치해야 해서 Replogle/Adamson에는 적용되지 못함. 3일 FT 상한에서 5 epoch([Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6)) |
| **TxPert** (Wenkel 2026, *Nat Biotech*) | UP, Combo(두 단일 perturbation 관측 시), **UC** | batch-matched control 기반 basal encoder + 다중 KG(GO, STRING, Reactome + 독점 PxMap/TxMap) GNN | 발현(Δ 표현 포함) | **Replogle K562/RPE1 + Nadig HepG2/Jurkat** + Norman ([논문](https://www.nature.com/articles/s41587-026-03113-4)) | **예.** 4개 세포주 leave-one-out에서 대상 세포주는 control만 학습에 포함 | i(4개 세포주), a(Norman) | 최고 성능 변형은 Recursion **독점 KG** 사용(공개 불가). UC에서 nearest-cell-line 기준선을 넘은 것은 K562·RPE1뿐이고 HepG2는 "거의 동등". DO는 보고되지 않음 |
| **State** (Adduri 2026, *Cell*) | 집단(set) 단위 분포 전이, 맥락 일반화 | control 세포 set(HVG 또는 SE-600M 임베딩) + perturbation one-hot | 세포 수준 발현(HVG/유전자) 및 임베딩 | ST: 1억 개 이상 perturbed 세포 70개 맥락, SE: 1.67억 관측세포 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42673963/), [preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)) | **예.** TOML `[zeroshot]`/`[fewshot]`로 세포주 보류 가능([README](https://github.com/ArcInstitute/state)). 주 평가는 "underrepresented"(대상 맥락 perturbation 30% 학습 포함)이고, zero-shot은 사전학습→FT 설정 | 유전적·signaling·화학 | perturbation이 one-hot이라 **UP·DO 불가**(공개 체크포인트의 `pert_onehot_map.pt`). 비상업 라이선스 |
| **Stack** (Dong 2026, bioRxiv) | in-context learning(prompt 세포→query 세포) | 149M 세포 사전학습 tabular attention, 55M 세포 post-training | 세포 수준 발현 생성 | CELLxGENE + Parse PBMC 사이토카인 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41542582/)) | 원리상 "타 세포주 perturbed 세포 prompt + 대상 control query"가 가능 | ICL 평가는 사이토카인·약물(PBMC) 중심 | **CRISPRi 세포주 간 전이에서는 검증되지 않음.** ICL 평가는 면역세포·donor·세포형 전이([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803207/)) |
| **PIE** (Verma 2026, bioRxiv) | **UC, UP, DO, 미관측 데이터셋** | Perceiver IO. Cellosaurus 텍스트, NCBI/GO 텍스트, ESM2, STRING, DepMap gene effect, PubChem(화학) + 대상 control 평균 + 학습셋 반응 증거 | **집단 수준 DE 확률 `p_de`, log2FC, Δ평균**(세포 단위 아님) | Replogle-Nadig 4개 세포주. xdataset은 Tahoe, Jiang, VCC25, Orion ([preprint](https://www.biorxiv.org/content/10.64898/2026.10.02.756297v1), [HF](https://huggingface.co/arcinstitute/PIE_replogle_wdataset)) | **예.** 새 맥락은 `controls_only=true` 디렉터리로 추론([HF 카드](https://huggingface.co/arcinstitute/PIE_replogle_wdataset)) | 유전적 + 화학(`tahoe_wdataset`) ([AGENTS.md](https://github.com/ArcInstitute/pie/blob/main/AGENTS.md)) | 단일세포 분포를 출력하지 않음. 저자 성능은 아직 동료심사 전. DepMap 등 맥락 prior는 대상 세포주 정보를 일부 포함 |
| **CPA** (Lotfollahi 2023, *MSB*) / **chemCPA** (Hetzel 2022, NeurIPS) | 용량·조합·맥락 조합(context transfer), chemCPA는 미관측 약물 | autoencoder + 공변량(세포주) 임베딩. chemCPA는 분자 임베딩 | 단일세포 발현 | sci-Plex, Norman, Kang 등 ([CPA DOI](https://doi.org/10.15252/msb.202211517), [chemCPA repo](https://github.com/theislab/chemCPA)) | 대상 세포주 control을 학습에 넣으면 "미관측 (세포주×약물) 조합"으로 예측 가능(CPA context-transfer 튜토리얼, [README](https://github.com/theislab/cpa)) | CPA는 유전·화학, chemCPA는 화학 | 미관측 유전자 perturbation용이 아님([Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6), [Systema](https://www.nature.com/articles/s41587-025-02777-8)). 저장소 유지보수가 정체됨(§2) |
| **scLAMBDA** (Wang 2024, bioRxiv) | UP, Combo | GenePT 등 LLM 유전자 임베딩 + disentangled VAE | 단일세포 발현 | Perturb-seq 여러 개 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39677694/)) | README에 multi-cell-line 학습 코드 언급. TxPert가 UC 비교군으로 개조해 사용([TxPert](https://www.nature.com/articles/s41587-026-03113-4)) | a/i | UC에서 TxPert에 일관되게 뒤짐 |
| **PRESAGE** (Littman 2025, bioRxiv, Genentech) | UP(단일 유전자) | 지식원별 유전자 임베딩(GO, Reactome, STRING node2vec, 교차 시스템 Perturb-seq) 집계 | 발현 Δ | Replogle K562/RPE1 등 ([Crossref](https://doi.org/10.1101/2025.06.03.657653), [repo](https://github.com/Genentech/PRESAGE)) | 세포주 전이는 주 과제가 아님 | i | 지식원 선택이 구조보다 중요하다고 보고[2차]. Genentech 비상업 라이선스 |
| **CellFlow** (Klein 2025, bioRxiv) | 복합 perturbation 분포 생성(flow matching) | 조건 임베딩(약물, 유전자, 사이토카인 등) | 단일세포 분포 | ([Crossref](https://doi.org/10.1101/2025.04.11.648220), [repo](https://github.com/theislab/CellFlow)) | [미확인] 새 세포주 control만 쓰는 LOCO 검증 여부 | 유전·화학(README) | JAX 기반. 본 검토에서 OOD 결과는 확인하지 못함 |
| **LPM** (Miladinovic 2025, *Nat Comput Sci*) | perturbation, readout, context를 분리한 대규모 모델 | 이산 조건 토큰(decoder-only) | 발현 등 | ([DOI](https://doi.org/10.1038/s43588-025-00870-1)) | **어휘 밖(out-of-vocabulary) 맥락은 예측 불가**[2차] | 유전·화학 | 새 세포주 LOCO에 원리적으로 부적합 |
| **Tahoe-x1** (Gandhi 2025, bioRxiv) | Tahoe-100M 포함 사전학습 FM(70M–3B) | scGPT식 encoder | 임베딩. perturbation 예측은 `scripts/state_transition/`(ST 결합) | ([Crossref](https://doi.org/10.1101/2025.10.23.683759), [repo](https://github.com/tahoebio/tahoe-x1)) | 보류 맥락 예측은 ST 위에서 수행 | 화학 중심 | **SNU-423을 포함한 Tahoe 전체로 사전학습**. SNU-423 평가에는 누수 위험 |

> GenePT/LLM-prior 계열(scLAMBDA, LangPert, Scouter, scELMo)은 별도 모델이라기보다 **UP용 유전자 prior**다. LangPert(ICLR 2025 MLGenX 워크숍)는 LLM이 유사 유전자를 고르고 kNN으로 집계한다([mlanthology](https://mlanthology.org/iclrw/2025/martens2025iclrw-langpert))[2차]. 세포주 전이 검증은 확인하지 못했다. **PertAdapt**(계획서 2절 표에 언급)는 2025-11 bioRxiv preprint로 실재한다. frozen FM(scFoundation, AIDO.Cell)에 adapter를 붙이며, 세포주 전이 검증은 확인하지 못했다([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.21.689655))[2차].

---

## 2. 표 (ii) 코드·체크포인트 상태 (2026-11-10 API 조회)

| 모델 | 공식 코드 | 코드 라이선스 | 마지막 push / release | 공개 체크포인트(실재 확인) | 비고 |
|---|---|---|---|---|---|
| 선형 기준선 | [const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper) | MIT | 2025-07-18 | 해당 없음 | Zenodo에 중간 결과 보관 |
| GEARS | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | MIT | 2025-02-01 | 데이터 자동 다운로드 | 본 연구에서 이미 재현함 |
| scGPT | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | MIT | 2026-04-29 | Google Drive(whole-human 등, README 표) | 본 연구에서 이미 사용함 |
| Geneformer | [HF ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) | Apache-2.0 | HF 2026-09-12 | 있음 | 임베딩 모델 |
| scFoundation | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | 코드 Apache-2.0, 가중치는 별도 Model License | 2025-11-23 | SharePoint 링크([model/README](https://github.com/biomap-research/scFoundation/tree/main/model)) | 고정 유전자 패널 |
| **TxPert** | [valence-labs/TxPert](https://github.com/valence-labs/TxPert) | **Recursion Non-Commercial EULA**(license.pdf) | 2026-03-25 / v0.1.0(2026-03-25) | `gat`, `exphormer`, `exphormer-mg`(STRING+GO), cross-cell `config-x-cell-gat`(README상 K562 중심) | 전처리 데이터는 [Zenodo 15420279](https://doi.org/10.5281/zenodo.15420279). 논문 최고 모델(독점 KG)은 미공개 |
| **State** | [ArcInstitute/state](https://github.com/ArcInstitute/state) | CC BY-NC-SA 4.0 + State Model Non-Commercial License | 2026-07-24 / v0.11.1(2026-06-29) | **[ST-HVG-Replogle](https://huggingface.co/arcinstitute/ST-HVG-Replogle), ST-SE-Replogle**(각각 `zeroshot/`·`fewshot/` × hepg2·jurkat·k562·rpe1), st-x/st-se-replogle-full, [ST-HVG-Tahoe](https://huggingface.co/arcinstitute/ST-HVG-Tahoe)(fewshot 보류: C32, HOP62, HepG2-C3A, Hs 766T, PANC-1), ST-HVG-Parse, [SE-600M](https://huggingface.co/arcinstitute/SE-600M) | preprint가 명시한 `ST-Tahoe`/`ST-Parse` 저장소는 현재 HF에 없고 `ST-HVG-*`/`ST-SE-*`로 바뀐 것으로 보임 |
| **Stack** | [ArcInstitute/stack](https://github.com/ArcInstitute/stack) | CC BY-NC-SA 4.0 + Stack Model Non-Commercial | 2026-04-28 | [Stack-Large](https://huggingface.co/arcinstitute/Stack-Large)(217M), Stack-Large-Aligned | `pip install arc-stack` |
| **PIE** | [ArcInstitute/pie](https://github.com/ArcInstitute/pie) | CC BY-NC-SA 4.0 + PIE Model Non-Commercial License | 2026-10-27 / v1.0.1(2026-10-27) | **[PIE_replogle_wdataset](https://huggingface.co/arcinstitute/PIE_replogle_wdataset)**(4 fold LOCO), [PIE_replogle_xdataset](https://huggingface.co/arcinstitute/PIE_replogle_xdataset). Tahoe용 공개 체크포인트는 확인하지 못함 | 데이터·지식원·split: [PIE_replogle_nadig_essential](https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential), [PIE_tahoe100m](https://huggingface.co/datasets/arcinstitute/PIE_tahoe100m), [PIE_sources](https://huggingface.co/datasets/arcinstitute/PIE_sources), [PIE_splits](https://huggingface.co/datasets/arcinstitute/PIE_splits) |
| CPA | [theislab/cpa](https://github.com/theislab/cpa) | BSD-3-Clause | 2024-08-14 | 튜토리얼 | `cpa-tools` |
| chemCPA | [theislab/chemCPA](https://github.com/theislab/chemCPA) | MIT | 2025-02-06 | [Backblaze zip](https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip)(README 링크, 다운로드 미시험) | 오래된 의존성 |
| scLAMBDA | [gefeiwang/scLAMBDA](https://github.com/gefeiwang/scLAMBDA) | GPL-3.0 | 2026-03-20 | 없음(유전자 임베딩 직접 제공 필요) | |
| PRESAGE | [Genentech/PRESAGE](https://github.com/Genentech/PRESAGE) | Genentech Non-Commercial | 2025-06-09 | 임베딩 캐시 [Zenodo 15587986](https://zenodo.org/records/15587986) | |
| CellFlow | [theislab/CellFlow](https://github.com/theislab/CellFlow) | MIT | 2026-11-09 | [미확인] | `cellflow-tools` |
| Tahoe-x1 | [tahoebio/tahoe-x1](https://github.com/tahoebio/tahoe-x1) | Apache-2.0 | 2026-08-25 | [HF tahoebio/Tahoe-x1](https://huggingface.co/tahoebio/Tahoe-x1)(70m, 1b, 3b) | FlashAttention 때문에 A100 이상 필요(README) |

**라이선스 주의:** TxPert, State, Stack, PIE, PRESAGE는 모두 **비상업 라이선스**다. 학술 연구에는 사용할 수 있지만 산업 협업이나 상업 이용 전에는 법무 확인이 필요하다.

---

## 3. 표 (iii) 데이터·학습 비용

| 모델 | 우리 4-line LOCO에 필요한 데이터 | 보고된·관측된 계산량 | 추정 노력(우리 클러스터, A30/A40/A100) |
|---|---|---|---|
| 기준선 | 로컬 raw h5ad → pseudobulk | CPU | 0.5–1일 |
| PIE | HF 전처리본(6,642 유전자, 5,459 (세포주, perturbation) 쌍) 또는 로컬 counts에서 `pie process`(GPU DE) | **fold당 H100 80GB × 2, 약 1.2 h, GPU당 20 GiB**. 평가는 1 GPU([AGENTS.md](https://github.com/ArcInstitute/pie/blob/main/AGENTS.md)). xdataset은 2노드 × 4 H100, 약 4.5 h | 체크포인트 평가 0.5일. 사용자 split으로 4-fold 재학습 1–2일(A100 2장 기준, 시간은 **추정**) |
| State ST-HVG | 세포주별 h5ad(HVG 2,000), TOML split | 공개 Replogle 설정: hidden 328, 8층, cell_set_len 64, batch 64, max_steps 100,000([config](https://huggingface.co/arcinstitute/ST-HVG-Replogle)). GPU 시간은 논문에서 확인하지 못함 [미확인] | 체크포인트 평가 0.5–1일. 4-fold 재학습 2–4일(**추정**, 단일 A100) |
| TxPert | 저자 cross-cell cache(Zenodo) 또는 onboarding 노트북 | GPU 사양 미보고 [미확인]. Python 3.12, PyTorch 2.6, CUDA 12.4 | 2–4일(설정 + KG 준비 + 4 fold) |
| scGPT / GEARS | 세포주별 따로 학습 | scGPT Norman FT: A100 약 74분/seed(본 연구 [FINAL_REPORT](../FINAL_REPORT.md)) | in-domain 참고용으로만 사용 |
| State on Tahoe (SNU-423) | Tahoe-100M 원본(약 9,560만 세포, 50 세포주)에서 세포 수 상한 다운샘플(`downsample_cells`) | 원 ST-Tahoe 학습 규모는 큼 [미확인 세부] | 3–7일(데이터 추출 I/O가 병목, **추정**) |
| PIE on Tahoe (SNU-423) | [PIE_tahoe100m](https://huggingface.co/datasets/arcinstitute/PIE_tahoe100m): 1,137 약물·용량 조건, 56,827 쌍, 18,151 유전자, SNU-423 150만 세포 DE 포함 | "Replogle 레시피와 같은 하드웨어"(AGENTS.md) | 1–3일(사용자 split JSON 작성 + 1 run) |

---

## 4. 표 (iv) 각 모델이 실제 검증한 OOD 유형

✓=원 논문·공식 체크포인트에서 보고, △=부분적·변형 조건, ✗=미지원 또는 미보고.

| 모델 | In-domain | UP | Combo | UC(새 세포주) | DO | 화학 UC | 근거 |
|---|---|---|---|---|---|---|---|
| 선형/평균 | ✓ | ✓ | ✓(additive) | △(교차 세포주 사전학습 P) | ✗ | ✗ | [Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6) |
| GEARS | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 위, [TxPert](https://www.nature.com/articles/s41587-026-03113-4) |
| scGPT | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | [scGPT](https://doi.org/10.1038/s41592-024-02201-0) |
| Geneformer | 임베딩만 | ✗ | ✗ | ✗ | ✗ | ✗ | [Ahlmann-Eltze Methods](https://www.nature.com/articles/s41592-025-02772-6) |
| scFoundation | ✓ | △(패널 제약) | ✓ | ✗ | ✗ | ✗ | 위 |
| TxPert | ✓ | ✓(4개 세포주 각각) | ✓(Norman, 두 단일 관측) | **✓ 4-line LOCO** | ✗(미보고) | ✗ | [TxPert](https://www.nature.com/articles/s41587-026-03113-4) |
| State | ✓ | ✗(one-hot) | ✗ | **✓** underrepresented(30% few-shot) + zero-shot(사전학습→FT). Replogle 4-line zeroshot/fewshot 체크포인트 공개 | ✗ | ✓ Tahoe 5개 보류 세포주(few-shot) | [preprint §4.2](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full), [HF](https://huggingface.co/arcinstitute/ST-HVG-Replogle) |
| Stack | — | ✗ | ✗ | △ ICL(PBMC 세포형·donor 전이) | ✗ | △(OpenProblems 약물, 세포형 전이) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803207/) |
| PIE | ✓ | ✓ | ✓(README) | **✓ 4-line LOCO** | **✓** | ✓ `tahoe_wdataset`(5개 보류 세포주) | [HF 카드](https://huggingface.co/arcinstitute/PIE_replogle_wdataset), [AGENTS.md](https://github.com/ArcInstitute/pie/blob/main/AGENTS.md) |
| CPA/chemCPA | ✓ | ✗(유전자) / ✓(약물, chemCPA) | ✓ | △(control 관측 맥락의 조합 전이) | ✗ | △ | [CPA](https://github.com/theislab/cpa), [Ahlmann-Eltze](https://www.nature.com/articles/s41592-025-02772-6) |
| scLAMBDA | ✓ | ✓ | ✓ | △(TxPert가 개조) | ✗ | ✗ | [TxPert](https://www.nature.com/articles/s41587-026-03113-4) |
| PRESAGE | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | [repo](https://github.com/Genentech/PRESAGE) |
| LPM | ✓ | △ | — | ✗(어휘 밖 맥락 불가)[2차] | ✗ | ✗ | [DOI](https://doi.org/10.1038/s43588-025-00870-1) |

**공개 수치 참고 (PIE `replogle_wdataset`, 4 fold 평균 ± SD, 출처 [HF 카드](https://huggingface.co/arcinstitute/PIE_replogle_wdataset)):** DE AUPRC는 unseen_ctx 0.416±0.029, unseen_pert 0.305±0.026, unseen_ctx_pert 0.237±0.040. Direction match는 각각 0.824, 0.778, 0.726이다. 같은 split에서 우리가 재현해 비교할 기준점으로 쓸 수 있다. **주의:** 이 split에서 TxPert/State/기준선을 PIE와 같은 지표로 비교한 결과는 PIE preprint 본문에서 확인하지 못했다 [미확인].

**TxPert의 UC 결과 해석([논문](https://www.nature.com/articles/s41587-026-03113-4)):** 4개 보류 세포주 모두에서 일반 기준선(대상 control + 학습 세포주 평균 Δ)을 넘었다. 그러나 **nearest-cell-line 기준선을 넘은 것은 K562·RPE1뿐이고 HepG2에서는 거의 동등**했다. HepG2 전이에서 모델 이득이 작을 수 있다는 사전 신호이므로 R1 판정 기준에 반영해야 한다.

---

## 5. 벤치마크 요약과 지표 함정

| 벤치마크 | 규모·과제 | 핵심 결론 | 근거 |
|---|---|---|---|
| **Ahlmann-Eltze et al. 2025** *Nat Methods* 22:1657 | FM 5개(scGPT, scFoundation, Geneformer, scBERT, UCE) + GEARS, CPA vs no-change/additive/mean/linear. Norman(이중), Replogle K562/RPE1, Adamson(단일 UP) | 어떤 딥러닝 모델도 단순 기준선을 일관되게 넘지 못했다. 가장 좋았던 것은 **다른 세포주 Perturb-seq로 사전학습한 perturbation 임베딩을 쓴 선형 모델**이다. 즉 교차 세포주 perturbation 데이터가 atlas 사전학습보다 유용했다 | [논문](https://www.nature.com/articles/s41592-025-02772-6) |
| **Wei et al. 2026** *Nat Methods* 23:451(온라인 2025-12-11) | 27개 방법 × 29개 데이터 × 6개 지표. cellular context generalization(i.i.d./o.o.d.)과 perturbation generalization(유전·화학) | 초록 결론은 "cellular context embedding 접근이 일반화에 필요하다"는 것이다. 저자들은 bioLord-emCell(세포주 임베딩 + disentanglement)을 제안했다. 세부 순위는 유료 본문이라 확인하지 못함 [미확인] | [논문](https://www.nature.com/articles/s41592-025-02980-0), [scPerturBench](https://github.com/bm2-lab/scPerturBench) |
| **Li et al. (계획서의 "Tian et al.") 2026** *Sci Adv* 12:eaed3414 | 13개 방법 + context-mean, perturbation-mean, 선형, MLP 기준선. 25개 데이터(공개 23 + 자체 2), 24개 지표, 3개 과제(UP, Combo, 세포형 전이) | 성능은 효과 크기와 지표 관점에 크게 좌우된다. 약한 perturbation은 control과 비슷해서 절대값 지표가 높게 나온다. 모델들은 분산을 압축하는 보수적 편향을 공유한다. **Task 3(세포형 전이)에서 일관되게 일반화한 방법은 없었고 DE 복원은 모두 저조했다.** E-distance가 작을수록 전이가 잘 되지만 완전한 예측 인자는 아니다. Task 3에는 Replogle가 포함되지 않았다(Kang, Haber, sciplex3, Tahoe 부분집합 등) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13557097/), [코드](https://github.com/TianGzlab/scPerturbBench) |
| **Systema** (Viñas Torné et al.) *Nat Biotech* 44:1050(온라인 2025-08-25, 2026-06 호) | 10개 데이터, 3개 기술, 5개 세포주. CPA, GEARS, scGPT vs perturbed-mean/matching-mean | 처리군과 대조군 사이의 **체계적 변동**(선택 편향, 교란. 예: RPE1에서 perturbed 46% vs control 25%가 G1)이 Pearson Δ 같은 지표를 부풀린다. 체계적 변동과 Pearson Δ의 상관은 GEARS 0.95, scGPT 0.91. 기준점을 **perturbed centroid**로 바꾸면 점수가 크게 떨어진다. centroid accuracy를 제안 | [논문](https://www.nature.com/articles/s41587-025-02777-8), [코드](https://github.com/mlbio-epfl/systema) |
| **VCC 2025** | H1 hESC, CRISPRi 300개 표적(학습 150 / 검증 50 / 테스트 100), 약 30만 세포. 지표는 DES, PDS, MAE를 perturbation-mean 기준선으로 정규화 | 1위 BM_xTVC(하이브리드 + DEG 빈도 특성), 3위 Outlier(요약통계 기반 선형 스케일링). "거의 모든 모델이 MAE에서 기준선보다 나빴다." L1 기반 PDS는 스케일에 민감하다. 상위 팀은 PDS/DES 최적화에 집중했다. 순수 AI 접근은 통계 기준선을 일관되게 넘지 못했다 | [wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up), [데이터·지표](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge) |
| **VCC 2026** | **zero-shot. 대회 전용 학습셋 없음.** 처음 보는 세포주 6개(검증 3 + 최종 3)에서 CRISPRi 반응 예측. 참가자에게는 non-targeting control 발현만 제공. cell-eval 6개 지표 종합 | 계획서 R1의 "대상 세포주 control만 관측" regime과 같은 정의다. 결과(11월 중하순 발표 예정)는 아직 공개되지 않았다(2026-11-10 기준) | [공지(2026-08-20)](https://arcinstitute.org/news/virtual-cell-challenge-2026) |

**지표 함정과 R1/R3에 대한 권고**

1. **Control 기준 Δ 지표의 편향:** control 대비 Δ의 Pearson이나 비가중 MSE/MAE는 control이 편향되었거나 신호가 희소할 때 "모든 perturbed 평균" 예측(mode collapse)에 보상을 준다. 대안은 DEG 가중 MSE(WMSE)와 perturbed-mean 기준 R²(Δ), 그리고 음성·양성(technical duplicate) 기준선으로 보정하는 것이다([Diversity by Design, arXiv 2506.22641](https://arxiv.org/abs/2506.22641)). Systema의 perturbed-centroid 기준도 같은 문제를 다룬다.
2. **체계적 변동:** 세포주마다 control과 perturbed의 세포주기 차이 등이 다르면 LOCO에서 "공통 효과"만 맞혀도 점수가 높아진다. **context-mean 기준선 대비 이득**과 Systema 지표를 함께 보고한다.
3. **DEG 기반 지표:** VCC의 DES는 Wilcoxon + BH FDR≤0.05로 DEG를 정의하고, 예측 DEG 수를 정답 수로 자른다([출처](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge)). 이 지표는 세포 수와 DE 검정 방법에 민감하다. PIE는 Mann-Whitney + BH(gpudge) DE를 정답 라벨로 학습한다([데이터 카드](https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential)). **같은 DE 정의를 모든 모델에 적용**해야 한다. Turbine은 세포 수 편향을 보정하면 효과 추정과 모델 평가가 바뀐다고 보고했다 [2차, [turbine.ai](https://turbine.ai/?p=2992)].
4. **스케일 민감성:** VCC 2025 분석에 따르면 L1 기반 PDS는 예측의 크기(스케일)에 민감해서, 예측 크기를 키우면 부호 기반 코사인 유사도에 가까워진다. 상위 팀들은 정확한 크기보다 발현 패턴을 맞추는 쪽에 집중했다([wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)). 따라서 **discrimination/retrieval 계열 지표는 Δ 크기 보정(norm matching) 없이 모델 간 비교에 쓰지 않는다.**
5. **효과 크기 층화:** Li et al. 2026은 약한 perturbation일수록 control과 비슷해 절대값 지표가 높게 나오고, 큰 효과에서는 Δ·DE 지표가 더 선명하다고 보고했다([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13557097/)). **효과 크기 구간별로 층화 보고**가 필수다. 이는 계획서 A5 가설과 같은 문제다.
6. **보수적 편향(분산 압축):** 미세조정된 foundation model은 분산을 압축하고 조합 효과를 과소평가한다([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13557097/)). 예측 Δ의 표준편차 대 정답 Δ의 표준편차 비율을 함께 보고한다.

---

## 6. 표 (v) 실제 실행 후보와 제외 근거

### 6-1. 유전적 LOCO (Replogle K562/RPE1 + Nadig HepG2/Jurkat, CRISPRi essential)

| 순위 | 모델 | 선정 이유 | 설치·실행 요건 | 예상 노력 |
|---|---|---|---|---|
| **필수** | **기준선 묶음**(no-change, context-mean, perturbation-mean, nearest-cell-line, GO/linear ridge) | Ahlmann-Eltze 2025, Systema, VCC 2025가 모두 "기준선을 못 넘는다"를 보고했다. TxPert조차 HepG2 UC에서 nearest-cell-line과 거의 동등했다. 기준선 없이는 어떤 결과도 해석 불가 | 기존 `src` 파이프라인 재사용. CPU만 필요 | 1일 |
| **1** | **PIE** (`replogle_wdataset`) | 4개 세포주 LOCO를 **공식 split·공식 체크포인트**로 제공하고, **unseen_ctx / unseen_pert / unseen_ctx_pert(double OOD) 세 조건을 한 모델에서** 평가하는 유일한 공개 모델이다. 새 맥락은 `controls_only=true`로 control만 주고 추론하므로 계획서 R1의 regime와 정확히 일치한다. 공개된 4-fold 지표(§4)가 있어 **우리 재현의 정답 대조군**이 된다 | Python 3.12, `uv`, `pip install arc-pie`(또는 `uv sync --frozen`). `PIE_DATA_ROOT`/`PIE_RUNS_ROOT`/`PIE_CACHE_DIR` 설정, HF 토큰(`hf auth login`), `logger.enabled=false`로 wandb 없이 실행. 데이터·지식원은 config가 HF 커밋으로 고정해 자동 다운로드. 로컬 h5ad에서 다시 만들려면 `[process]` extra(GPU DE, gpudge)와 DepMap `CRISPRGeneEffect.csv` 수동 다운로드, 텍스트 임베딩 재생성 시 `OPENAI_API_KEY` 필요(공개 `PIE_sources`를 쓰면 불필요) | 체크포인트 평가 재현 **0.5일**. 자체 split 4-fold 재학습 **1–2일**(저자 보고: fold당 H100 80GB×2, 약 1.2 h, GPU당 20 GiB → **A100 80GB 2장 권장, A40 48GB 가능, A30 24GB는 한계**) |
| **2** | **State ST-HVG** (`zeroshot`/`fewshot` Replogle) | 같은 4개 세포주에 대해 **zeroshot·fewshot 체크포인트가 세포주별로 공개**되어 있다. 출력이 **단일세포 수준 발현**이라 기존 scGPT 공통 벤치마크의 ΔRMSE/ΔPearson/top-DE와 **같은 지표로 직접 비교**할 수 있다(PIE는 DE 요약만 출력하므로 상호보완). TOML `[zeroshot]` 한 줄로 세포주를 보류한다 | `uv tool install arc-state`(+`cell-eval`, `cell-load`). `state tx preprocess_train`으로 HVG h5ad 생성 후 TOML 작성. 공개 config: hidden 328, 8층, `cell_set_len` 64, batch 64, `max_steps` 100,000 | 체크포인트 평가 **0.5–1일**, 4-fold 재학습 **2–4일**(추정, A100 1장/fold) |
| **3(선택)** | **TxPert** (`config-x-cell-gat`) | 4개 세포주 LOCO를 수행한 **유일한 동료심사 게재 모델**이며, cross-cell cache(K562/RPE1/HepG2/Jurkat + Adamson)와 nearest-cell-line·split-half reproducibility 기준선 스크립트를 함께 제공한다. 다만 **최고 성능 변형은 Recursion 독점 KG**(PxMap/TxMap)를 쓰므로 공개 재현은 STRING+GO 버전으로 제한된다 | `uv venv --python 3.12`, CUDA 12.4, `uv pip install -e .`. 데이터는 실행 시 `./cache`로 자동 다운로드. Hydra `--config-name=config-x-cell-gat`, 새 세포주는 `datamodule.suppress_cell_type_validation=true` | **2–4일**(환경 + KG 준비 + 4 fold) |

**제외와 사유**

| 제외 대상 | 사유 |
|---|---|
| **GEARS** | 세포주 조건화 구조가 없어 LOCO에 원래 설계가 맞지 않는다. TxPert의 UP 비교에서 비학습 일반 기준선보다 낮았고, UC 과제에는 저자들도 포함하지 않았다. 본 연구에서 이미 재현했으므로 **in-domain 참고치로만** 유지한다 |
| **scGPT / scFoundation** | 세포 맥락 조건화 입력이 없다. scFoundation은 자체 19,264 유전자 패널과 정확히 일치해야 해서 Replogle 계열에 적용 실패 사례가 있다. scGPT는 **기존 공통 벤치마크의 in-domain 기준선**으로만 유지 |
| **Geneformer** | 발현 response 예측기가 아니라 임베딩 모델이다. 별도 ridge decoder를 붙여야 평가 가능하므로 "모델 비교"가 아니라 "임베딩 + 선형 모델" 비교가 된다. 기준선 묶음의 선형 모델로 흡수 |
| **CPA / chemCPA** | 미관측 **유전자** perturbation용으로 설계되지 않았다(저자들도 그렇게 명시). 유전적 LOCO에서는 제외하고 화학 실험에서만 검토 |
| **LPM** | 어휘 밖(out-of-vocabulary) 맥락은 예측할 수 없다고 보고되어 새 세포주 LOCO에 원리적으로 부적합[2차] |
| **Stack** | CRISPRi 세포주 간 전이로 검증된 바 없다(ICL 평가는 PBMC 사이토카인·약물·donor 전이). 흥미롭지만 **1차 실행 대상은 아니며** 여유가 있을 때 탐색 |
| **Tahoe-x1** | SNU-423을 포함한 Tahoe-100M 전체로 사전학습되어 SNU-423 평가에서 **누수 위험**. 유전적 과제와도 무관 |
| **scLAMBDA / PRESAGE / LangPert / PertAdapt / CellFlow** | 전부 UP(미관측 유전자)용 prior 중심이고 세포주 전이 검증이 없거나 확인하지 못했다. 단, **scLAMBDA는 TxPert가 UC 비교군으로 개조해 사용**했으므로 비용이 낮은 추가 비교군으로 여유 시 검토 가능 |

### 6-2. 화학적 교차 세포주 (Tahoe-100M, 타깃 SNU-423)

| 순위 | 모델 | 선정 이유 | 설치·실행 요건 | 예상 노력 |
|---|---|---|---|---|
| **1** | **PIE** (`tahoe_wdataset`, 사용자 split) | Tahoe 전처리본이 이미 공개되어 있다(1,137 약물·용량 조건, 56,827 (세포주×조건) 쌍, 18,151 유전자, DMSO 대조). **SNU-423(CVCL_0366) 150만 세포가 포함**되어 있고 PIE는 소분자 지식원(PubChem)을 지원한다. 공개 `tahoe_wdataset` split의 보류 세포주는 CVCL_0334/0480/1097/1098/1285이고 **SNU-423은 학습 측에 있으므로**, SNU-423을 보류하는 **자체 split으로 재학습**해야 한다(누수 방지) | §6-1의 PIE 요건과 동일. split JSON(`<dataset>.<context>` → perturbation 목록)만 새로 작성 | **1–3일** |
| **2** | **State ST** (Tahoe 부분집합 + TOML `[zeroshot]`) | 단일세포 분포 출력이 필요할 때의 선택지. Tahoe 5개 보류 세포주 few-shot 체크포인트가 공개되어 있어 레시피가 검증되어 있다. 단, 그 체크포인트의 보류 세포주는 C32, HOP62, **HepG2-C3A**, Hs 766T, PANC-1이고 **SNU-423은 학습에 포함**되므로 역시 재학습 필요 | §6-1 State 요건 + Tahoe 부분집합 추출. `data.kwargs.downsample_cells`로 (cell_type, perturbation) 그룹당 세포 수 상한 | **3–7일**(Tahoe 추출 I/O가 병목) |
| 선택 | chemCPA | 미관측 **약물** 축의 비교군으로만. 새 세포주 축은 주 설계가 아니고 저장소 유지보수가 2025-02에서 멈췄다 | `theislab/chemCPA`, 오래된 의존성 | 2–3일 |

**화학 실험 주의:** ① Tahoe-100M은 CC0-1.0이지만 **cell line / plate / drug / dose / DMSO 대조의 정합성**을 먼저 확인해야 한다(계획서 G2). ② CRISPRi 유전적 실험과 **같은 성능표에 섞지 않는다**(계획서 체크리스트). ③ HepG2는 Cellosaurus 기준 **간모세포종 유래로 재분류**되었고(CVCL_0027, "Problematic cell line: Misclassified"), SNU-423(CVCL_0366)이 성인 HCC(40세 남성, DepMap ACH-000493)다. 두 실험의 생물학적 증거 수준을 분리해 기술한다.

### 6-3. 권고 실행 순서

1. **G1-a(0.5일, GPU 1장):** PIE 공개 4-fold 체크포인트로 `unseen_ctx`/`unseen_pert`/`unseen_ctx_pert` 평가를 그대로 재실행해 공개 지표(§4)와 일치하는지 확인한다. 일치하지 않으면 그 자체가 보고 대상이다.
2. **G1-b(1일, CPU):** 같은 split에서 기준선 묶음(no-change, context-mean, perturbation-mean, nearest-cell-line, GO-ridge)을 우리 코드로 계산한다. **PIE 대비 기준선 격차**가 R1 진행 여부의 1차 Gate다.
3. **G3(1–2일):** HepG2 1개 fold만으로 PIE 재학습 + State ST-HVG 평가를 수행한다(TxPert의 HepG2 UC 결과가 nearest-cell-line과 거의 동등했으므로 **가장 보수적인 fold**다).
4. **G4:** 살아남은 후보 2개 이하로 4 fold × seed 반복. 지표는 Systema(perturbed-centroid 기준), DEG 가중 MSE, 효과 크기 층화를 함께 보고한다.

---

## 7. 표 (vi) 계획서 인용 검증

검증일 2026-11-10. 모든 URL은 HTTP 200으로 접근 확인했다.

| # | 계획서 기재 | 검증 결과 | 판정 |
|---|---|---|---|
| 1 | Ahlmann-Eltze et al., *Nature Methods* (2025), 단순 선형 기준선 비교 | "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines", Ahlmann-Eltze C., Huber W., Anders S., *Nat Methods* **22**:1657(2025), 온라인 2025-08-04. 초록 주장 일치 | **정확** |
| 2 | Wei et al., *Nature Methods* (2026), 27개 방법·29개 데이터 | "Benchmarking algorithms for generalizable single-cell perturbation response prediction", 제1저자 Wei Zhiting, 교신 Liu Qi. *Nat Methods* **23**:451(2026-02호), **온라인 2025-12-11**. 27 방법·29 데이터·6 지표 일치 | **정확**(온라인 공개는 2025-12임을 병기 권장) |
| 3 | **Tian et al.**, *Science Advances* (2026), 25개 데이터·3개 과제 | 실제 제1저자는 **Lanxiang Li**이고 Luyi Tian은 마지막(교신) 저자다. "A systematic comparison of single-cell perturbation response prediction models", *Sci Adv* **12**(37):eaed3414(2026-09), DOI 10.1126/sciadv.aed3414. 25개 데이터·3개 과제·"세포 간 전이에서 일관되게 우수한 방법 없음" 주장은 일치. **방법 수는 13개 + 기준선**이다 | **저자명 수정 필요** → "Li et al." |
| 4 | *Systema*, *Nature Biotechnology* (2026) | "Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation", Viñas Torné R. … Brbić M., *Nat Biotechnol* **44**:1050, **온라인 2025-08-25**, 2026-06호. 주장 일치 | **정확**(온라인 2025-08 병기 권장) |
| 5 | Wenkel et al., *TxPert*, *Nature Biotechnology* (2026) | "TxPert: using multiple knowledge graphs for prediction of transcriptomic perturbation effects", Wenkel F. … Denton A.K., 온라인 2026-05-01. 다중 KG·cross-cell 주장 일치 | **정확** |
| 6 | Adduri et al., *State*, Cell (2026), URL = arcinstitute.org/tools/state | 논문 실재 확인: Adduri A.K. … Roohani Y.H., ***Cell* 189(19):5914-5931.e20 (2026-09-17), DOI 10.1016/j.cell.2026.07.052**. 제시된 URL은 논문이 아니라 Arc 도구 페이지다(접근은 됨) | **보강 필요** → DOI 또는 PubMed 42673963 병기 |
| 7 | Verma et al., *PIE* (2026 preprint) | "PIE: Generalizing perturbation effects across unseen perturbations, contexts and datasets", Verma R., Adduri A., Bevilacqua B., Eraslan B., Burke D.P., Goodarzi H., Roohani Y.H., bioRxiv 2026-10-05, **DOI 10.64898/2026.10.02.756297**. 두 URL 모두 접근 가능 | **정확**(DOI 추가 권장) |
| 8 | Replogle/Nadig 4-line 전처리(HF `PIE_replogle_nadig_essential`, scPertEval datasets.md) | 둘 다 실재. HF: K562 179,753 / RPE1 167,372 / HepG2 90,826 / Jurkat 173,260 세포, 6,642 유전자 축, 5,459 쌍, DE parquet + 전처리 배열. scPertEval: `replogle22k562/rpe1`, `nadig25hepg2/jurkat`, 세포 수(예: nadig25hepg2 133,757) 수록. **주의: HF 자료는 DE 테이블·요약 배열이고 raw counts가 아니다**(PIE README의 raw counts 링크는 `[PLACEHOLDER]`로 비어 있음). 세포 수가 두 출처 간 다르므로 **전처리 기준이 다름**을 전제해야 한다 | **정확, 단 단서 추가 필요** |
| 9 | GSE264667 HepG2/Jurkat | 실재. 제목 "Transcriptome-wide characterization of genetic perturbations", 2024-04-23 제출, 연계 논문 PubMed 40259084 = Nadig et al., *Nat Genet* 57:1228(2025). Jurkat·HepG2 Perturb-seq, dual-sgRNA, DepMap common essential 중심 | **정확** |
| 10 | Tahoe-100M HF + arc-virtual-cell-atlas README | 둘 다 접근 가능. HF 데이터셋 라이선스 **CC0-1.0** | **정확** |
| 11 | Cellosaurus SNU-423 (CVCL_0366) | 실재. Adult hepatocellular carcinoma, 남성 40세, DepMap ACH-000493. 참고로 HepG2(CVCL_0027)는 "Problematic cell line: Misclassified … hepatoblastoma"로 등재되어 계획서 4절 서술과 일치 | **정확** |
| 12 | 2026 Virtual Cell Challenge 공식 OOD 설명 | 실재(2026-08-20 게시). zero-shot, 대회 전용 학습셋 없음, 미관측 6개 세포주 CRISPRi, 참가자에게는 non-targeting control 발현만 제공, cell-eval 6개 지표. 최종 제출 2026-11-05, 수상자 11월 중하순 발표 예정 | **정확**(결과는 2026-11-10 현재 미공개) |
| 13 | 현재 연구 결과(honbul/vcell_liver FINAL_REPORT, 커밋 540ec93) | URL 접근 가능 | **정확** |

**본문(참고문헌 외)에서 보강이 필요한 서술**

| 위치·계획서 서술 | 검증 결과 |
|---|---|
| 2절 표 "유전자 그래프: GEARS, TxPert — 미관측 유전자·조합·**다른 세포주로의 전이**" | GEARS는 세포주 전이를 지원하지 않는다. 세포주 조건화 입력이 없고 TxPert도 UC 과제에서 GEARS를 비교군으로 쓰지 않았다. 계획서 3절의 "GEARS는 지원 범위를 검증해야 한다"는 단서는 적절하며, 2절 표도 같은 단서를 달아야 한다 |
| 2절 표 "사전학습 Transformer: scGPT, Geneformer, scFoundation, PertAdapt" | Geneformer는 발현 response 예측기가 아니다(임베딩 전용, 별도 decoder 필요). PertAdapt는 실재하는 2025-11 bioRxiv preprint이며 frozen FM + adapter 구조다[2차] |
| 2절 "2026년 *Science Advances* 비교는 25개 데이터셋…" | 데이터·과제 수는 맞으나 **방법 수는 13개 + 기준선 4종**이다. 제1저자는 Li |
| 4절 "Tahoe-100M SNU-423 … **후보 확인**" | SNU-423은 Tahoe-100M 50개 세포주에 실제로 포함되며 PIE 전처리본 기준 1,501,700 세포 · 1,137 약물·용량 조건이다. 계획서의 "후보 확인" 표기는 더 강하게 "포함 확인"으로 승격 가능 |
| 5절 R3 "다른 49개 세포주에서 학습" | Tahoe-100M이 50개 세포주이므로 수치 일치 |

---

## 8. 산출물

- 본 문서: `reports/feedback/F2_LITERATURE_REVIEW.md`
- 기계 판독용 표: `reports/feedback/F2/model_matrix.tsv`, `reports/feedback/F2/code_checkpoint_status.tsv`, `reports/feedback/F2/citation_verification.tsv`

**미확인 항목 정리:** ① Wei et al. 2026의 방법별 순위(유료 본문), ② TxPert의 GPU 사양·학습 시간, ③ State ST의 Replogle 학습 GPU 시간, ④ CellFlow의 LOCO 검증 여부, ⑤ PIE preprint 본문의 TxPert/State 대비 직접 비교표, ⑥ chemCPA 체크포인트 zip의 실제 다운로드 가능 여부, ⑦ VCC 2026 결과.
