# 간암 Virtual Cell 연구 요약 — V3 보완

개발 분류, 공식 실행, 실측 genetic/chemical response, 48조건 요인 대조, V3 보완과 **원래 고정한 외부8후보 최종 평가**를 실제 실행·집계했다. 단일 진입점은 [결과 인덱스](RESULTS_INDEX.md), 외부 수치·불확실성은 [최종 외부 결과](external_final.md)에 있다. ScFoundation 접근 차단과 공식 MS 수치 불일치는 해결된 것처럼 포장하지 않는다. V1 main243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79의 기존6개 노트북·원 모델 환경·결과를 보존했다.

## 실용적인 후보와 PCA 대비 이득

개발8명/63036cells의4fold OOF에서 같은12000개 자연 표본으로 head를 학습했다. PCA .93483, scGPT .93770, Geneformer .94708, CLcancer .94584, scVI .93786 macroF1이었다. Geneformer−PCA는 +.01225,8donor paired95CI[+.00010,+.02215]다. 이미 반복 확인한 개발 데이터의 모델 고정 bootstrap이며 선택 편향·재학습 변동을 포함하지 않는다. scGPT−PCA CI는[−.00246,+.00653]다.

실측 결과를 바탕으로 다음 운용 검토 후보는 **PCA와 frozen scGPT**다. PCA는 저비용 기준선이고 scGPT는 외부 고정 비교에서 가장 높은 점추정치를 보였다. 개발에서 미리 정한 primary는 Geneformer로 그대로 보고하며 사후 변경하지 않는다. 전체 개발 frozen 추출은 A30에서 scGPT210.5초/peak1.29GiB, Geneformer1665.5초/3.49GiB였다. Head fitting과 별도 구간이며 사전학습 비용은 미측정이다. 외부 scGPT−PCA fixed-six 차이 +.01517 CI[+.00750,+.02616], primary Geneformer−PCA +.00720 CI[−.00128,+.01766]이다. 개발과 외부의 순위가 달라 보편적 우위를 주장하지 않는다. 비교 입력·feature 차원·reference 정보량이 다르므로 architecture만의 비교도 아니다.

외부50023cells의 A40 frozen feature 추출은 scGPT149.6초/1.33GiB, Geneformer1293.7초/3.48GiB였다. 순수 inference 구간이며 preprocessing/tokenization·head fit은 별도다. Geneformer feature는 DM00/10에도 공유했으므로3번 추출한 비용으로 합산하지 않는다. Partial DM01/11의 online encoder는 각각1118.7/1118.6초, process peak3.68GiB였다. 현재 process와 공유 feature 비용은 [원 resource 표](external_final/resource_summary.tsv) 및 [측정 범위](resource_scope.md)를 따른다.

## 어떤 subgroup에서 한계가 보였는가

Geneformer B recall .87737은 PCA .90889보다 낮다. 조직 위치는 원본 `site`가 canonical `tissue_site`로 변환되어 있었다. 개발63036cells에서 원본과 정확히 일치함을 확인하고 기존 OOF 예측으로 donor×site×lineage 오류를 추가했다. Geneformer site별 F1은 Normal .95300, Tumor .93323, PVTT .90401, Lymph .94201이다. PVTT는2명, Lymph는1명이므로 독립적 site 일반화 효과를 추정하지 않는다. HCC03 Tumor T/NK203cells의 recall .24631처럼 pooled score 아래에 큰 국소 오류가 남는다.

발현량·vocabulary coverage와 오류의 연관은 class/donor 구성과 얽힌다. B 내부에서 vocabulary count coverage 최하위 사분위 recall .99482, 최상위 .81865였으므로 누락 immunoglobulin token이 낮은 성능의 원인이라고 단정하지 않는다. 자세한 support·confusion·반례는 [개발 해석](development_insights.md)에 있다.

## 간/암 특화의 추가 이득

개발에서 CLcancer−base Geneformer는 −.00124, paired95CI[−.00367,+.00145]로 추가 이득이 확인되지 않았다. LiverCT는 외부 labeled liver reference와20epoch 비라벨 query adaptation을 사용한 별도 regime이다. Native F1 .90167/coverage .88710, 같은 adapted SCANVI head 대조 .70609/coverage .80318이었다. Native broad head는30차원 latent의 RF/MLP/OVR/XGB soft-voting ensemble이다. Reference registry331127cells/21batches/8labels는 정확한 ensemble fitting label 수를 뜻하지 않는다. Unknown은 주로 고정6개 ontology 밖 Cholangiocyte mapping이며 confidence rejection 검증이 아니다. 외부 reference label 양과 adaptation이 달라 inductive 모델과 구조만의 효과로 비교하지 않는다.

외부 GSE151530에서는 CLcancer−base +.00437 CI[+.00162,+.00892]로 방향이 바뀌었다. 이 고정 비교의 이득은 보고하되 모든 간암 cohort의 특화 효과 또는 암 세포 기능 예측의 검증으로 확대하지 않는다. 개발과 외부 평가 모두 사전학습 corpus overlap은 unknown이다.

scFoundation은 공식 두 SharePoint 경로403이고 확인한 Figshare 예제 archive에는 weights가 없어 차단됐다. 대체 checkpoint를 임의로 사용하거나 실행 score를 만들지 않았다.

## 데이터와 모델 변경의 기여

4096개 학습 표본 조건에서 동결NN/partial과 자연/donor-balanced를4fold×3seed×4조건, 총48실행으로 비교했다. 첫 fold의 inner-validation만으로 partial 후보를 선택했고 실제 LoRA22모듈의 gradient/update·저장복원도 검증했다. 선택된 후보와 확인된 개선은 구분한다.

| 조건 | 3seed pooled F1 평균±SD |
|---|---:|
| DM00 자연+동결NN | .94399±.00333 |
| DM10 donor-balanced+동결NN | .94007±.00891 |
| DM01 자연+partial | .94544±.00105 |
| DM11 donor-balanced+partial | .93531±.00490 |

Data gain −.00392 CI[−.01060,+.00259], Model gain +.00145 CI[−.00223,+.00584], Combined −.00869 CI[−.02010,+.00186], Interaction −.00621 CI[−.01052,−.00131]다. 모델만 변경한 효과도8명 중4명에서만 비음수이며 확실한 개선으로 부르지 않는다. Partial12실행의 현 process wall 합은 약11762초로 frozen12실행579초보다 컸고, peak process GPU는5.08GiB 대 .30GiB다. Shared frozen extraction은 이 합계 밖에 있다.

외부 고정4096 refit에서는 Data−.01747 CI[−.02607,−.00922], Model+.00640 CI[+.00293,+.01167], Combined+.00124 CI[−.00708,+.01106], Interaction+.01231 CI[+.00105,+.02282]였다. Partial은 외부 model-only 이득이 있었지만 개발 반복에서 확실하지 않았고, 결합은 외부에서도 CI가0을 포함한다. 따라서 donor balancing·결합을 확인된 일반적 개선으로 부르지 않는다. 외부는 seed42의 한 번의 refit이며25donor bootstrap에 재학습 변동과 다중 비교 보정은 없다.

Data-only D2 donor×class sampling도 첫 validation에서 감소해 선택하지 않았다. 세 번째 데이터 가설 KG PCA32는4fold/seed42의 추가 탐색이며 real32 .94659 대 raw128 .94696/random32 .94776으로 지지되지 않았다. 외부 후보 선택에서 제외했다. 실제 gene–pathway 평균 특징의 계산은 [설명](kg_feature_method.md)에 있다. 선택과 무관한 KG 완료를 외부 release 필수 조건으로 둔 코드는 수정했으며, 원 external freeze와 예측을 재발행하지 않았다.

## 실제 intervention을 어디까지 검증했는가

공식 MS scGPT는 reference7844/query13468,10epochs/batch32를 실행했지만 F1 .67558 대 원 notebook .732로 불일치한다. 제한 점검에서 입력 population·원 concatenation 후 ID/label 순서·metric 계산·기본 설정 차이는 찾지 못했다. Non-flash backend와 loader/runtime 차이를 원인별로 분리하지 않았으므로 수치 재현 성공이 아니다. 공식 GEARS Norman은 native simulation flow20epochs를 실행했으며 K562 CRISPRa 결과로 분리한다.

Strict HepG2는 day7 단일-target CRISPRi이며 hepatoblastoma-derived 세포주다. Full214testtargets/2000genes에서 GO/PCA/ridge RMSE .10584, GEARS .13091이었다. 별도 matched-native191targets/primary1977genes에서는 GO/ridge .11073, GEARS .13732, scGPT pretrained .12394, random .12268이다. Pretrained의 random 대비 RMSE gain −.00126,191intervention CI[−.00328,+.00086]로 사전학습 이득이 확인되지 않았다. 이는 같은 architecture·입력·학습 budget의 대조다.

**Simple-vs-native 순위에는 control 평균화 차이가 포함된다.** V3에서 GO/ridge의 full-control mean 대신 같은 recorded sampled-control mean을 넣자 RMSE가 .13050으로 바뀌었다.20resampling seed 평균은 .13007±.00046이다. 고정한 첫 target AAR2에서 saved native 두 모델의 full-mean 입력도 작은 추론으로 확인했으나 한 target만으로 전체 순위를 바꾸지 않는다. [감도 설명](control_sensitivity.md)에서 실제 입력 교체와 단순 평가 reference 변경을 구분한다. Top20 response recovery는 관측 mean Delta의 평가용 순위이며 FDR DEG 검증이 아니다. Biological replicate assignment는 unknown이다.

Mouse TCDD 조건부 경로에서는 portal-treated1074cells holdout에서 같은 VAE의 native arithmetic RMSE .13854, scVIDR regression .07629였다. 다른 celltype에서 관측한 **같은 TCDD30µg/kg·28일** 반응을 옮긴 결과이며 새 약물·새 dose·사람 HCC 약효 일반화가 아니다. 동물별 assignment가 없어 biological replicate CI를 제시하지 않는다. Geneformer 가상 삭제는 별도의 embedding 변화 탐색으로 남는다. HCC 전체 dynamics나 임상 치료효과 simulator를 검증한 것은 아니다.

## 외부 평가 범위와 다음 실험 하나

GSE151530은 HCC25명/32biopsies/50023cells 중 알려진 **다섯 비악성 세포 계열 전이** 평가다. Known truth27212cells(54.40%), Unknown malignant/unclassified22811cells(45.60%)이며 gold Hepatocyte lineage가 없다. 사전 정의 fixed-six F1의 ceiling은5/6이고 fixed-five는 보조 지표다. 어느 지표도 hepatocyte 전이나 malignant 세포의 정답 lineage를 검증하지 않는다. 원 freeze 이후 categorical storage reader 실패를 실제64-cell 확률 동등성으로 검증하고 동일8후보 batch만 복구해 완료했다.

12000-label 조건의 fixed-six/fixed-five는 PCA .80987/.97184, scGPT .82503/.99004, primary Geneformer .81707/.98048, CLcancer .82143/.98572다. 네 후보의 prediction coverage는100%이지만 gold coverage54.40%와 다르다. 개발6개 정답 계열과 외부의 absent Hepatocyte가 있는 fixed-six를 그대로 빼 성능 하락으로 해석하지 않는다. Unknown의 예측 분포는 악성 판정 성능이 아니다. 모든8후보·donor별 CI·calibration·cost는 [외부 결과](external_final.md)에 공개하며, 원 evaluation·prediction을 다시 만들지 않았다. 이 cohort를 이후 튜닝에 쓰면 새로운 독립 최종 검증이라고 부르지 않는다.

다음 실험 하나는 **같은 control estimator와 biological replicate holdout을 갖춘 liver-line intervention 대조**를 권한다. 현재191target 감도에서 control 평균화의 영향이 native pretrained–random 차이보다 크게 나타났고 biological replicate 불확실성은 측정하지 못했기 때문이다. 새 모델을 더하기 전에 같은 제공 control 정보·평균화와 사전 고정 선택/평가 단위를 맞추는 것이 이 연구의 다음 미해결 질문에 직접 답한다. 이번 V3에서는 해당 전체 재학습을 시작하지 않았다.
