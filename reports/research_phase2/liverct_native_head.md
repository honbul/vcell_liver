# LiverCT 실제 분류 경로와 coverage

실제 점수에 사용한 native 출력은 `test_function.LabelPredict()`의 `pred_lv1`이다. 해당 단계는 이름이 `mlp_clf_lv1.pkl`인 파일을 읽지만, 저장 객체의 실제 type은 sklearn `VotingClassifier`다. RandomForestClassifier, MLPClassifier, OneVsRestClassifier, XGBClassifier 네 member의 **unweighted soft voting**이며 입력은30차원 적응 SCANVI latent다. 폴더 이름의 `svm`이나 file 이름의 `mlp`만으로 scored head를 단일 SVM/MLP라고 부르면 부정확하다. 기존 문서의 SVM 표현을 바로잡았다.

SCANVI reference 저장 registry는331127cells,21batches,2000genes,8개 lineage category다. Exact native ensemble fitting example 수는 검사한 checkpoint에 따로 보존되어 있지 않아 reference registry 크기와 같다고 주장하지 않는다. Reference annotation 정보량이 개발 HCC12000-label 조건과 맞지 않으며 query20epoch adaptation도 사용한다. Strict inductive 표와 분리한다.

Native8개 출력 중 B/Plasma B를 B로 묶고 TNK를 T/NK로 대응시킨다. Cholangiocyte는 개발 cohort의 고정6개 lineage 밖이므로 Unknown이다. 보고한 prediction coverage .8871은 **이 ontology mapping의 coverage**이며 confidence 기반 rejection의 정확도나 OOD 검출 성능이 아니다. 같은 적응 모델 SCANVI head의 .8032도 같은 규칙이다. Native의 별도 suggest_label threshold 및 disease/transition 출력은 이6-lineage metric에 사용하지 않았다.

64-cell replay의 첫 시도는 epithelial 예측이 없는 slice에서 unscored auxiliary transition SVM이 빈 배열을 받아 실패했다. 작은 replay는 원 함수의 동일한 broad lv1 classifier 단계를 직접 호출한다. 기존4fold native 실행과 score를 바꾸지 않으며, 같은 원 query cohort에서 이미 적응한 checkpoint를 재추론한다. 이를 새로운 query에 대한 zero-adaptation 결과로 해석하지 않는다. 실패와 수정 후 실제64-cell 비교는 별도 probe 파일에 기록한다.
