# 첫 데이터 변경 비교

사전 등록한 fold0/seed42에서 D0 macroF1은 validation .9389, outer development .9324였다. 같은4096 training cells와 고정 C=.01/class_weight=balanced에서 donor-balanced는 .9281/.9161, donor×class-balanced는 .9183/.9013이었다. validation ID hash와 평가 순서·개수, head 설정과 checkpoint 일치를 실제 파일로 확인했다.

두 변경 모두 이번 screen에서는 개선되지 않았다. D1은 변경 후보 중 validation 점수가 높아 후속 paired factorial의 데이터 후보로만 선택했다. 이를 개선 성공으로 표기하지 않는다. 기존 balanced loss와 sampling의 상호작용 및 자연 분포에서의 precision/recall 변화를 후속 결과와 함께 해석한다. 전 fold/3seed 확인과 같은 NN head factor 대조는 남아 있다. 외부 final 점수는 관찰하지 않았다.
