**질문** — 두 공개 스크린에서 동결 H1-v2 K=5 앵커를 물리 웰에서 재구성할 수 있는가?<br>
**실제 바뀐 것** — native 시료와 exact parent를 물리 농도·시간·반복 키에 연결했다.<br>
**주 비교 결과·구간** — 완전 K5 지원은 CTRPv2 271/622, GDSC2 679/680계통이다. CTRPv2 Reduced-8pt 92웰, ΔNDCG −0.0004 [−0.0013, +0.0004], ΔPrecision −0.0007 [−0.0054, +0.0040].<br>
**판정** — Q8 유지. CTRPv2 Dual 8점 압축 통과. GDSC2 8점과 패널 전체 2반복은 지원 없음.<br>
**다음 결정** — 확인된 생존율 분모에서 해석한다. LICOB 3D의 농도별 원자료 지원은 확인되지 않았다.<br>

# E2 원곡선 지원과 적합 경계

CTRPv2는 저자 공개 per_cpd_well, per_cpd_post_qc, 실험·plate·시료·화합물 메타데이터를 사용했다. SNP-matched 실험과 master_ccl_id를 DepMap 모델에 연결했다. 웰 키는 실험 ID·native 약물·µM·plate barcode다. 생존율은 DMSO 차감 로그 신호를 2의 거듭제곱으로 환산해 [0,1]에 제한했다. 처리 후 72시간은 공개 논문의 공통 설계다. QC 파일에서는 농도 키만 사용하고, 모든 농도에서 두 웰로 복원되는 16점 곡선을 주 분석에 적용했다. [CTRP 원자료](https://portals.broadinstitute.org/ctrp.v2.1/) · [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4631646/)

GDSC2는 공개 27Oct23 raw archive의 Sanger 모델 ID·native drug ID·barcode·scan·제작일·라이브러리·well position·µM 키를 사용했다. 시간은 배포 파일의 DURATION=4 필드를 그대로 표기한다. Glo, 차량 대조 ≥8개, blank ≥4개, Z′≥0.3과 등록 물리 농도 범위 양 끝을 확인했다. 두 스크린의 native 약물 ID는 vc_build_v1의 exact-parent 등록에 연결했다. [GDSC 배포 설명](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [종말점 프로토콜](../vc_build_v1/v0plus/LICOB_DOSE_ENDPOINT_PROTOCOL.md) · [곡선 전이](../vc_build_v1/v0plus/LICOB_CURVE_TRANSFER_RESULTS.md)

| 스크린 | 공개 웰 키 | 앵커 수치 웰 | QC 통과 곡선 | 전 농도 두 웰 | 불완전 탈락 곡선 |
| --- | --- | --- | --- | --- | --- |
| CTRPv2 | 1512411 | 333839 | 8965 | 8798 | 1759 |
| GDSC2 | 574896 | 118993 | 16065 | 686 | 62 |

**원래 H1-v2 전체 분모와 다른 원곡선 지원 서브셋이다.** A8 전체를 감사하고 두 팔의 고정 K5 합집합에 완전한 계통만 포함했다. GDSC2 첫 폴드에서 팔별 K5가 달라 합집합을 적용했다. 원 Q8의 60개 후보와 계통별 기존 Q 관측 마스크는 유지했다. Q 원자료 매칭은 식별 키 감사다. GDSC2 Q 원자료에는 plate 수치 QC 적용 전 키가 포함된다. 스크린 단위 추출은 폴드들의 A8 합집합을 읽는다. 각 폴드의 입력과 적합은 그 폴드의 앵커 약물 및 허용 계통에 한정했다. 한 약물이 다른 폴드의 앵커가 될 수 있다. 해당 폴드의 Q 반응값은 예측 봉인 뒤 기존 H1 Q8 정답으로 채점할 때 사용했다. [예측 봉인](e2_forecast_seal_receipt.json)

| 폴드 | E2/원 평가 계통 | Inner 적합 계통 | K5 두 웰 계통 | Q8 raw 약물/60 | Q8 raw 시료×약물/등록 | atlas 적합 n | atlas native 실측쌍 | atlas raw 처리 웰 키 | atlas 12대조/계통 가정 | atlas 처리+가정 대조 합계 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRPv2-0 | 105/204 (48.5% 제외) | 320 | 105 | 60 | 12202/12361 | 480 | 31006 | 1008120 | 5760 | 1013880 |
| CTRPv2-1 | 96/214 (55.1% 제외) | 319 | 96 | 60 | 12585/12625 | 479 | 30672 | 1001461 | 5748 | 1007209 |
| CTRPv2-2 | 70/204 (65.7% 제외) | 320 | 70 | 60 | 11995/12033 | 481 | 30884 | 1015241 | 5772 | 1021013 |
| GDSC2-0 | 220/220 (0.0% 제외) | 319 | 0 | 60 | 12496/12658 | 479 | 30945 | 364728 | 5748 | 370476 |
| GDSC2-1 | 232/233 (0.4% 제외) | 312 | 0 | 60 | 13047/13254 | 468 | 30438 | 388458 | 5616 | 394074 |
| GDSC2-2 | 227/227 (0.0% 제외) | 320 | 0 | 60 | 12890/13055 | 481 | 30973 | 396606 | 5772 | 402378 |

기존 68약물 reference atlas를 재사용해 신규 구축에 든 웰은 0이다. 표의 처리 웰 키 수는 atlas 계통에 물리 키가 연결되는 공개 archive의 관측값이다. 폴드와 스크린 간에 계통이 겹친다. 대조 열은 재구축 시 12대조/계통을 가정하며 과거 실험의 실제 대조 웰 총수와 구별한다.

## A8 exact parent별 지원

원자료 미매칭·curve 탈락은 시료×약물 값이고, 적격과 2웰은 약물별 계통 수다. 전체 K5 반복 지원은 앞 표에 있다. 약물별 Full·8·6·3·단일점의 실제 µM, fit-only 변환 계수, 반복에서 직접 관측한 AUC 및 웰 변동, 원파일과 SHA-256은 영수증에 있다. [지원 영수증](curve_support_receipt.json)

| 폴드 | 공개 약물명 · exact parent | native ID | K5 용도 | Full µM / 점 | Full / 단일 적합 n | 등록쌍 | raw 미매칭 | raw 후 curve 탈락 | 잠금 grid 적격 | 2웰 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRPv2-0 | Sorafenib · MLDQJTXFUGDVEO-UHFFFAOYSA-N | 349006 | A8만 | 0.002–66 / 16 | 233/233 | 209 | 4 | 45 | 151 | 151 |
| CTRPv2-0 | Rapamycin · QFJCIRLUMZQUOT-HPLJOQBZSA-N | 56703 | AnchorOnly, Dual | 0.001–33 / 16 | 258/258 | 209 | 4 | 33 | 161 | 161 |
| CTRPv2-0 | ciclopirox · SCKYRAXSEDYPSA-UHFFFAOYSA-N | 28784 | AnchorOnly, Dual | 0.002–66 / 16 | 247/247 | 210 | 0 | 37 | 166 | 166 |
| CTRPv2-0 | Topotecan · UCFGDBYHRUNTLO-QHCPKHFHSA-N | 44580 | AnchorOnly, Dual | 0.002–66 / 16 | 250/250 | 210 | 4 | 29 | 169 | 169 |
| CTRPv2-0 | clofarabine · WDDPHFBMKLOVOX-AYQXTPAHSA-N | 660288 | A8만 | 0.002–66 / 16 | 243/243 | 210 | 4 | 41 | 159 | 159 |
| CTRPv2-0 | PD318088 · XXSSGBYXSKOLAM-UHFFFAOYSA-N | 660410 | A8만 | 0.002–66 / 16 | 294/294 | 210 | 0 | 5 | 196 | 196 |
| CTRPv2-0 | leptomycin B · YACHGFWEQXFSBS-XYERBDPFSA-N | 660086 | AnchorOnly, Dual | 0.002–66 / 16 | 232/232 | 210 | 4 | 47 | 151 | 151 |
| CTRPv2-0 | ceranib-2 · YIJGNIIQDRGGFT-DTQAZKPQSA-N | 660078 | AnchorOnly, Dual | 0.002–66 / 16 | 262/262 | 210 | 0 | 22 | 179 | 179 |
| CTRPv2-1 | Fluorouracil · GHASVSINZRGABV-UHFFFAOYSA-N | 25344 | A8만 | 0.0041–130 / 16 | 236/236 | 216 | 0 | 47 | 166 | 166 |
| CTRPv2-1 | Rapamycin · QFJCIRLUMZQUOT-HPLJOQBZSA-N | 56703 | AnchorOnly, Dual | 0.001–33 / 16 | 234/234 | 216 | 1 | 34 | 173 | 173 |
| CTRPv2-1 | ciclopirox · SCKYRAXSEDYPSA-UHFFFAOYSA-N | 28784 | A8만 | 0.002–66 / 16 | 237/237 | 216 | 0 | 46 | 168 | 168 |
| CTRPv2-1 | Topotecan · UCFGDBYHRUNTLO-QHCPKHFHSA-N | 44580 | AnchorOnly, Dual | 0.002–66 / 16 | 243/243 | 216 | 1 | 34 | 176 | 176 |
| CTRPv2-1 | clofarabine · WDDPHFBMKLOVOX-AYQXTPAHSA-N | 660288 | A8만 | 0.002–66 / 16 | 237/237 | 215 | 1 | 36 | 173 | 173 |
| CTRPv2-1 | PD318088 · XXSSGBYXSKOLAM-UHFFFAOYSA-N | 660410 | AnchorOnly, Dual | 0.002–66 / 16 | 288/288 | 216 | 0 | 12 | 200 | 200 |
| CTRPv2-1 | leptomycin B · YACHGFWEQXFSBS-XYERBDPFSA-N | 660086 | AnchorOnly, Dual | 0.002–66 / 16 | 238/238 | 216 | 1 | 59 | 153 | 153 |
| CTRPv2-1 | ceranib-2 · YIJGNIIQDRGGFT-DTQAZKPQSA-N | 660078 | AnchorOnly, Dual | 0.002–66 / 16 | 252/252 | 216 | 0 | 29 | 182 | 182 |
| CTRPv2-2 | 17-AAG · AYUNIORJHRXIBJ-TXHRRWQRSA-N | 50134 | AnchorOnly, Dual | 0.001–33 / 16 | 236/236 | 206 | 1 | 39 | 161 | 161 |
| CTRPv2-2 | GDC-0941 · LHNIIDJUOCFXAP-UHFFFAOYSA-N | 639759 | AnchorOnly, Dual | 0.002–66 / 16 | 291/291 | 206 | 1 | 10 | 192 | 192 |
| CTRPv2-2 | MLN2238 · MXAYKZJJDUDWDS-LBPRGKRZSA-N | 632873 | AnchorOnly, Dual | 0.001–33 / 16 | 231/231 | 205 | 1 | 43 | 155 | 155 |
| CTRPv2-2 | Rapamycin · QFJCIRLUMZQUOT-HPLJOQBZSA-N | 56703 | AnchorOnly, Dual | 0.001–33 / 16 | 240/240 | 206 | 1 | 36 | 157 | 157 |
| CTRPv2-2 | CAY10618 · SJOLTIOPWDLDEB-UHFFFAOYSA-N | 632104 | A8만 | 0.001–33 / 16 | 281/281 | 206 | 1 | 16 | 184 | 184 |
| CTRPv2-2 | 1S,3R-RSL-3 · TXJZRSRTYPUYRW-NQIIRXRSSA-N | 609060 | AnchorOnly, Dual | 0.001–33 / 16 | 250/250 | 206 | 1 | 47 | 152 | 152 |
| CTRPv2-2 | PD318088 · XXSSGBYXSKOLAM-UHFFFAOYSA-N | 660410 | A8만 | 0.002–66 / 16 | 288/288 | 206 | 0 | 6 | 193 | 193 |
| CTRPv2-2 | Dasatinib · ZBNZXTGUTAYRHI-UHFFFAOYSA-N | 52882 | A8만 | 0.002–66 / 16 | 283/283 | 205 | 1 | 20 | 182 | 182 |
| GDSC2-0 | 17-AAG · AYUNIORJHRXIBJ-TXHRRWQRSA-N | 1026 | AnchorOnly, Dual | 0.0010005–1 / 7 | 310/310 | 220 | 0 | 0 | 220 | 180 |
| GDSC2-0 | GDC0810 · BURHGPHDEVGCEZ-KJGLQBJMSA-N | 1925 | AnchorOnly, Dual | 0.010005–10 / 7 | 314/314 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | VE821 · DUIHHZKTCSNTGM-UHFFFAOYSA-N | 2111 | AnchorOnly, Dual | 0.010005–10 / 7 | 309/309 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | Rucaparib · HMABYWSNWIZPAG-UHFFFAOYSA-N | 1175 | AnchorOnly | 0.0050026–5 / 7 | 310/310 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | JNK Inhibitor VIII · KQMPRSZTUSSXND-UHFFFAOYSA-N | 1043 | A8만 | 0.010005–10 / 7 | 311/311 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | YM155 · OTSOOHRUMBRSHZ-UHFFFAOYSA-N | 1941 | AnchorOnly, Dual | 0.010005–10 / 7 | 314/314 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | Motesanib · RAHBGWKEPAQNFF-UHFFFAOYSA-N | 1029 | Dual | 0.010005–10 / 7 | 310/310 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-0 | PLX-4720 · YZDJQTHVDDOVHR-UHFFFAOYSA-N | 1036 | A8만 | 0.010005–10 / 7 | 316/316 | 220 | 0 | 0 | 220 | 0 |
| GDSC2-1 | 17-AAG · AYUNIORJHRXIBJ-TXHRRWQRSA-N | 1026 | AnchorOnly, Dual | 0.0010005–1 / 7 | 306/306 | 233 | 1 | 0 | 232 | 187 |
| GDSC2-1 | JNJ-26854165 · CEGSUKYESLWKJP-UHFFFAOYSA-N | 1133 | AnchorOnly, Dual | 0.010005–10 / 7 | 306/306 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | Buparlisib · CWHUFRVAEUJCEF-UHFFFAOYSA-N | 1873 | AnchorOnly, Dual | 0.010005–10 / 7 | 309/309 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | Bortezomib · GXJABQQUPOEUTA-RDJZCZTQSA-N | 1191 | A8만 | 2.0011e-05–0.02 / 7 | 309/309 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | SGC0946 · IQCKJUKAQJINMK-HUBRGWSESA-N | 2177 | AnchorOnly, Dual | 0.00050026–0.5 / 7 | 308/308 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | P22077 · RMAMGGNACJHXHO-UHFFFAOYSA-N | 1933 | A8만 | 0.010005–10 / 7 | 309/309 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | Cediranib · XXJWYDDUDKYVKI-UHFFFAOYSA-N | 1922 | AnchorOnly, Dual | 0.010005–10 / 7 | 309/309 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-1 | MIRA-1 · YXEWPGYLMHXLPS-UHFFFAOYSA-N | 1931 | A8만 | 0.010005–10 / 7 | 312/312 | 233 | 1 | 0 | 232 | 0 |
| GDSC2-2 | 17-AAG · AYUNIORJHRXIBJ-TXHRRWQRSA-N | 1026 | A8만 | 0.0010005–1 / 7 | 313/313 | 227 | 0 | 0 | 227 | 186 |
| GDSC2-2 | JNJ-26854165 · CEGSUKYESLWKJP-UHFFFAOYSA-N | 1133 | AnchorOnly, Dual | 0.010005–10 / 7 | 313/313 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | Buparlisib · CWHUFRVAEUJCEF-UHFFFAOYSA-N | 1873 | AnchorOnly, Dual | 0.010005–10 / 7 | 314/314 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | Bortezomib · GXJABQQUPOEUTA-RDJZCZTQSA-N | 1191 | A8만 | 2.0011e-05–0.02 / 7 | 314/314 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | SGC0946 · IQCKJUKAQJINMK-HUBRGWSESA-N | 2177 | AnchorOnly, Dual | 0.00050026–0.5 / 7 | 311/311 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | P22077 · RMAMGGNACJHXHO-UHFFFAOYSA-N | 1933 | AnchorOnly, Dual | 0.010005–10 / 7 | 314/314 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | Cediranib · XXJWYDDUDKYVKI-UHFFFAOYSA-N | 1922 | AnchorOnly, Dual | 0.010005–10 / 7 | 314/314 | 227 | 0 | 0 | 227 | 0 |
| GDSC2-2 | MIRA-1 · YXEWPGYLMHXLPS-UHFFFAOYSA-N | 1931 | A8만 | 0.010005–10 / 7 | 318/318 | 227 | 0 | 0 | 227 | 0 |

LICOB에서는 S5 농도 설계와 S6–S8 AUC·IC50·Emax가 공개 등록돼 있다. 농도별 3D 생존율 원자료는 등록되지 않았다. 2D에서 확인한 물리 농도 지원은 3D의 독립 실측 근거가 아니다. [H1-v2 LICOB 자산 범위](../nextgen_h1_v2/ROUND1_TO_3_CEILING_TUNING.md)
