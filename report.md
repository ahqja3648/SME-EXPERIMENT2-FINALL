# Reliability-aware Multi-Candidate Ensemble Localization Report

| 항목 | 내용 |
|---|---|
| 이름 | 양준범 |
| 학과 | 스마트모빌리티공학과 |
| 학번 | 12223639 |
| 프로젝트 | 스마트모빌리티공학실험2 Final Project |
| 최종 알고리즘명 | Reliability-aware Multi-Candidate Ensemble Localization |

## 1. 모티베이션 & 인트로

본 프로젝트의 목표는 18개 기지국에서 측정된 RTT 기반 거리값을 이용하여 사용자 위치를 추정하는 것이다. 제공 데이터는 정답 사용자 위치, 기지국별 RTT 측정값, 기지국 좌표로 구성되어 있으며, 채점 시에는 학생에게 공개되지 않은 hidden test set으로 main.py가 평가된다. 따라서 제공된 데이터에만 과하게 맞춘 단순 memorization 방식보다는, 측정값의 구조와 오차 원인을 반영하는 일반화 가능한 알고리즘이 필요하다.

사전 분석에서 가장 중요하게 확인한 점은 d_hat이 실제 기하학적 거리와 완전히 일치하지 않는다는 것이다. 일부 측정값은 실제 거리보다 크게 나타나는 양의 bias를 보였고, 특정 사용자와 anchor 조합에서는 큰 outlier가 존재하였다. 이런 상황에서 모든 anchor를 동일하게 사용하는 일반 최소제곱법은 신뢰도가 낮은 측정값 하나 때문에 위치 추정 결과가 크게 흔들릴 수 있다. 따라서 본 실험에서는 거리 측정값을 그대로 좌표 회귀에 넣는 방식뿐 아니라, anchor별 측정 신뢰도 예측, 다중 WLS 후보 생성, 모델 ensemble을 함께 사용하는 구조를 설계하였다.

| 분석 항목 | 확인된 의미 | 알고리즘 설계 반영 |
|---|---|---|
| anchor별 거리 오차 차이 | 모든 기지국이 같은 신뢰도를 갖지 않음 | anchor error prediction model 추가 |
| 일부 d_hat의 큰 outlier | 단일 WLS 결과가 특정 측정값에 끌릴 수 있음 | top-k anchor subset 기반 다중 WLS 후보 생성 |
| 거리 순위와 위치의 관계 | 절대 거리값뿐 아니라 anchor 간 상대적 순서가 위치 정보를 가짐 | rank feature, pairwise distance difference feature 추가 |
| tree 계열과 선형 모델의 장단점 차이 | tree 모델은 비선형 패턴에 강하고 선형 모델은 안정적임 | Ridge, ExtraTrees, GradientBoosting ensemble 사용 |
| hidden test overfitting 위험 | 훈련 데이터에만 맞는 복잡한 단일 모델은 위험 | 여러 모델 비교 후 compact ensemble만 저장 |

## 2. 알고리즘 설명

최종 알고리즘은 하나의 회귀 모델만 사용하는 방식이 아니라, 측정값의 신뢰도를 먼저 예측하고 그 정보를 이용해 여러 위치 후보를 만든 뒤, 서로 다른 성격의 모델을 결합하여 최종 좌표를 예측하는 방식이다. 전체 구조는 anchor error prediction, feature construction, multi-candidate WLS, model comparison, final ensemble의 다섯 부분으로 구성된다.

| 구성 요소 | 역할 | 입력 | 출력 |
|---|---|---|---|
| Anchor error prediction | 각 anchor 측정값의 예상 오차 추정 | anchor별 d_hat, rank, 사용자별 통계, anchor 좌표 | anchor별 predicted error |
| Multi-candidate WLS | 신뢰도 높은 anchor 조합으로 위치 후보 생성 | d_hat, predicted error, BS_positions | 여러 개의 WLS 위치 후보 |
| Feature construction | ML 모델이 사용할 통합 feature 생성 | 원본 거리, 거리 통계, rank, pairwise difference, WLS 후보 | 사용자별 feature vector |
| Model comparison | 여러 회귀 모델의 성능 비교 | train feature, validation feature | validation 결과표 |
| Final ensemble | 가장 안정적인 모델 조합으로 최종 위치 예측 | feature vector | p_hat |

Anchor error prediction은 각 anchor의 측정값이 실제 거리와 얼마나 다를 가능성이 있는지를 학습한다. 학습 데이터에서는 정답 위치 p와 기지국 좌표 BS_positions를 이용해 실제 거리 d_true를 계산할 수 있으므로, anchor별 target error를 |d_hat - d_true|로 정의한다. 이 모델은 위치를 직접 예측하는 것이 아니라, 어떤 anchor가 현재 사용자에 대해 신뢰도가 낮을 가능성이 큰지를 판단하는 보조 모델이다.

| 기호 | 의미 |
|---|---|
| d_i | i번째 anchor의 RTT 기반 측정값 |
| a_i | i번째 anchor의 좌표 |
| p | 사용자 위치 |
| d_true,i | 사용자와 i번째 anchor 사이의 실제 거리 |
| e_i | anchor별 거리 오차 |
| w_i | predicted error에서 변환한 anchor 신뢰도 가중치 |

Anchor별 실제 거리는 d_true,i = ||p - a_i||로 계산하였다. 학습 시 anchor error model은 e_i = |d_i - d_true,i|를 예측하도록 구성하였다. 추론 시에는 정답 위치 p를 사용할 수 없기 때문에, 이 모델이 d_hat과 anchor feature만으로 predicted error를 추정한다. predicted error가 작을수록 해당 anchor를 더 신뢰할 수 있다고 보고, WLS 후보 생성에서 더 큰 가중치를 부여한다.

Multi-candidate WLS는 하나의 WLS 결과만 만드는 것이 아니라 여러 anchor subset에서 위치 후보를 만든다. 전체 anchor를 모두 사용하는 후보, predicted error가 작은 anchor만 사용하는 후보, d_hat이 작은 anchor만 사용하는 후보를 함께 생성하였다. 이렇게 한 이유는 outlier가 많은 상황에서 전체 anchor WLS 하나만 쓰면 잘못된 anchor의 영향이 크게 반영될 수 있기 때문이다.

| WLS 후보 유형 | 사용 anchor 기준 | 목적 |
|---|---|---|
| All-anchor reliability WLS | 전체 anchor와 predicted error 기반 weight | 전체 측정값을 사용한 기준 후보 |
| Error top-k WLS | predicted error가 작은 anchor subset | 신뢰도 높은 측정값 중심 후보 |
| Distance top-k WLS | d_hat이 작은 anchor subset | 가까운 anchor 중심의 지역 후보 |
| Residual statistic feature | 각 후보 위치에서 residual 분포 계산 | 후보 위치가 전체 측정값과 얼마나 일관적인지 표현 |

최종 feature는 원본 d_hat만 사용하지 않고, 측정값의 상대적 구조와 기하학적 정보를 함께 담도록 구성하였다. 특히 pairwise distance difference는 좌우 anchor 또는 상하 anchor 사이의 거리 차이가 위치 방향성을 담는다는 점을 반영하기 위한 feature이다. rank feature는 절대 거리 scale이 흔들리더라도 상대적으로 가까운 anchor의 순서가 위치 정보를 제공할 수 있다는 점을 반영한다.

| Feature 그룹 | 포함 내용 | 사용 이유 |
|---|---|---|
| Raw distance feature | d_hat 18개 | 기본 RTT 측정 패턴 반영 |
| Nonlinear transformed distance | log(d_hat), sqrt(d_hat), squared distance scale | 거리 scale 변화와 비선형 관계 반영 |
| User statistic feature | 평균, 중앙값, 표준편차, 최솟값, 최댓값, 분위수 | 사용자별 측정 안정성 표현 |
| Rank feature | anchor별 거리 순위 | 절대값이 흔들려도 상대적 위치 정보 유지 |
| Top-k anchor mask | 가까운 anchor 집합 표시 | 지역적 anchor 그룹 정보 반영 |
| Inverse-distance center | 거리 역수 기반 anchor 중심 | 대략적 위치 힌트 제공 |
| Pairwise difference | anchor 간 d_hat 차이 | x, y 방향성 학습 보조 |
| Predicted anchor error | anchor별 예상 측정 오차 | 신뢰도 낮은 측정값의 영향 완화 |
| WLS candidate position | 여러 anchor subset으로 계산한 위치 후보 | 기하학 기반 후보 정보를 ML에 제공 |
| Candidate residual statistics | 후보별 residual 평균, 중앙값, 최댓값 등 | 후보의 일관성과 outlier 정도 표현 |

최종 모델은 단일 모델 하나를 사전에 정하지 않고 여러 후보를 같은 validation split으로 비교하였다. 비교 모델에는 선형 모델, 부분최소제곱 회귀, KNN, Random Forest, Extra Trees, Gradient Boosting, 그리고 세 가지 ensemble 후보가 포함된다. 최종적으로 validation RMSE가 가장 낮은 weighted ensemble을 선택하였다.

| 최종 선택 모델 | 구성 모델 | 결합 방식 | 선택 이유 |
|---|---|---|---|
| ensemble_ridge_extra_gbr | Ridge alpha 50, Extra Trees 60, Gradient Boosting 60 | validation RMSE의 역수 기반 weighted average | 선형 안정성, tree 기반 비선형성, boosting의 median error 개선 효과를 함께 사용 |

이 방식은 참고 논문에서 제안한 NLOS 식별, WLS 가중치 조정, 실내측위 feature 기반 학습 아이디어를 그대로 복사한 것이 아니라, 본 데이터에 맞게 anchor error prediction과 multi-candidate WLS feature를 구성한 점이 다르다. 참고 논문들은 UWB/RTT 측정에서 NLOS와 multipath가 거리 오차를 키울 수 있고, 신뢰도 또는 오차 완화 과정이 필요하다는 근거로 활용하였다. 본 프로젝트에서는 원시 채널 impulse response가 제공되지 않기 때문에, 대신 d_hat의 통계량, 순위, anchor geometry, WLS residual을 이용하여 신뢰도 정보를 간접적으로 구성하였다.

## 3. Agent AI 활용 방안

Agent AI는 알고리즘을 자동으로 결정하는 도구가 아니라, 실험 설계와 구현 검토를 보조하는 도구로 사용하였다. 본인은 데이터셋을 직접 실행하고 결과를 확인하며, validation 결과를 기준으로 최종 모델을 선택하였다.

| 구분 | 수행 내용 |
|---|---|
| 본인 역할 | README 조건 확인, 데이터 실행, 모델 성능 확인, 최종 제출 파일 검토, report 내용 최종 선택 |
| Agent AI 활용 | 데이터 분석 항목 제안, 머신러닝 후보 모델 정리, main.py와 train.py 구조 설계 보조, report.md 초안 작성 보조 |
| 본인 검증 방식 | 실제 DH_FR1.mat로 train.py와 main.py를 실행하여 model.pkl 생성과 p_hat shape를 확인 |
| AI 사용 시 주의점 | AI가 제안한 구조를 그대로 제출하지 않고, README의 데이터 경로, 사용자 수 동적 처리, report 형식 제한을 다시 확인 |
| 최종 판단 기준 | hidden test 과적합을 줄이기 위해 단일 split 최고 성능만 보지 않고 모델 복잡도, 파일 크기, 실행 시간, 일반화 가능성을 함께 고려 |

## 4. 결과 도출 & 디스커션

실험은 제공된 700명 데이터를 train과 validation으로 나누어 진행하였다. validation set은 최종 모델 선택을 위한 내부 평가용으로만 사용하였다. hidden test set의 정답은 알 수 없으므로, main.py는 정답 p를 사용하지 않고 d_hat과 BS_positions, 그리고 학습된 model.pkl만 사용하여 p_hat을 반환하도록 구성하였다.

| 평가 설정 | 값 |
|---|---:|
| 전체 제공 사용자 수 | 700 |
| train 비율 | 0.8 |
| validation 비율 | 0.2 |
| anchor 수 | 18 |
| 최종 model.pkl 크기 | 76.68 MB |
| main.py 반환 shape | (2, num_user) |

모델 비교 결과는 다음과 같다. 모든 수치는 같은 train/validation split에서 계산하였으며, 위치 오차는 예측 좌표와 정답 좌표 사이의 Euclidean distance로 계산하였다.

| model | mode | RMSE | MAE | Median_error | m90 | Max_error |
|---|---|---:|---:|---:|---:|---:|
| ensemble_ridge_extra_gbr | ensemble | 6.3721 | 4.5244 | 3.0303 | 9.3845 | 28.7452 |
| ensemble_ridge_rf_extra_gbr | ensemble | 6.4245 | 4.5486 | 3.1422 | 9.3377 | 29.2361 |
| extra_trees_60 | single | 6.5447 | 4.8498 | 3.5102 | 9.2203 | 28.5595 |
| extra_trees_80_depth20 | single | 6.5459 | 4.8182 | 3.4838 | 9.4546 | 29.0085 |
| ensemble_rf_extra_gbr | ensemble | 6.5798 | 4.6317 | 3.2402 | 9.7126 | 28.7231 |
| gradient_boosting | single | 6.8933 | 4.6408 | 2.7566 | 11.2005 | 26.8821 |
| random_forest_120 | single | 6.9313 | 5.1057 | 3.7953 | 9.6440 | 30.7532 |
| random_forest_60 | single | 6.9409 | 5.0788 | 3.7917 | 9.9239 | 30.9800 |
| ridge_alpha50 | single | 6.9749 | 5.3630 | 4.0834 | 11.3608 | 31.0842 |
| ridge_alpha12 | single | 7.0196 | 5.3436 | 4.0921 | 11.0664 | 31.1943 |
| pls_regression | single | 7.2621 | 5.5859 | 4.4571 | 12.1507 | 31.9340 |
| knn_9_distance | single | 9.9305 | 8.3826 | 7.1581 | 15.6118 | 34.0396 |
| knn_5_distance | single | 9.9477 | 8.3418 | 7.4357 | 14.8105 | 33.7725 |

비교 결과에서 Extra Trees 계열은 RMSE가 낮고 안정적이었지만, Gradient Boosting은 Median_error와 Max_error 측면에서 장점을 보였다. Ridge는 단독 성능은 가장 좋지 않았지만, 선형적이고 과적합 위험이 낮아 ensemble에 포함했을 때 안정화 역할을 하였다. 따라서 최종 모델은 Ridge, Extra Trees, Gradient Boosting을 결합한 ensemble_ridge_extra_gbr로 선정하였다.

| 비교 관점 | 단일 Extra Trees | 단일 Gradient Boosting | 단일 Ridge | 최종 Ensemble |
|---|---:|---:|---:|---:|
| RMSE | 6.5447 | 6.8933 | 6.9749 | 6.3721 |
| MAE | 4.8498 | 4.6408 | 5.3630 | 4.5244 |
| Median_error | 3.5102 | 2.7566 | 4.0834 | 3.0303 |
| m90 | 9.2203 | 11.2005 | 11.3608 | 9.3845 |
| Max_error | 28.5595 | 26.8821 | 31.0842 | 28.7452 |

이 비교는 딥러닝처럼 복잡도가 완전히 다른 모델과 단순 baseline을 불공정하게 비교하는 방식이 아니다. 모든 모델은 동일한 feature set을 사용하고, 같은 validation split에서 평가되었다. 차이는 최종 회귀기의 종류와 ensemble 여부뿐이다. 이 때문에 모델 비교 결과는 feature 설계 효과와 회귀기 선택 효과를 비교적 공정하게 보여준다.

본 알고리즘의 장점은 측정 오차가 큰 anchor를 완전히 제거하지 않고 predicted error로 부드럽게 반영한다는 점이다. 또한 WLS 후보를 여러 개 만들어 ML이 기하학 기반 후보 정보를 함께 학습할 수 있도록 구성하였다. 단점은 feature 생성과 ensemble로 인해 단순 Ridge나 KNN보다 model.pkl 크기와 연산량이 크다는 점이다. 

| 항목 | 만족 여부 | 확인 내용 |
|---|---|---|
| main.py에 main 함수 정의 | 만족 | main()이 p_hat을 반환 |
| p_hat shape | 만족 | (2, num_user) 형태 반환 |
| 사용자 수 하드코딩 방지 | 만족 | d_hat.shape[1] 사용 |
| 데이터 파일명 | 만족 | DH_FR1.mat 사용 |
| 표준 패키지 사용 | 만족 | numpy, scipy, scikit-learn, pandas만 사용 |
| requirements.txt 필요성 | 불필요 | README 표준 패키지만 사용 |
| model 파일 형식 | 만족 | model.pkl 사용 |
| report.md 형식 제한 | 만족 | 코드 블록, 의사코드, 이미지 미사용 |
| 결과 수치 표기 | 만족 | 모든 실험 결과 수치를 markdown 표로 작성 |

향후 개선 방향은 validation 방식을 단일 holdout에서 spatial K-fold로 확장하는 것이다. 현재는 random split 기반이므로 위치 공간의 특정 영역이 validation에 많이 포함되는 경우 결과가 달라질 수 있다. 또한 hidden test set에서의 일반화를 더 높이기 위해 model ensemble의 구성 수를 늘리는 대신, anchor error prediction의 calibration을 개선하거나 residual이 큰 사용자에 대한 adaptive model selection을 추가할 수 있다.

## 5. Reference

| 번호 | Reference | 신뢰도 및 핵심 내용 | 본 알고리즘에 반영한 부분 | 본 알고리즘과의 차이 |
|---:|---|---|---|---|
| 1 | S. Maranò, W. M. Gifford, H. Wymeersch, and M. Z. Win, NLOS Identification and Mitigation for Localization Based on UWB Experimental Data, IEEE Journal on Selected Areas in Communications, 2010. | UWB 실험 데이터 기반으로 NLOS 상황에서 ranging error가 커지고 이를 식별 및 완화해야 함을 보인 대표 IEEE 논문이다. | d_hat의 큰 오차를 단순 noise가 아니라 anchor별 신뢰도 문제로 보고 predicted error model을 설계하였다. | 논문은 UWB 실험 채널 데이터와 NLOS 식별을 직접 다루지만, 본 프로젝트는 채널 원시 정보가 없으므로 d_hat 통계와 residual로 신뢰도를 간접 추정하였다. |
| 2 | İ. Güvenç, C. C. Chong, F. Watanabe, and H. Inamura, NLOS Identification and Weighted Least-Squares Localization for UWB Systems Using Multipath Channel Statistics, EURASIP Journal on Advances in Signal Processing, 2008. | NLOS 가능성이 있는 측정값에 낮은 WLS weight를 부여하는 localization 구조를 제안한 신뢰도 높은 논문이다. | predicted anchor error를 이용하여 신뢰도 높은 anchor subset WLS 후보를 만들고, error가 큰 anchor의 영향력을 줄였다. | 논문은 multipath channel statistics를 사용하지만, 본 프로젝트는 제공된 d_hat만 사용해야 하므로 rank, pairwise difference, WLS residual을 대체 feature로 사용하였다. |
| 3 | F. Zafari, A. Gkelias, and K. K. Leung, A Survey of Indoor Localization Systems and Technologies, IEEE Communications Surveys & Tutorials, 2019. | ToF, RTT, RSS, fingerprinting 등 실내측위 기술을 종합적으로 정리한 IEEE survey이다. | RTT 기반 거리 측위와 fingerprinting 기반 feature regression을 결합하는 hybrid ML 구조의 근거로 활용하였다. | Survey는 다양한 기술을 개괄하지만, 본 프로젝트는 18개 RTT 측정값만 주어진 제한된 문제에 맞게 feature engineering과 compact ensemble을 구현하였다. |
| 4 | F. Wang et al., Survey on NLOS Identification and Error Mitigation for UWB Indoor Positioning, Electronics, 2023. | UWB 실내측위에서 NLOS 식별, error mitigation, ML 기반 보정 방법을 정리한 최신 survey이다. | anchor error prediction, residual feature, outlier-aware ensemble의 필요성을 설명하는 이론적 근거로 사용하였다. | Survey는 여러 센서와 채널 정보를 다루지만, 본 프로젝트는 BS_positions와 d_hat만 사용하여 README 조건에 맞는 단일 입력 구조로 제한하였다. |
| 5 | C. L. Sang et al., Identification of NLOS and Multi-Path Conditions in UWB Localization Using Machine Learning Methods, Applied Sciences, 2020. | UWB 환경에서 LOS, NLOS, multipath 조건을 머신러닝으로 구분하는 실험 연구이다. | anchor별 오차를 하나의 고정 bias로 보지 않고 사용자별 feature에 따라 달라지는 ML prediction 문제로 모델링하였다. | 논문은 조건 분류 문제를 다루지만, 본 프로젝트는 class label이 없으므로 predicted distance error를 회귀 문제로 변환하였다. |
