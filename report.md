# Reliability-aware Multi-Candidate Ensemble Localization Report

| 항목 | 내용 |
|---|---|
| 이름 | 양준범 |
| 학과 | 스마트모빌리티공학과 |
| 학번 | 12223639 |
| 프로젝트 | 스마트모빌리티공학실험2 Final Project |
| 최종 알고리즘명 | Reliability-aware Multi-Candidate Ensemble Localization |

## 1. 모티베이션 & 인트로

본 프로젝트의 목표는 18개 기지국에서 측정된 RTT 기반 거리값을 이용하여 사용자의 2차원 위치를 추정하는 것이다. 입력 데이터는 사용자별 거리 측정값 d_hat과 기지국 좌표 BS_positions로 구성되어 있으며, 학습 과정에서는 정답 좌표 p를 이용해 모델을 만들 수 있다. 그러나 실제 추정 단계에서는 정답 좌표를 사용할 수 없기 때문에, 단순히 학습 데이터의 좌표를 외우는 방식보다 거리 측정값의 구조와 오차 특성을 함께 반영하는 방식이 더 적합하다고 판단하였다.

중간 발표 단계에서는 Random Forest로 관측값의 신뢰도를 예측하고, 그 신뢰도를 WLS에 반영하는 방향을 중심으로 실험을 진행하였다. 당시에는 평균, 중앙값, 분산과 같은 통계량을 이용해 각 관측값이 얼마나 안정적인지를 판단하는 구조를 생각하였다. 이후 최종 데이터 구조를 기준으로 다시 정리하면서, 개별 관측의 신뢰도를 직접적인 reliability label로 두기보다는 각 anchor의 거리 오차를 예측하고, 그 예측값을 위치 후보 생성과 최종 회귀 모델의 feature로 사용하는 방식으로 확장하였다.

사전 분석에서 가장 중요하게 확인한 점은 d_hat이 실제 기하학적 거리와 항상 일치하지 않는다는 점이었다. 어떤 anchor에서는 실제 거리보다 크게 측정되는 양의 bias가 반복적으로 나타났고, 일부 사용자와 anchor 조합에서는 다른 측정값에 비해 큰 outlier가 포함되어 있었다. 이 상황에서 모든 anchor를 같은 정도로 신뢰하면, 오차가 큰 anchor 하나가 전체 위치 추정 결과를 크게 흔들 수 있다. 그래서 본 알고리즘은 모든 측정값을 무조건 동일하게 사용하는 대신, anchor별 예상 오차를 먼저 추정하고 여러 개의 WLS 기반 위치 후보를 만든 뒤, 마지막으로 서로 다른 성격의 회귀 모델을 결합하여 최종 좌표를 예측하도록 설계하였다.

| 관찰 내용 | 의미 | 최종 설계에 반영한 방식 |
|---|---|---|
| anchor별 오차 크기가 일정하지 않음 | 같은 RTT 값이라도 anchor 위치와 사용자 위치에 따라 신뢰도가 달라질 수 있음 | anchor error prediction model을 두어 각 anchor의 예상 거리 오차를 추정 |
| 일부 측정값에서 큰 outlier가 발생 | 전체 anchor를 한 번에 사용하는 WLS가 특정 anchor에 끌릴 수 있음 | predicted error top-k, distance top-k 기반의 여러 WLS 후보 생성 |
| 거리의 절대값만으로는 위치 관계를 충분히 설명하기 어려움 | 가까운 anchor의 순서나 anchor 간 거리 차이도 위치 정보를 포함함 | rank feature, pairwise distance difference feature 추가 |
| 단일 모델마다 장단점이 다름 | 선형 모델은 안정적이고, tree 모델은 비선형 패턴을 잘 잡음 | Ridge, Extra Trees, Gradient Boosting을 결합한 ensemble 사용 |
| validation 결과가 모델마다 다른 양상을 보임 | 평균 오차, 중앙값 오차, 큰 오차를 동시에 고려해야 함 | RMSE뿐 아니라 MAE, Median error, m90, Max error를 함께 확인 |

최종 알고리즘은 단순한 삼각측량이나 단일 머신러닝 회귀 모델이 아니라, 기하학 기반 위치 후보와 데이터 기반 회귀 모델을 함께 사용하는 하이브리드 구조이다. WLS는 RTT 측정값과 anchor 좌표 사이의 물리적 관계를 반영할 수 있고, ensemble 회귀 모델은 실제 데이터에 포함된 bias와 비선형 오차 패턴을 보정할 수 있다. 따라서 본 방법의 핵심은 “anchor 측정값의 신뢰도 차이를 먼저 반영하고, 여러 후보 위치를 비교할 수 있는 feature를 만든 뒤, 최종 좌표를 안정적으로 선택하는 것”이다.

## 2. 알고리즘 설명

최종 알고리즘은 크게 anchor error prediction, feature construction, multi-candidate WLS, final ensemble의 네 단계로 구성된다. 전체 흐름은 먼저 각 사용자에 대해 18개 anchor 거리값을 정리하고, anchor별 예상 오차를 계산한 뒤, 이 정보를 이용해 여러 위치 후보를 만든다. 이후 원본 거리값, 통계량, 거리 순위, anchor 간 거리 차이, WLS 후보 좌표와 residual 통계량을 하나의 feature vector로 합쳐 최종 회귀 모델에 입력한다.

| 단계 | 역할 | 입력 | 출력 |
|---|---|---|---|
| Anchor error prediction | anchor별 거리 측정 오차를 예측 | d_hat, anchor 좌표, 사용자별 거리 통계, rank | predicted anchor error |
| Basic feature construction | 원본 거리값에서 위치 추정에 필요한 파생 feature 생성 | d_hat, BS_positions | 거리 기반 feature vector |
| Multi-candidate WLS | 여러 anchor 조합으로 기하학적 위치 후보 생성 | d_hat, predicted error, BS_positions | WLS 후보 좌표와 residual 통계 |
| Final ensemble regression | 여러 회귀 모델의 예측을 가중 평균하여 최종 좌표 계산 | 전체 feature vector | 예측 위치 p_hat |

Anchor error prediction은 최종 좌표를 바로 예측하는 모델이 아니라, 각 anchor 측정값이 실제 거리에서 얼마나 벗어날 가능성이 있는지를 예측하는 보조 모델이다. 학습 데이터에서는 정답 위치 p와 anchor 좌표 a_i를 알고 있으므로, i번째 anchor의 실제 거리는 다음과 같이 계산할 수 있다.

| 기호 | 의미 |
|---|---|
| p | 사용자 실제 위치 |
| a_i | i번째 anchor의 좌표 |
| d_i | i번째 anchor의 측정 거리 |
| d_true,i | 사용자와 i번째 anchor 사이의 실제 거리 |
| e_i | i번째 anchor의 거리 측정 오차 |

| 수식 | 설명 |
|---|---|
| d_true,i = ||p - a_i||₂ | 정답 좌표와 anchor 좌표 사이의 실제 유클리디안 거리 |
| e_i = |d_i - d_true,i| | 측정 거리와 실제 거리의 차이 |

이때 e_i가 작으면 해당 anchor의 측정값이 실제 거리와 비교적 일치한다고 볼 수 있고, e_i가 크면 NLOS, multipath, 특정 위치에서의 bias 등으로 인해 신뢰도가 낮을 가능성이 크다. 본 알고리즘에서는 e_i를 직접 feature로 사용할 수 없기 때문에, 학습 단계에서 Extra Trees 기반 회귀 모델을 이용해 e_i를 예측하도록 하였다. 추정 단계에서는 정답 위치 p가 없으므로, d_hat의 크기, 사용자별 통계량, anchor별 rank, anchor 좌표, 다른 anchor와의 거리 차이 등을 이용해 predicted error를 계산한다.

예측된 error는 두 가지 방식으로 사용된다. 첫째, WLS 후보를 만들 때 predicted error가 작은 anchor를 더 신뢰할 수 있는 anchor로 보고 우선적으로 사용한다. 둘째, predicted error 자체와 그 역수 형태의 값을 최종 회귀 모델의 feature에 포함하여, 모델이 “현재 사용자에서 어떤 anchor가 상대적으로 불안정한지”를 학습할 수 있도록 한다.

다음 단계는 사용자별 feature를 구성하는 과정이다. 원본 d_hat 18개만 사용하면 anchor별 거리 패턴은 반영할 수 있지만, 측정값의 분포나 상대적인 위치 관계를 충분히 표현하기 어렵다. 따라서 본 알고리즘은 다음과 같은 feature들을 함께 사용하였다.

| Feature 그룹 | 포함 내용 | 사용 의도 |
|---|---|---|
| Raw distance | 18개 anchor의 d_hat | 가장 기본적인 RTT 거리 패턴 반영 |
| Nonlinear distance transform | log(1+d), sqrt(d), d²/100 | 거리 scale 변화에 따른 비선형 관계 보완 |
| User-level statistics | 평균, 중앙값, 표준편차, 최솟값, 최댓값, range, 분위수 | 사용자별 측정값 분포와 outlier 정도 표현 |
| Rank feature | 18개 anchor 거리의 상대 순위 | 절대 거리값이 흔들릴 때도 가까운 anchor 순서 보존 |
| Top-k anchor mask | 가까운 anchor 1, 2, 3, 4, 5, 6, 8개 표시 | 어느 anchor 그룹이 사용자와 가까운지 표현 |
| Inverse-distance center | 거리 역수 가중치를 이용한 anchor 중심 좌표 | 대략적인 위치 힌트 제공 |
| Local anchor geometry | 가까운 k개 anchor의 좌표 평균과 분산 | 주변 anchor 배치 구조 반영 |
| Pairwise difference | 모든 anchor 쌍의 d_i - d_j | x, y 방향성에 대한 상대 거리 정보 제공 |
| Predicted anchor error | anchor별 예상 오차와 역수형 신뢰도 | anchor 신뢰도 차이 반영 |
| WLS candidate | 여러 anchor subset으로 계산한 후보 좌표 | 기하학 기반 초기 위치 후보 제공 |
| Residual statistics | 후보 위치에서의 residual 평균, 중앙값, 표준편차, 최댓값 등 | 후보 좌표가 입력 거리와 얼마나 일관적인지 표현 |

Multi-candidate WLS는 본 알고리즘에서 중요한 부분이다. 일반적인 WLS는 모든 anchor를 사용하여 하나의 좌표를 구하지만, 측정값 중 일부에 큰 outlier가 섞이면 그 하나의 결과가 불안정해질 수 있다. 그래서 본 실험에서는 WLS 결과를 하나로 고정하지 않고, 서로 다른 anchor subset을 이용해 여러 후보 좌표를 생성하였다.

| WLS 후보 유형 | anchor 선택 기준 | 목적 |
|---|---|---|
| All-anchor reliability WLS | 18개 anchor 전체 사용, predicted error 기반 weight 적용 | 전체 측정값을 반영한 기본 후보 생성 |
| Error top-5 WLS | predicted error가 작은 5개 anchor 사용 | 가장 신뢰도 높은 소수 anchor 기반 후보 |
| Error top-7 WLS | predicted error가 작은 7개 anchor 사용 | 신뢰도와 anchor 수 사이의 균형 후보 |
| Error top-9 WLS | predicted error가 작은 9개 anchor 사용 | 중간 규모의 안정적인 후보 |
| Error top-12 WLS | predicted error가 작은 12개 anchor 사용 | outlier 일부를 줄이면서 충분한 anchor 확보 |
| Distance top-5 WLS | 측정 거리 d_hat이 작은 5개 anchor 사용 | 가까운 anchor 중심의 지역 후보 |
| Distance top-7 WLS | 측정 거리 d_hat이 작은 7개 anchor 사용 | 가까운 anchor를 조금 더 넓게 반영 |
| Distance top-9 WLS | 측정 거리 d_hat이 작은 9개 anchor 사용 | 지역성과 안정성의 균형 후보 |
| Distance top-12 WLS | 측정 거리 d_hat이 작은 12개 anchor 사용 | 가까운 anchor 중심으로 넓은 후보 생성 |

각 WLS 후보는 anchor 좌표와 측정 거리의 제곱 차이를 선형화하여 계산하였다. 기준 anchor를 하나 잡고, 다른 anchor와의 거리 제곱식 차이를 이용하면 위치 p에 대한 선형 방정식 형태를 만들 수 있다. predicted error를 사용하는 후보에서는 error가 작은 anchor일수록 더 큰 weight를 갖도록 하여 오차가 클 것으로 예상되는 측정값의 영향을 줄였다. 후보 좌표를 만든 뒤에는 그 좌표에서 각 anchor까지의 기하학적 거리와 실제 d_hat 사이의 차이를 residual로 계산하였다.

| Residual 관련 값 | 의미 |
|---|---|
| residual_i = ||p_candidate - a_i||₂ - d_i | 후보 좌표가 i번째 anchor 측정값과 얼마나 맞는지 |
| mean absolute residual | 전체 anchor에 대한 평균 불일치 정도 |
| median absolute residual | 이상치 영향을 줄여 본 일반적인 불일치 정도 |
| max absolute residual | 가장 크게 어긋난 anchor의 오차 |
| 75th percentile residual | residual 분포의 상위 구간 크기 |

최종 회귀 모델은 단일 모델 하나만 사용하지 않았다. Ridge Regression, Extra Trees, Gradient Boosting은 각각 성격이 다르기 때문에 같은 feature를 보더라도 서로 다른 방식으로 좌표를 예측한다. Ridge는 선형 모델이라 복잡한 비선형 구조를 모두 잡지는 못하지만, 과하게 흔들리지 않는 안정적인 예측을 제공한다. Extra Trees는 무작위성이 큰 tree ensemble이기 때문에 복잡한 거리 패턴과 anchor 배치에 따른 비선형 관계를 잘 반영한다. Gradient Boosting은 이전 모델이 놓친 오차를 순차적으로 줄여 가는 방식이라 median error를 낮추는 데 도움이 되었다.

| 최종 ensemble 구성 | 역할 |
|---|---|
| Ridge alpha 50 | 전체 예측을 안정화하고 과도한 tree 기반 변동을 완화 |
| Extra Trees 60 | 비선형 위치 패턴과 anchor 조합의 영향을 반영 |
| Gradient Boosting 60 | 작은 오차 구간의 정밀도를 보완 |

세 모델의 결과는 단순 평균이 아니라 validation RMSE의 역수에 비례하는 가중 평균으로 결합하였다. 최종 저장된 ensemble weight는 Ridge 0.3288, Extra Trees 0.3403, Gradient Boosting 0.3309로 거의 균형 있게 분포하였다. 이는 특정 모델 하나가 압도적으로 지배하는 구조라기보다, 세 모델이 서로 비슷한 정도로 최종 좌표 추정에 기여했다는 의미로 해석할 수 있다.

참고한 논문들은 UWB/RTT 기반 실내 측위에서 NLOS와 multipath가 거리 측정 오차를 키우고, 이러한 오차를 식별하거나 가중치를 조정하는 과정이 필요하다는 점을 보여준다. 다만 본 프로젝트에서는 원시 채널 정보나 LOS/NLOS 라벨이 제공되지 않았기 때문에, 논문에서 사용하는 채널 통계량을 그대로 적용하지 않았다. 대신 d_hat의 상대 순위, anchor 간 거리 차이, WLS residual, predicted anchor error를 이용해 신뢰도 정보를 간접적으로 구성하였다. 이 점이 기존 NLOS 식별 기반 WLS 논문과 본 알고리즘의 가장 큰 차이이다.

## 3. Agent AI 활용 방안

본 실험에서 Agent AI는 알고리즘을 대신 정해 주는 도구로 사용하지 않고, 아이디어를 비교하고 구현 과정에서 놓칠 수 있는 부분을 점검하는 보조 도구로 사용하였다. 특히 중간 발표 이후 RF 기반 신뢰도 WLS 구조를 최종 데이터셋에 맞게 어떻게 바꿀지 정리하는 과정에서 여러 후보를 비교하는 데 활용하였다.

| 구분 | 수행 내용 |
|---|---|
| 본인이 직접 수행한 부분 | 데이터 구조 확인, 거리 오차 분석, anchor error prediction 구조 선택, WLS 후보 종류 결정, 모델 비교 실행, 최종 ensemble 선택 |
| Agent AI를 활용한 부분 | 실내 측위에서 사용할 수 있는 feature 후보 정리, WLS와 머신러닝 회귀를 결합하는 방식 비교, 보고서 문장 흐름 정리, 코드 구조에서 오류가 생길 수 있는 부분 검토 |
| 최종 판단 방식 | validation 결과와 모델 특성을 같이 보고 최종 모델을 선택하였다. RMSE가 낮은지뿐 아니라 median error, m90, max error, 모델 크기, 실행 구조도 함께 확인하였다. |
| 반영하지 않은 부분 | 단순 Random Forest 하나만 사용하는 방식, 데이터에만 맞춘 복잡한 암기형 회귀 방식, 실제 코드와 맞지 않는 WiFi/UWB 모달리티 보정 설명은 최종 구조에서 제외하였다. |

AI가 제안한 내용 중 일부는 그대로 사용하지 않았다. 예를 들어, 처음에는 Random Forest가 직접 reliability를 예측하고 그 값을 WLS weight로 사용하는 구조가 가장 단순해 보였지만, 최종 데이터에서는 18개 RTT 값과 anchor 좌표만으로 좌표를 추정해야 했기 때문에 anchor error를 예측하고 그 결과를 feature와 WLS 후보 생성에 함께 사용하는 방식이 더 자연스럽다고 판단하였다. 또한 단일 tree 모델만 사용하면 validation split에서는 좋은 결과가 나와도 위치 공간의 다른 구간에서 불안정할 수 있으므로, Ridge를 ensemble에 포함해 예측의 변동성을 줄였다.

최종 정리에서는 실제 구현 흐름과 맞지 않는 내용은 제외하고, 왜 해당 feature와 모델 구조를 선택했는지가 자연스럽게 드러나도록 문장을 다시 다듬었다.

## 4. 결과 도출 & 디스커션

실험은 제공된 700명 데이터를 8:2로 나누어 train 560명, validation 140명 기준으로 진행하였다. 평가지표는 예측 좌표와 정답 좌표 사이의 유클리디안 거리 오차를 기준으로 계산하였다. RMSE는 큰 오차에 민감하므로 전체 안정성을 보는 데 사용하였고, MAE와 Median error는 평균적인 위치 추정 성능을 보기 위해 함께 확인하였다. m90은 전체 사용자 중 상위 10% 수준의 큰 오차가 어느 정도인지 확인하기 위한 지표로 사용하였다.

| 평가 설정 | 값 |
|---|---:|
| 전체 데이터 수 | 700 |
| train 데이터 수 | 560 |
| validation 데이터 수 | 140 |
| anchor 수 | 18 |
| 최종 선택 모델 | ensemble_ridge_extra_gbr |
| 최종 model.pkl 크기 | 76.68 MB |

모델 비교는 같은 feature set과 같은 train/validation split을 기준으로 수행하였다. 즉, 어떤 모델은 더 많은 정보를 사용하고 다른 모델은 적은 정보를 사용하는 방식이 아니라, 동일한 입력 feature를 두고 회귀기 구조만 다르게 비교하였다. 따라서 아래 결과는 feature 설계의 차이라기보다 최종 회귀 모델 선택에 따른 차이를 보여준다.

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

가장 좋은 RMSE를 보인 모델은 Ridge, Extra Trees, Gradient Boosting을 결합한 ensemble_ridge_extra_gbr이었다. 단일 Extra Trees는 RMSE와 m90이 안정적이었고, Gradient Boosting은 Median error와 Max error에서 장점이 있었다. Ridge는 단독 성능만 보면 가장 좋은 모델은 아니지만, tree 기반 모델이 특정 영역에서 과하게 흔들리는 것을 줄여 주는 역할을 하였다. 이 때문에 최종 ensemble은 단일 모델보다 RMSE와 MAE가 낮게 나타났다.

| 비교 항목 | Ridge alpha50 | Extra Trees 60 | Gradient Boosting | 최종 Ensemble |
|---|---:|---:|---:|---:|
| RMSE | 6.9749 | 6.5447 | 6.8933 | 6.3721 |
| MAE | 5.3630 | 4.8498 | 4.6408 | 4.5244 |
| Median_error | 4.0834 | 3.5102 | 2.7566 | 3.0303 |
| m90 | 11.3608 | 9.2203 | 11.2005 | 9.3845 |
| Max_error | 31.0842 | 28.5595 | 26.8821 | 28.7452 |

이 결과를 보면 Gradient Boosting은 중간값 오차가 가장 낮지만, m90은 상대적으로 크다. 즉 대부분의 샘플에서는 정밀하게 맞추지만, 일부 샘플에서 오차가 커질 수 있다는 의미이다. Extra Trees는 m90이 낮아 전체적으로 안정적이지만, Median error는 Gradient Boosting보다 높다. 최종 ensemble은 두 모델의 장점을 어느 정도 섞으면서 Ridge의 안정성을 추가한 형태이다. 최종 모델의 Median error가 Gradient Boosting 단일 모델보다 약간 커진 점은 아쉬운 부분이지만, RMSE와 MAE를 함께 고려했을 때 전체적인 균형은 ensemble이 더 좋았다.

KNN 계열 모델은 같은 feature를 사용했음에도 RMSE가 9.9m 수준으로 높게 나타났다. 이는 단순히 가까운 학습 샘플을 찾는 방식만으로는 새로운 위치나 측정 오차 패턴을 충분히 일반화하기 어렵다는 것을 보여준다. 반대로 Ridge, Extra Trees, Gradient Boosting 계열은 모두 KNN보다 낮은 오차를 보였고, 특히 WLS 후보와 residual feature를 함께 사용했을 때 좌표 회귀 모델이 기하학적 정보를 더 잘 활용할 수 있었다.

본 알고리즘의 장점은 크게 세 가지이다. 첫째, 측정값이 이상하다고 판단되는 anchor를 완전히 버리지 않고 predicted error를 통해 부드럽게 반영한다. 완전 제거 방식은 특정 상황에서 필요한 anchor까지 버릴 수 있지만, 본 방식은 낮은 신뢰도를 feature와 weight에 반영하기 때문에 더 유연하다. 둘째, 하나의 WLS 결과에 의존하지 않고 여러 anchor subset에서 후보 좌표를 만들기 때문에 outlier에 의한 쏠림을 줄일 수 있다. 셋째, 선형 모델과 비선형 tree 모델을 함께 사용하여 안정성과 표현력을 동시에 확보하려고 하였다.

반대로 한계도 있다. feature 수가 많고 anchor error model과 여러 WLS 후보를 모두 계산하기 때문에 단순 선형 회귀나 단일 KNN보다 구조가 무겁다. 또한 validation은 random split 기준으로 진행했기 때문에, 위치 공간 전체에 대해 고르게 일반화되는지를 완전히 확인했다고 보기는 어렵다. 만약 추가 실험을 진행한다면, 좌표 공간을 구역별로 나누는 spatial K-fold 방식으로 검증하여 특정 공간 영역에 대한 일반화 성능을 더 엄격하게 확인할 필요가 있다. 또한 residual이 매우 작은 사용자에 대해서는 복잡한 ensemble까지 가지 않고 더 단순한 모델로 조기 종료하는 adaptive 구조를 적용하면 실행 시간을 줄일 수 있을 것이다.

최종적으로 본 실험에서는 단순한 거리 기반 측위보다, anchor별 오차 예측과 WLS 후보 생성, ensemble 회귀를 결합한 구조가 더 안정적이라고 판단하였다. 특히 실내 RTT 측정값처럼 일부 anchor가 크게 흔들리는 데이터에서는 모든 anchor를 동일하게 믿는 방식보다, 측정값의 신뢰도와 기하학적 일관성을 함께 보는 방식이 더 적합했다.

## 5. Reference

| 번호 | Reference | 본 알고리즘과 관련된 내용 | 본 알고리즘에서 다르게 적용한 점 |
|---:|---|---|---|
| 1 | S. Maranò, W. M. Gifford, H. Wymeersch, and M. Z. Win, “NLOS Identification and Mitigation for Localization Based on UWB Experimental Data,” IEEE Journal on Selected Areas in Communications, 2010. | UWB 실험 데이터에서 NLOS 환경이 거리 측정 오차를 키우며, 이를 식별하고 완화하는 과정이 필요함을 보였다. | 본 프로젝트에서는 채널 원시 정보가 없기 때문에 NLOS를 직접 분류하지 않고, d_hat과 anchor geometry에서 파생한 feature로 anchor별 거리 오차를 예측하였다. |
| 2 | İ. Güvenç, C. C. Chong, F. Watanabe, and H. Inamura, “NLOS Identification and Weighted Least-Squares Localization for UWB Systems Using Multipath Channel Statistics,” EURASIP Journal on Advances in Signal Processing, 2008. | NLOS 가능성이 있는 측정값에 낮은 WLS weight를 부여하여 위치 추정의 오차를 줄이는 구조를 제안하였다. | 본 알고리즘은 multipath channel statistics 대신 predicted anchor error를 사용하고, 단일 WLS가 아니라 여러 anchor subset WLS 후보를 feature로 활용하였다. |
| 3 | F. Zafari, A. Gkelias, and K. K. Leung, “A Survey of Indoor Localization Systems and Technologies,” IEEE Communications Surveys & Tutorials, 2019. | RTT, ToF, RSS, fingerprinting 등 실내 측위 기술의 특징과 머신러닝 기반 측위 접근을 정리하였다. | 본 프로젝트에서는 fingerprinting식 회귀와 기하학식 WLS를 분리하지 않고, WLS 후보 좌표와 residual을 머신러닝 feature로 합치는 방식으로 사용하였다. |
| 4 | F. Wang et al., “Survey on NLOS Identification and Error Mitigation for UWB Indoor Positioning,” Electronics, 2023. | UWB 실내 측위에서 NLOS 식별, error mitigation, ML 기반 보정 방법의 흐름을 정리하였다. | 본 알고리즘은 딥러닝 기반 end-to-end 구조 대신, anchor error prediction과 WLS 후보 생성 과정을 명시적으로 두어 결과 해석이 가능하도록 구성하였다. |
| 5 | C. L. Sang et al., “Identification of NLOS and Multi-Path Conditions in UWB Localization Using Machine Learning Methods,” Applied Sciences, 2020. | 머신러닝을 이용해 LOS, NLOS, multipath 조건을 분류하는 접근을 다루었다. | 본 데이터에는 LOS/NLOS label이 없으므로 분류 문제가 아니라 거리 오차 크기를 예측하는 회귀 문제로 바꾸어 적용하였다. |
