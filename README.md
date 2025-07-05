# 🏥 데이콘 갑상선암 진단 분류 해커톤

![image](https://github.com/user-attachments/assets/b5ac8100-c411-40de-abd1-e84f02be6a5e)
`public : 110위`

![image](https://github.com/user-attachments/assets/6af6718f-abca-41ca-9037-6c45ebf78a56)
`private : 58위` (13위~90위 동일한 점수)

## 📖 대회 개요
- 공식사이트: [데이콘](https://dacon.io/competitions/official/236488/overview/description)
- 목적: 갑상선과 관련된 건강 데이터를 기반으로 양성인지 악성인지 예측하는 이진 분류 모델
- 평가 지표: F1 Score (주 평가 기준)
- 대회 기간: 2025.05.07 ~ 2025.06.30
- 참여 기간: 2025.05.20 ~ 2025.05.26


## 📂 데이 구성
`train.csv`
| #   | Column              | Non-Null Count   | Dtype    |
|-----|---------------------|------------------|----------|
| 0   | ID                  | 87159 non-null   | object   |
| 1   | Age                 | 87159 non-null   | int64    |
| 2   | Gender              | 87159 non-null   | object   |
| 3   | Country             | 87159 non-null   | object   |
| 4   | Race                | 87159 non-null   | object   |
| 5   | Family_Background   | 87159 non-null   | object   |
| 6   | Tadiation_History   | 87159 non-null   | object   |
| 7   | Iodine_Deficiency   | 87159 non-null   | object   |
| 8   | Smoke               | 87159 non-null   | object   |
| 9   | Weight_Risk         | 87159 non-null   | object   |
| 10  | Diabetes            | 87159 non-null   | object   |
| 11  | Nodule_Size         | 87159 non-null   | float64  |
| 12  | TSH_Result          | 87159 non-null   | float64  |
| 13  | T4_Result           | 87159 non-null   | float64  |
| 14  | T3_result           | 87159 non-null   | float64  |
| 15  | Cancer              | 87159 non-null   | int64    |



## 🛠 주요 처리 과정
### 📊 데이터 전처리
- 범주형 변수 인코딩 (map, get_dummies)
- 수치형 변수 시각화 및 분포 분석 (skewness 확인)
- 상관관계 시각화 (seaborn.heatmap)

### 🤖 모델 비교
- 랜덤 포레스트(RandomForest)
- 로지스틱 회귀(Logistic Regression)
- LightGBM
- XGBoost
- PyTorch 기반 MLP (딥러닝)




🧠 PyTorch 모델 구조
``` python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
```
임계값(threshold) 0.3으로 조정하여 F1 score 개선 시도

🔍 주요 성능 비교 
| 모델   | F1 Score |
|-----|-------------|
| RandomForest   | 0.2986 |
| LightGBM   | 0.3451 |
| XGBoost  | 0.3153 |
| LogisticRegression   | 0.0066 |
| PyTorch MLP   | 0.4886 |


✅ PyTorch MLP가 가장 높은 F1 Score를 가짐

## 📤 결과 
- Public Score: 0.51094
- Private Score: 0.50974
