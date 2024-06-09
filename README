import os

import random

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC

import category_encoders as ce

# 동일한 결과 보장을 위해 Seed값을 고정합니다
def seed_everything(seed):

    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)

seed_everything(42) # Seed를 42로 고정

# 데이터 일부만 불러오기 (예: 10%)

data_percent = 0.1

# 전체 데이터를 불러오는 대신 일부만 불러옵니다.

train = pd.read_csv('train.csv', nrows=int(data_percent * 28600000))

test = pd.read_csv('test.csv')

# 데이터 전처리

train_x = train.drop(columns=['ID', 'Click'])

train_y = train['Click']

test_x = test.drop(columns=['ID'])

for col in tqdm(train_x.columns):

    if train_x[col].isnull().sum() != 0:
    
        if train_x[col].dtype == 'float64' or train_x[col].dtype == 'int64':  # 숫자 타입인 경우에만 중앙값 계산
        
            median_value = train_x[col].median()
            
            train_x[col].fillna(median_value, inplace=True)
            
            test_x[col].fillna(median_value, inplace=True)

numeric_columns = train_x.select_dtypes(include=['float64', 'int64']).columns

train_x_numeric = train_x[numeric_columns]

test_x_numeric = test_x[numeric_columns]

# 스케일링

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x_numeric)

test_scaled = scaler.transform(test_x_numeric)

# Count Encoding

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

enc = ce.CountEncoder(cols=encoding_target).fit(train_x, train_y)

X_train_encoded = enc.transform(train_x)

X_test_encoded = enc.transform(test_x)

# 각 모델의 하이퍼파라미터 그리드 설정

ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

svm_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

et_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

# 각 모델에 대한 그리드 서치 정의

ada_grid = GridSearchCV(AdaBoostClassifier(), ada_params, cv=3)

rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)

gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_params, cv=3)

xgb_grid = GridSearchCV(XGBClassifier(), xgb_params, cv=3)

svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=3)

et_grid = GridSearchCV(ExtraTreesClassifier(), et_params, cv=3)

# Voting Classifier 정의

voting_clf_soft = VotingClassifier(estimators=[('ada', ada_grid), ('rf', rf_grid), ('gb', gb_grid), ('xgb', xgb_grid), ('svm', svm_grid), ('et', et_grid)], voting='soft')

# 모델 학습

voting_clf_soft.fit(X_train_encoded, train_y)

# 예측

pred_soft = voting_clf_soft.predict_proba(X_test_encoded)

# 제출 파일 생성

sample_submission = pd.read_csv('sample_submission.csv')

sample_submission['Click'] = pred_soft[:, 1]

sample_submission.to_csv('sample_submission_with_sampling_and_gridsearch.csv', index=False)
