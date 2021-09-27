
# In[0]: IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # thư viện tạo ra hình ảnh trực quan (phần mở rộng của Matphotlib)


# In[1]: LOOK AT THE BIG PICTURE (DONE)
# In[2]: GET THE DATA (DONE). LOAD DATA 
#present taken from:- https://www.kaggle.com/unanimad/us-election-2020?select=president_county_candidate.csv
#past taken from:- https://electionlab.mit.edu/data
raw_past = pd.read_csv(r'D:\DaiHoc\Nam3Ky1\Học Máy\Final\1976-2016-president.csv')
raw_present = pd.read_csv(r'D:\DaiHoc\Nam3Ky1\Học Máy\Final\president_county_candidate.csv')

# In[3] DISCOVER THE DATA TO GAIN INSIGHTS
raw_past
raw_present
print('n____________________________________ Dataset info ____________________________________')
print(raw_past.info()) 
print(raw_present.info()) 
print('n____________________________________ Some first data examples ____________________________________')
print(raw_past.head(10)) 
print(raw_present.head(10)) 
print('n____________________________________ Counts on a feature ____________________________________')
print(raw_past['state'].value_counts()) 
print(raw_present['state'].value_counts())

raw_present['year'] = 2020 #Tạo thêm cột năm 2020 vào dữ liệu hiện tại để có thứ so sánh
raw_present = raw_present.rename({'total_votes': 'candidatevotes'}, axis=1) # đổi cột tên total_vote sang candicatevote vì dữ liệu của hiện tại là số lượt bình chọn của ứng viên
raw_present

# In[04]: PREPARE THE DATA 
#4.1 Split training-test set and NEVER touch test set until test phase
#Tạo tập train từ tập dữ liệu quá khư 1976-2016 và chỉ lấy lại các features ‘state’, ‘candidate’, ‘party’, ‘year’ and ‘candidatevotes’ 
train_set = raw_past[['state', 'candidate', 'party', 'year','candidatevotes']]
#Tạo bảng Test từ bảng năm 2020
#Gộp những feature state, 'state','candidate','party', 'year' đem tổng hợp lại thành bảng candidatevotes
# Vì ta cần gọn số dữ liệu cần thiết, ví dụ như để có được số phiếu của ông Trump năm 2020 của bang Alabama
test_set = raw_present.groupby(['state','candidate','party', 'year'])['candidatevotes'].sum().reset_index()
#Sau khi có hai tập train và test ta cần kiểm tra  liệu còn cột nào null hay không
train_set.isnull().sum()
test_set.isnull().sum()
a=1
# 4.2 Remove unused features
# Đây là năm 2020, dù Mỹ là nước đa đảng nhưng con đường đua thật chất là của 2 Đảng là 
# Đảng dân chủ (DEM = democrat) và Đảng cộng hoà (REP = republican) nên ta chỉ tập trung vao
# 2 đảng có lượng phiếu chiếm là đáng kể nhât nên ta chỉ định đảng dân chủ =1, đảng cộng hoà =2
# và mọi đảng khác là 3
party1 = {'democrat':1, 'DEM':1, 'republican':2, 'REP':2}
train_set.party = train_set.party.map(party1)
test_set.party = test_set.party.map(party1)
train_set['party'] = train_set['party'].replace(np.nan, 3)
test_set['party'] = test_set['party'].replace(np.nan, 3)
# Ứng viên nào trả ra giá trị NaN thì cho luôn phiếu trống
train_set['candidate'] = train_set['candidate'].replace(np.nan, 'Blank Vote')
test_set['candidate'] = test_set['candidate'].replace(np.nan, 'Blank Vote')
train_set
test_set
# Ta thử kiểm tra biểu đồ của party trong 2 train và test
train_set.groupby('party')['candidatevotes'].count().plot.bar(ylim=0)
plt.show()
test_set.groupby('party')['candidatevotes'].count().plot.bar(ylim=0)
plt.show()
a=1
# 4.3 Separate labels from data, since we do not process label values
#candidate_votes là giá trị mà ta hướng đến và sử dụng sau này là label
#Xác định các biến X, y, X_test được lấy từ train và test
candidate_votes = test_set.candidatevotes
y = train_set.candidatevotes
train_set_2= train_set.drop(['candidatevotes'], axis=1)
test_set_2 = test_set.drop(['candidatevotes'], axis=1)
 
train_set_2.info()
test_set_2.info()
#Tách Train_set cho 2 phần training và validation
from sklearn.model_selection import train_test_split #hàm hỗ trợ tách
X_train, X_val, y_train, y_val = train_test_split(train_set_2, y, random_state=1, test_size=0.10)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, test_set_2.shape

# 4.4 Define pipelines for processing data. 
# xác định các numeric và transformers để xử lý kiểu dữ liệu
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

numeric_features = [2, 3]
numeric_transformer = Pipeline(steps=[('imputer', IterativeImputer(random_state=1)),
    ('scaler', StandardScaler())])

categorical_features = [0, 1]
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# Tạo bộ preprocessor và đưa các numeric và transformers vào 
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ])

# 5.1.4 Store models to files, to compare latter
import joblib
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    #print(model)
    return model


# In[5]: TRAIN AND EVALUATE MODELS with RandomForestRegressor model
# Tạo pipeline và đưa các pre và regressor vào. Trong preprocessor được chuẩn bị transformed và regreeesor 
# được tạo bằng cách dùng Adabootingregressor() và RandomForestRegressor() làm công cụ ước tính cơ sở của nó
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
a=1
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', AdaBoostRegressor(base_estimator=RandomForestRegressor(random_state=1, n_estimators=1000), learning_rate=.1))])
model.fit(X_train, y_train)
store_model(model)
#model = joblib.load('saved_objects/Pipeline_model.pkl')
print(model.score(X_train, y_train))
# kết quả là 99,8%
# dự đoán thử trên bộ validation 
y_pred = model.predict(X_val)
y_pred = y_pred.astype(int)
y_pred = [0 if i < 0 else i for i in y_pred]
print(model.score(X_val, y_val))
df=pd.DataFrame({'Thực tế': y_val, 'Dự đoán':y_pred})
df
#kết quả là 95%
# Dự đoán trên bộ test là tập dữ liệu năm 2020 
prediction = model.predict(test_set_2)
prediction = prediction.astype(int)
prediction = [0 if i < 0 else i for i in prediction]
print(model.score(test_set_2, candidate_votes))
df=pd.DataFrame({'Thực tế': candidate_votes, 'Dự đoán':prediction})
df
#kết quả là 91%
#Thêm cột tên predicted cho bộ test để các giá trị dự đoán hiện ra đem so sánh với giá trị thực
test_set['predicted'] = prediction
test_set
# Thử đếm số phiếu dự đoán và so sánh với số phiếu thực tế 
# Kết quả cho thấy rằng phiếu đảng dân chủ nhiều hơn khoảng 18 triệu so với dự đoán
# phiếu cộng hoà khoảng 13,5 triệu so với dự đoán , vậy có hơn 18+13,5 triệu phiếu 
# nhiều hơn so với dự đoán.
# Đảng dân chủ có hơn 2 triệu phiếu so với đảng cộng hoà vì vậy có khả năng họ sẽ thắng nếu không có can thiệp nào xảy ra

test_set.groupby('party')['predicted'].count().plot.bar(ylim=0)
plt.show()
predict_winner = test_set.groupby('party')['predicted'].sum()
predict_winner
winnerr = test_set.groupby('party')['candidatevotes'].sum()
winnerr



