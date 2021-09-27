# Link Dataset: laydulieu.com
# In[0]: IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
# In[1]: LOOK AT THE BIG PICTURE (DONE)
# In[2]: GET THE DATA (DONE). LOAD DATA
raw_data = pd.read_csv(r'C:\Users\Vi Diep\OneDrive\Tài liệu\GiaXeMay_2.csv')
# In[3] DISCOVER THE DATA TO GAIN INSIGHTS
print('n____________________________________ Dataset info ____________________________________')
print(raw_data.info())              
print('n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(10)) 
print('n____________________________________ Counts on a feature ____________________________________')
print(raw_data['KIỂU XE'].value_counts()) 
print('n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())    
print('n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.iloc[[0,5,20], [2, 7]] ) 
# 3.2 Scatter plot bw 2 features
if 0:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="SỐ KM ĐÃ ĐI", alpha=0.2)
    plt.savefig('figuresscatter_2_feat.png', format='png', dpi=300)
    plt.show()
if 0:
    raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="NĂM SẢN XUẤT", alpha=0.2)
    plt.savefig('figuresscatter_1_feat.png', format='png', dpi=300)
    plt.show()
# 3.3 Scatter plot bw every pair of features
if 0:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["GIÁ - TRIỆU ĐỒNG", "SỐ KM ĐÃ ĐI", "NĂM SẢN XUẤT"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note histograms on the main diagonal
    plt.savefig('figuresscatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

# 3.4 Plot histogram of numeric features
if 0:
    raw_data.hist(figsize=(10,5)) 
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figureshist_raw_data.png', format='png', dpi=300) # must save before show()
    plt.show()
# 3.5 Compute correlations bw features
corr_matrix = raw_data.corr()
print(corr_matrix) # print correlation matrix
print(corr_matrix["GIÁ - TRIỆU ĐỒNG"].sort_values(ascending=False)) # print correlation bw a feature and other features

# In[04]: PREPARE THE DATA 
# 4.1 Remove unused features
raw_data.drop(columns = ["STT", "GIỐNG NHAU - LOẠI", "GIỐNG NHAU - NHU CẦU", "GIỐNG NHAU - TỈNH", "QUẬN HUYỆN","PHƯỜNG XÃ" ,"NGÀY ĐĂNG"], inplace=True) 
# 4.2 Split training-test set and NEVER touch test set until test phase
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
print(train_set.head(4))
# 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["GIÁ - TRIỆU ĐỒNG"].copy()
train_set = train_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 
test_set_labels = test_set["GIÁ - TRIỆU ĐỒNG"].copy()
test_set = test_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 

# 4.4 Define pipelines for processing data. 
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py 

# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

num_feat_names = ['NĂM SẢN XUẤT', 'SỐ KM ĐÃ ĐI'] 
cat_feat_names = ['HÃNG XE', 'DÒNG XE', 'DUNG TÍCH','KIỂU XE'] 

# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value ="Khác", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])    

# 4.4.3 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place 
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])  

# 4.4.5 Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  
# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)

print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature  + 80 cols of onehotvector for categorical features.'%(len(num_feat_names)))

# In[5]: TRAIN AND EVALUATE MODELS with RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state=42) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse    
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
# Compute R2 score and root mean squared error
print('\n____________________________________ RandomForestRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
# 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
print('\n____________________________________ K-fold cross validation ____________________________________')
run_evaluation = 0
if run_evaluation:
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
else:
    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
# In[6]: FINE-TUNE MODELS 
# NOTE: this takes TIME
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 
from sklearn.model_selection import GridSearchCV
    
run_new_search = 0
if run_new_search:
        model = RandomForestRegressor(random_state=42)
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 30, 60], 'max_features': [2, 30, 50, 80]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 30], 'max_features': [2, 30, 50]} ]
            # Train across 5 folds, hence a total of (12+6)*5=90 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")    
else :
     grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
     print_search_result(grid_search, model_name = "RandomForestRegressor")    
# In[7]: ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 

# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_


print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUION")   
# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for rand forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + [""] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# 7.3 Run on test data
processed_test_set = full_pipeline.transform(test_set)


# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')


print("Done")