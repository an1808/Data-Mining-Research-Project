#load dataset

import pandas as pd 
import numpy as np
import os
from scipy.stats import chi2_contingency


# =============================================================================
#   Data Cleaning \n",
# =============================================================================
# =============================================================================
#   Merge Files into a single dataset\n",
# =============================================================================\n",
    
folder_loc = r'D:/BITS/SEM 6/Academics/CS F415/DMFinalTerm_TeamID_34/data/raw data'
output_loc = r'D:/BITS/SEM 6/Academics/CS F415/DMFinalTerm_TeamID_34data/datafiles/data2.csv'
    
file_list = os.listdir(folder_loc)
df_dataset = pd.read_csv(folder_loc + file_list[0])
for file in file_list[1:]:
    full_name = folder_loc + file
    df = pd.read_csv(full_name)
    df_dataset = df_dataset.append(df)
    f_dataset = df_dataset.reset_index()   
    #=============================================================================
    # remove null points\n",
    # =============================================================================
   
    df_dataset = df_dataset.drop(columns=['BlockName', 'CreatedOn','index','Season'])
    col = df_dataset.columns
    df_dataset.dropna(subset = col, inplace=True)
   ,
    #Remove rows which have '0' as an entry\n",
    df_dataset = df_dataset[df_dataset['Category']!='0']
    df_dataset = df_dataset[df_dataset['Sector']!='0']
    df_dataset = df_dataset[df_dataset['Crop']!='0']
    df_dataset = df_dataset[df_dataset['QueryType']!='0']
    df_dataset = df_dataset[df_dataset['StateName']!='0']
    df_dataset = df_dataset[df_dataset['DistrictName']!='0']
    
    df_dataset = df_dataset.reset_index() 
   
    df_dataset.to_csv(output_loc,index = 0) 
#preprocess
#feature selection
def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


input_loc = r'D:/BITS/SEM 6/Academics/CS F415/DMFinalTerm_TeamID_34/data/datafiles/data2.csv'
data = pd.read_csv(input_loc) 

# =============================================================================
# 

#factors_paired = [(i,j) for i in df.columns.values for j in df.columns.values] 
#
#chi2, p_values =[], []
#
#for f in factors_paired:
#    if f[0] != f[1]:
#        chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))   
#        chi2.append(chitest[0])
#        p_values.append(chitest[1])
#    else:      # for same factor pair
#        chi2.append(0)
#        p_values.append(0)
#
#chi2 = np.array(chi2).reshape((6,6)) # shape it as a matrix
#chi2 = pd.DataFrame(chi2, index=df.columns.values, columns=df.columns.values) # then a df for convenience


# =============================================================================

#feature creation
df = pd.get_dummies(data)
#feature selection
corr_mat = df.corr(method=histogram_intersection)
#dropped sector as it is highly correlated to category

data = data.drop(['Sector'], axis=1)
#aggregating columns with similar meaning

#one hot encoding
#data = pd.get_dummies(data)
data = pd.get_dummies(data, prefix=['Category', 'Crop','StateName','DistrictName'], columns=['Category', 'Crop','StateName','DistrictName'])
##grouping of rows with exact same values
##cols = data_oh.columns
##c = ['Category_Animal', 'Category_Avian']
##print(cols)
#data_count = data_oh.groupby(c).count() 
#standardisation
#df = data_oh
#normalized_df=(df-df.mean())/df.std()
#Random forest training
cols = data.columns
training_columns = list(cols[1:])
response_column = cols[0]
# =============================================================================
# =============================================================================
# pip install requests
# pip install tabulate
# pip install scikit-learn
# pip uninstall h2o
# pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
# pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o --user
# =============================================================================
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split
h2o.init()

train, test = train_test_split(data, test_size=0.2, stratify=data['QueryType'])
#train = train.values.tolist()
#train = train[1:]
#test = test.values.tolist()
#test = test[1:]
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
 # Define model
model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
 
 # Train model
model.train(x=training_columns, y=response_column, training_frame=train)
 # Model performance
performance = model.model_performance(test_data=test)
 
print(performance)
# =============================================================================
#rf testing
