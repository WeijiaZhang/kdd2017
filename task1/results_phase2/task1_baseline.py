
# coding: utf-8

# In[9]:

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from sklearn.svm import SVR


# In[166]:

# Our small data set
d = {'one':[1,1,1,1,1],
     'two':[2,2,2,2,2],
     'letter':['a','a','b','b','c']}

# Create dataframe
df = pd.DataFrame(d)
letterone = df.groupby(['letter','one']).size()


# In[244]:

# using SVR to predict
train_path = '../dataset/training/trajectories(table 5)_training.csv'
test_path = '../dataset/testing_phase1/trajectories(table 5)_test1.csv'



train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



# In[252]:

def select_hour(dt):
    minute = int(math.floor(dt.minute / 20) * 20)
    second = 0
    dt_new = datetime(dt.year, dt.month, dt.day, dt.hour,minute, 0)
    return dt_new

def time_window(df):
    df.starting_time = pd.to_datetime(df.starting_time)
    df['starting_time'] = df.starting_time.apply(
        select_hour)
    if set(['travel_seq', 'vehicle_id']).issubset(df.columns):
        df.drop(['travel_seq', 'vehicle_id'], axis=1)
    df = df.groupby(['intersection_id', 'tollgate_id', 'starting_time']).sum()
    df = df.reset_index()
    df = df.rename_axis({'travel_time':'avg_travel_time'}, axis='columns')
    return df
    
def generate_features(df):
    in_id = df['intersection_id'].apply(lambda x: ord(x) - 64)
    df['route_id'] = in_id + df['tollgate_id']
    # 分别给月，天，时，分权重...
    df['day_sum'] = df['starting_time'].apply(lambda t: 
              t.month*1 + t.day*1 + t.hour*1+(t.minute*1+20)) 
    return df
    
def slice_feature_label(df):
    df = generate_features(df)
    hour = df['starting_time'].dt.hour;
    df_prev = df.loc[((hour >= 6) & (hour < 8)) | ((hour >= 15) & (hour < 17))]
    df_follow = df.loc[(hour >= 8) & (hour < 10) | ((hour >= 17) & (hour < 19))]
    train_features = df_prev.ix[:, 'avg_travel_time':'day_sum']
    train_labels = df_follow.ix[:, 'avg_travel_time']
    return train_features, train_labels




# In[274]:

train_df = time_window(train_df)
test_df = time_window(test_df)
test_df.head()


# In[265]:

train_features, train_labels = slice_feature_label(train_df)
test_features = generate_features(test_df)
test_features = test_features.ix[:, 'avg_travel_time':'day_sum']
test_features.head(10)


# In[260]:

print train_features.shape[0]
print train_labels.shape[0]
print test_features.shape[0]
test_features.to_csv('../test_features.csv', index=False)


# In[264]:

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
train_num = train_features.shape[0]
svr_rbf.fit(train_features, train_labels[:train_num])


# In[269]:

test_pred = svr_rbf.predict(test_features)
print test_pred.size


# In[296]:

intersection_id = test_df.intersection_id
tollgate_id = test_df.tollgate_id
start_time= test_df.starting_time.apply(lambda dt: dt+timedelta(hours=2))
end_time = start_time.apply(lambda dt: dt + timedelta(minutes=20))
time_window  = '['+ start_time.astype(np.str) + ',' + end_time.astype(np.str) + ')'
data = {'intersection_id':intersection_id, 'tollgate_id': tollgate_id,
        'time_window': time_window, 'avg_travel_time': test_pred}
columns = ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
test_baseline = pd.DataFrame(data=data, columns=columns)
test_baseline.to_csv('../test_baseline.csv', index=False)
test_baseline.head()


# In[ ]:




# In[ ]:




# In[ ]:



