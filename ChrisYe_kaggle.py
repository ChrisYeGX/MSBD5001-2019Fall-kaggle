# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:32:27 2019
new project
@author: Chris Ye
"""
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
def feature_enginer(train,flag,flag2):
    #process the data
    train['money']=train['is_free'].apply(lambda x: 1 if x==0 else 0)
    train2=train
    if type(flag)==int:
        z=train['genres'].str.get_dummies(sep=',')
        d=train['categories'].str.get_dummies(sep=',')
        e=train['tags'].str.get_dummies(sep=',')
   # print(len(z.columns))
    r=train2['genres'].values
    r2=train2['categories'].values
    """
    Use labelEncoder to handle the catogorical data (column:
        genres, tags, catogorical.
    )
    """
    if type(flag)!=int:
        z=flag
        encoder = LabelEncoder()
        encoder.fit(z.columns)
        b=[]
        for i in range(len(r)):
            c=encoder.transform(r[i].split(','))
            b.append(c)
        train2['genres']=b
        z2=train2['genres'].str.get_dummies(sep=',')
        df2=pd.DataFrame(np.zeros((z2.shape[0],len(z.columns))),columns=z.columns)
        for i in range(len(b)):
            for j in b[i]:
                df2.iat[i,j]=1.0
        train1=pd.merge(train, df2,left_index=True,right_index=True)
        d=flag2
        encoder = LabelEncoder()
        encoder.fit(d.columns)
        b=[]
        for i in range(len(r2)):
            c=encoder.transform(r2[i].split(','))
            b.append(c)
        train2['categories']=b
        z2=train2['categories'].str.get_dummies(sep=',')
        df3=pd.DataFrame(np.zeros((z2.shape[0],len(d.columns))),columns=d.columns)
        for i in range(len(b)):
            for j in b[i]:
                df3.iat[i,j]=1.0        
        train2=pd.merge(train, df3,left_index=True,right_index=True)
        #print(train2.shape)
#        d2=train2['genres'].str.get_dummies(sep=',')
#        df2=pd.DataFrame(np.zeros((d2.shape[0],len(d.columns))),columns=.columns)
#        for i in range(len(b)):
#            for j in b[i]:
#                df2.iat[i,j]=1.0
#        train1=pd.merge(train, df2,left_index=True,right_index=True)
#        z2=train2['genres'].str.get_dummies(sep=',')
#        df2=pd.DataFrame(np.zeros((z2.shape[0],len(z.columns))),columns=z.columns)
#        for i in range(len(b)):
#            for j in b[i]:
#                df2.iat[i,j]=1.0
#        train1=pd.merge(train, df2,left_index=True,right_index=True)
    else:
        train1=pd.merge(train, z,left_index=True,right_index=True)
       # train2=pd.merge(train, d,left_index=True,right_index=True)
       # print(train2.shape)
       # train2=pd.merge(train1, d,left_index=True,right_index=True)
       # train3=pd.merge(train2, e,left_index=True,right_index=True)
    all_data=train1
    '''
    Add some new features: time/ buy day,buymonth,buyyear
    releaseday,releasemonth,releasemonth
    totoalreview, netreview, positive review ratio
    '''
    all_data.fillna(0, inplace=True)
    all_data['buydate_time']=pd.to_datetime(all_data['purchase_date'])
    all_data['reldate_time']=pd.to_datetime(all_data['release_date'])
    all_data['date_last']=all_data['buydate_time']-all_data['reldate_time']
    all_data['date_last']=all_data['date_last'].dt.days
    all_data['date_last']=all_data['date_last'].map(lambda x: 0 if x<0 else x)
    all_data['buy_month']=pd.to_datetime(all_data['release_date'])
    all_data['buy_day']=pd.to_datetime(all_data['release_date'])
    all_data['release_month']=pd.to_datetime(all_data['release_date'])
    all_data['release_year']=pd.to_datetime(all_data['release_date'])
    all_data['release_day']=pd.to_datetime(all_data['release_date'])
    all_data['buy_date']=pd.to_datetime(all_data['purchase_date'])
    s=len(all_data['buy_date'])
    for i in range(s):
        all_data['buy_date'][i]=all_data['buydate_time'][i].year
        all_data['buy_month'][i]=all_data['buydate_time'][i].month
        all_data['buy_day'][i]=all_data['buydate_time'][i].day
        all_data['release_year'][i]=all_data['reldate_time'][i].year
        all_data['release_month'][i]=all_data['reldate_time'][i].month
        all_data['release_day'][i]=all_data['reldate_time'][i].day
    all_data['net_positive']=all_data['total_positive_reviews']-all_data['total_negative_reviews']
    all_data['totalreview']=all_data['total_positive_reviews']+all_data['total_negative_reviews']
    all_data['positivereviewratio']=all_data['total_positive_reviews']/all_data['totalreview']
    all_data.fillna(0, inplace=True)
    first_try=all_data   
    final=first_try.drop(['id','is_free','genres','tags','categories','purchase_date','release_date','buydate_time','reldate_time'],axis=1)
    #standerlization
#    final['price'] = (final['price']-final['price'].min())/(final['price'].max()-final['price'].min())
#    final['total_positive_reviews'] = (final['total_positive_reviews']-final['total_positive_reviews'].min())/(final['total_positive_reviews'].max()-final['total_positive_reviews'].min())
#    final['total_negative_reviews'] = (final['total_negative_reviews']-final['total_negative_reviews'].min())/(final['total_negative_reviews'].max()-final['total_negative_reviews'].min())
#    final['date_last'] = (final['date_last']-final['date_last'].min())/final['date_last'].max()-final['date_last'].min()
#    final['net_positive'] = (final['net_positive']-final['net_positive'].min())/final['net_positive'].max()-final['net_positive'].min()
#    
#    final['price'] = (final['price']-final['price'].mean())/final['price'].std()
#    final['total_positive_reviews'] = (final['total_positive_reviews']-final['total_positive_reviews'].mean())/final['total_positive_reviews'].std()
#    final['total_negative_reviews'] = (final['total_negative_reviews']-final['total_negative_reviews'].mean())/final['total_negative_reviews'].std()
#    final['date_last'] = (final['date_last']-final['date_last'].mean())/final['date_last'].std()
#    final['net_positive'] = (final['net_positive']-final['net_positive'].mean())/final['net_positive'].std()
    return final,z,d

# use xgboost as tool to model the data
def model(train,test):
    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.6,
        'max_depth': 5,
        'lambda': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'min_child_weight': 6,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    X_all = train.drop(['playtime_forever'], axis=1)
    y_all = train['playtime_forever']
    dtrain = xgb.DMatrix(X_all, y_all)
    dtest = xgb.DMatrix(test)
    num_rounds = 300
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)
    y_predict= model.predict(dtest)
    return y_predict

train = pd.read_csv('train_data.csv')
test=pd.read_csv('test.csv')
train_process,z,d=feature_enginer(train,0,0)
test_process,nothing1,nothing2=feature_enginer(test,z,d)
result = model(train_process,test_process)
d=pd.DataFrame(columns=['playtime_forever'])
d['playtime_forever']=result
d.to_csv("finaloutput.csv")