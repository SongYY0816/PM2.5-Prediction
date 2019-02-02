#%%
import pandas as pd
import numpy as np
import math
from math import sqrt
from numpy import linalg
from numpy.core.umath_tests import inner1d
#--------------***********************----------------#
#--------------*** obtain coordinate ***--------------#
##read air_quality coordinates
excel_path="/Users/song/Desktop/beijing_airquality_station.xlsx"
air=pd.read_excel(excel_path)
air.rename(columns={"Station ID":"station_id_x"},inplace=True)

##read grid coordinates
grid=pd.read_csv("/Users/song/Desktop/Beijing_grid_weather_station.csv",header=-1)
grid.columns=["ID","latitude1","longitude1"]
cols = list(grid)
cols.insert(1,cols.pop(cols.index('longitude1')))
grid = grid.loc[:,cols]

##read observe coordinates
observe=pd.read_csv("/Users/song/Desktop/observedWeather_201701-201801.csv")
obser=observe.drop_duplicates("station_id")[['station_id','longitude','latitude']]
obser=obser.reset_index(drop=True)

#--------------***********************--------------------------------------------------#
#--------------*** compute distances and pair the minimal one with air ***--------------#
##compute the distance between air and grid
##compute the distance between air and observed data
def find(x):
    l3=[]
    for indexs in air.index:
        l1=[]
        l2=[]
    #print(air.loc[indexs].values[0:3])
        vector1=np.array(air.loc[indexs].values[1:3])
        for i in x.index:
            vector2=np.array(x.loc[i].values[1:3])
            l1.append(np.linalg.norm(vector1-vector2))
            j=l1.index(min(l1))
       
        l2.append(air.loc[indexs].values[0:1][0])
        l2.append(air.loc[indexs].values[1:2][0])
        l2.append(air.loc[indexs].values[2:3][0])
        l2.append(x.loc[j].values[0:1][0])
        l2.append(x.loc[j].values[1:2][0])
        l2.append(x.loc[j].values[2:3][0])
        l2.append(min(l1))
        l3.append(l2)
    return l3

of=find(obser)
gf=find(grid)
#gf=pd.DataFrame(gf)
#of=pd.DataFrame(of)

final=[]
for i in range(0,35):
    if gf[i][0]==of[i][0]:
        if gf[i][6]<of[i][6]:
            final.append(gf[i])
        elif gf[i][6]>of[i][6]:
            final.append(of[i])
        else:
            final.append(gf[i])
final=pd.DataFrame(final)
'''
listall=[]
for i in range(0,35):
    list1=[]
    list1.append(final[0][i][0])
    list1.append(final[1][i][0])
    listall.append(list1)
#listall=pd.DataFrame(listall)
'''

#--------------***********************-------------------------------#
#--------------*** merge tables of air,grid,observe ***--------------#
##merge tables--grid
grid1=pd.read_csv("/Users/song/Desktop/gridWeather_201701-201803.csv")
grid2=pd.read_csv("/Users/song/Desktop/gridWeather_201804.csv")
del grid2['id']
del grid1['longitude']
del grid1['latitude']
grid1.rename(columns={'stationName':'station_id', 'utc_time':'time',
                      'wind_speed/kph':'wind_speed'}, inplace = True)
col_name = grid1.columns.tolist()
col_name.insert(2,'weather')
grid1.reindex(columns=col_name)
gridall = pd.concat([grid1,grid2],axis=0,sort=False)

gridfinal=gridall[gridall['station_id'].isin(['beijing_grid_303','beijing_grid_282','beijing_grid_304',
                                  'beijing_grid_263','beijing_grid_262','beijing_grid_239',
                                  'beijing_grid_261','beijing_grid_238','beijing_grid_301',
                                  'beijing_grid_323','beijing_grid_366','beijing_grid_240',
                                  'beijing_grid_265','beijing_grid_224','beijing_grid_414',
                                  'beijing_grid_452','beijing_grid_385','beijing_grid_278',
                                  'beijing_grid_216'])]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars=gridfinal['weather']
gridfinal['weather'] = le.fit_transform(cat_vars.tolist())
gridfinal.groupby('station_id').agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
gridfinal=gridfinal.fillna(-1)

##merge tables--observed
obser1=pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/observedWeather_201701-201801.csv')
obser2=pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/observedWeather_201802-201803.csv')
obser3=pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/observedWeather_201804.csv')

del obser3['id']
obser3.rename(columns={'time':'utc_time'}, inplace = True)
obser2.insert(1,'longitude',None)
obser2.insert(2,'latitude',None)
obserall=pd.concat([obser1,obser2,obser3], axis=0, join='outer', sort=False,ignore_index=True) 

obserfinal=obserall[obserall['station_id'].isin(['chaoyang_meo','hadian_meo','fengtai_meo',
                                                'shunyi_meo','pingchang_meo','pinggu_meo',
                                                'huairou_meo','miyun_meo','yanqing_meo'])]

del obserfinal['longitude']
del obserfinal['latitude']
obserfinal.columns=['station_id','time','temperature','pressure','humidity','wind_direction','wind_speed','weather']

obserfinal[obserfinal.isnull().values==True]
obserfinal.insert(1,'date',obserfinal['time'])
#先复制原来的列
obserfinal.insert(1,'hour',obserfinal['time'])
obserfinal["date"]=obserfinal["time"].map(lambda x:x.split()[0])
#分别处理新旧两列
obserfinal["hour"]=obserfinal["time"].map(lambda x:x.split()[1])
#print(obserfinal.isnull().sum()[obserfinal.isnull().sum()!=0])
#obserfinal['wind_direction'].median()

obserfinal['wind_direction']=obserfinal['wind_direction'].fillna(method='pad')
obserfinal['wind_speed']=obserfinal['wind_speed'].fillna(method='pad')
#.groupby(['station_id','time']).wind_direction.median()
obserfinal.sort_values(by=['station_id','time'],ascending=True,inplace=True)
obserfinal['wind_direction']=obserfinal['wind_direction'].replace(999017,np.NaN)
obserfinal['wind_direction']= obserfinal['wind_direction'].fillna(method='pad')

##merge tables--air quality(targets)
target1 = pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/airQuality_201701-201801.csv')
target2 = pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/airQuality_201802-201803.csv')
target3 = pd.read_csv('/Users/song/Desktop/MSBD5002PROJECT_data/aiqQuality_201804.csv')

del target3['id']
target3.rename(columns={'station_id':'stationId', 'time':'utc_time',
                       'PM25_Concentration':'PM2.5', 'PM10_Concentration':'PM10',
                       'NO2_Concentration':'NO2', 'CO_Concentration':'CO',
                       'O3_Concentration':'O3', 'SO2_Concentration':'SO2'}, inplace = True)

target =pd.concat([target1,target2,target3],axis=0,sort=False)

target =target.replace(' ', np.nan)
target['CO'] = target['CO'].fillna(method='pad')
target['NO2'] = target['NO2'].fillna(method='pad')
target['O3'] = target['O3'].fillna(method='pad')
target['PM10'] = target['PM10'].fillna(method='pad')
target['PM2.5'] = target['PM2.5'].fillna(method='pad')
target['SO2'] = target['SO2'].fillna(method='pad')

target.rename(columns={'stationId':'station_id','utc_time':'time'}, inplace = True)

#--------------***********************-----------------#
#--------------*** data preprocessing ***--------------#
##merge three tables into air quality
final.columns=['station_id','s1','s2','station','i1','i2','t']
f=final[['station_id','station']]
he=pd.merge(target,f,on='station_id')
obserfinal['stt']=obserfinal['station_id']+obserfinal['time']
#obserfinal.head(10)
he['stt']=he['station']+he['time']
#he.head(10)
gridfinal.head(10)
gridfinal['stt']=gridfinal['station_id']+gridfinal['time']
q=pd.merge(he,obserfinal,on='stt')
p=pd.merge(he,gridfinal,on='stt')
q.head(10)
re=pd.concat([q,p], axis=0, sort=False,ignore_index=True) 
re.isnull().any()
#re.to_csv('hhh.csv')
#re.isnull().sum()[re.isnull().sum() !=0]
#gridfinal.isnull().sum()[gridfinal.isnull().sum() !=0]
#obserfinal.isnull().sum()[obserfinal.isnull().sum() !=0]

re=re.fillna(-1000)
#re.isnull().any()

##training data feature building
train=pd.merge(re,air,on='station_id_x')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars=train['weather']
train['weather'] = le.fit_transform(cat_vars.tolist())

import datetime
train['time_x']=train['time_x'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
cat_vars1=train['time_x']
train['time_x'] = le.fit_transform(cat_vars1.tolist())

X=train[['longitude','latitude','time_x','temperature','pressure','humidity',
        'wind_direction','wind_speed','weather']]

Y=train[['PM2.5','PM10','O3']]

#--------------***********************----------#
#--------------*** train model ***--------------#
##use random forest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
rf = RandomForestRegressor(max_depth=25,max_features=4 ,n_estimators=500)
rf.fit(X_train, y_train)

##evaluate the model
def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    c = np.array(actual) + np.array(predicted)
    denominator = c
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=(denominator!=0), casting='unsafe'))

m=rf.predict(X_test)
n=y_test
smape(n, m)

#--------------***********************---------#
#--------------*** prediction ***--------------#
##process predictive data
gridpre=pd.read_csv("/Users/song/Desktop/MSBD5002PROJECT_data/gridWeather_20180501-20180502.csv")
obserpre=pd.read_csv("/Users/song/Desktop/MSBD5002PROJECT_data/observedWeather_20180501-20180502.csv")
del gridpre['id']
final.columns=['aq','s1','s2','station_id','longitude','latitude','t']
f=final[['aq','station_id','longitude','latitude']]
gt=pd.merge(gridpre,f,on='station_id')
del obserpre['id']
ot=pd.merge(obserpre,f,on='station_id')

gt=gt[['aq','longitude','latitude','time','temperature','pressure',
       'humidity','wind_direction','wind_speed','weather']]
ot=ot[['aq','longitude','latitude','time','temperature','pressure',
       'humidity','wind_direction','wind_speed','weather']]
test=pd.concat([gt,ot],axis=0)
test.sort_values(by=['aq' ,'time'], ascending=True, inplace=True)

##the same method as dealing with training data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_vars=test['weather']
test['weather'] = le.fit_transform(cat_vars.tolist())

import datetime
test['time']=test['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
cat_vars1=test['time']
test['time'] = le.fit_transform(cat_vars1.tolist())
Z=test[['longitude','latitude','time','temperature','pressure','humidity',
        'wind_direction','wind_speed','weather']]

result=rf.predict(Z)
result=pd.DataFrame(result)

##to csv submission
tr=np.array(test['aq'])
for i in range(0,7):
    for m in range(48*i,48*(i+1)):
        tr[m]=tr[m]+'#'+str(m-i*48)
        #print(tr[m])
for j in range(336,382):
    tr[j]=tr[j]+'#'+str(j-336)
    #print(tr[j])
for j in range(382,430):
    tr[j]=tr[j]+'#'+str(j-382)
    #print(tr[j])
for j in range(430,476):
    tr[j]=tr[j]+'#'+str(j-430)
    #print(tr[j])
for j in range(476,524):
    tr[j]=tr[j]+'#'+str(j-476)
    #print(tr[j])
for j in range(524,572):
    tr[j]=tr[j]+'#'+str(j-524)
    #print(tr[j])
for j in range(572,618):
    tr[j]=tr[j]+'#'+str(j-572)
    #print(tr[j])
for j in range(618,666):
    tr[j]=tr[j]+'#'+str(j-618)
    #print(tr[j])
for j in range(666,714):
    tr[j]=tr[j]+'#'+str(j-666)
    #print(tr[j])
for j in range(714,760):
    tr[j]=tr[j]+'#'+str(j-714)
    #print(tr[j])
for j in range(760,808):
    tr[j]=tr[j]+'#'+str(j-760)
    #print(tr[j])
for j in range(808,856):
    tr[j]=tr[j]+'#'+str(j-808)
    #print(tr[j])
for j in range(856,902):
    tr[j]=tr[j]+'#'+str(j-856)
    #print(tr[j])
for j in range(902,948):
    tr[j]=tr[j]+'#'+str(j-902)
    #print(tr[j])
for j in range(948,994):
    tr[j]=tr[j]+'#'+str(j-948)
    #print(tr[j])
for j in range(994,1042):
    tr[j]=tr[j]+'#'+str(j-994)
    #print(tr[j])
for j in range(1042,1088):
    tr[j]=tr[j]+'#'+str(j-1042)
    #print(tr[j])
for j in range(1088,1136):
    tr[j]=tr[j]+'#'+str(j-1088)
    #print(tr[j])
for j in range(1136,1184):
    tr[j]=tr[j]+'#'+str(j-1136)
    #print(tr[j])
for j in range(1184,1230):
    tr[j]=tr[j]+'#'+str(j-1184)
    #print(tr[j])
for j in range(1230,1278):
    tr[j]=tr[j]+'#'+str(j-1230)
    #print(tr[j])
for j in range(1278,1324):
    tr[j]=tr[j]+'#'+str(j-1278)
    #print(tr[j])
for j in range(1324,1370):
    tr[j]=tr[j]+'#'+str(j-1324)
    #print(tr[j])
for i in range(29,35):
    for m in range(1370+48*(i-29),1370+48*(i-29)+48):
        tr[m]=tr[m]+'#'+str(m-i*48+22)
        #print(tr[m])

tr=pd.DataFrame(tr)
ans=pd.concat([tr,result], axis=1)
ans.columns=['test_id','PM2.5','PM10','O3']
#ans.to_csv("5002final.csv")