import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as ncolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import operator
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline
confirmed_cases=pd.read_csv('time_series_covid19_confirmed_global.csv')
death_cases=pd.read_csv('time_series_covid19_deaths_global.csv')
recovered_cases=pd.read_csv('time_series_covid19_recovered_global.csv')
confirmed_cases.head()
confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
death=death_cases.loc[:,cols[4]:cols[-1]]
recoveries=recovered_cases.loc[:,cols[4]:cols[-1]]
dates=confirmed.keys()
world_cases=[]
total_deaths=[]
total_recovered=[]
mortality_rate=[]
for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=death[i].sum()
    recovered_sum=recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    mortality_rate.append(death_sum/confirmed_sum)
days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)
dates = confirmed_cases.columns[4:]
days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)
days_in_future=30
future_forcast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forcast[:-10]
start='1/22/20'
start_date=datetime.datetime.strptime(start,'%m/%d/%y')
future_forcast_date=[]
future_forcast = range(30)
for i in  range(len(future_forcast)):
    future_forcast_date.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%y'))
latest_confirmed=confirmed_cases[dates[-1]]
latest_death=death_cases[dates[-1]]
latest_recoveries=recovered_cases[dates[-1]]
unique_countries=list(confirmed_cases['Country/Region'].unique())
unique_countries
country_confirmed_cases=[]
no_cases=[]
for i in unique_countries:
    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_countries.remove(i)
unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()
print("confirmed cases by country")
for i in range(len(unique_countries)):
    print(f"{unique_countries[i]}: {country_confirmed_cases[i]},cases")
unique_provinces=list(confirmed_cases['Province/State'].unique())
outliers=['Summer Olympics 2020','Diamond Princess','Winter Olympics 2022','Holy See','Antarctica','MS Zaandam','Korea, North']
if outliers in unique_provinces:  
        unique_provinces.remove(outliers)
provience_confirmed_cases=[]
no_cases=[]
for i in unique_provinces:
    cases=latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases>0:
        provience_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_provinces.remove(i)
nan_indies=[]
for i in range(len(unique_provinces)):
    if type(unique_provinces[i])==float:
        nan_indies.append(i)
unique_provinces=list(unique_provinces)
for i in nan_indies:
    unique_provinces.pop(i)
    provience_confirmed_cases.pop(i)
visual_unique_countries=[]
visual_confirmed_cases=[]
others=np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
visual_unique_countries.append('others')
visual_confirmed_cases.append(others)
X_train_confirmed,X_test_confirmed,y_train_confirmed,y_test_confirmed = train_test_split(days_since_1_22,world_cases,test_size=0.15,shuffle=False)
future_forcast = np.array(range(0, 30)).reshape(-1, 1) 
# Reshape future_forcast to a 2D array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,max_depth=3,random_state=0)
rf.fit(X_train_confirmed,y_train_confirmed)
rf_pred=rf.predict(future_forcast)
print('MSE',mean_squared_error(rf_pred,future_forcast))
print('MAE',mean_absolute_error(rf_pred,future_forcast))
print('RMSE',np.sqrt(mean_squared_error(rf_pred,future_forcast)))
days_in_future = 30
future_forcast = np.array([i for i in range(len(days_since_1_22) + days_in_future)]).reshape(-1, 1)

future_predictions = linear_model.predict(future_forcast[-days_in_future:])

for i, prediction in enumerate(future_predictions):
    print(f"Day {i + 1}: Predicted cases = {prediction[0]}")

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(days_since_1_22, world_cases, label="Actual Cases")
plt.plot(future_forcast, linear_model.predict(future_forcast), linestyle="dashed", color="red", label="Predicted Cases")
plt.title("COVID-19 Cases Prediction")
plt.xlabel("Days Since 1/22/20")
plt.ylabel("Number of Cases")
plt.legend()
plt.show()
import numpy as np
from sklearn.metrics import r2_score

x = np.array([1, 6, 1])
y = np.array([1, 9, 3])  

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Added rcond=None for compatibility
print(m, c)

f = m * x + c
print(f)

yminusf2 = (y - f)**2
rss = np.sum(yminusf2)

mean = float(sum(y) / float(len(y)))
yminusmean = (y - mean)**2
tss = np.sum(yminusmean)

R2 = 1 - (rss / tss)  
print("R2:", R2) # changed variable to avoid shadowing
print(r2_score(y, f))