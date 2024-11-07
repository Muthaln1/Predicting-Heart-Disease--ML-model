## Introduction:

I assume I am working in an R&D team focused on providing data solutions. The team is tasked with accurately predicting the likelihood of a new patient developing heart disease in the future

There are several factors that might lead to cardiovascular disease, such as an unhealthy diet, smoking, diabetes, age, stress, and family history, among others. Some of the features that could help us build the prediction model are provided. We will use them for our purposes

## Goal:
+ To accurately predict the likelihood of a new patient having heart disease in the future

### Heart Disease dataset contains:

`Age`: age of the patient [years]


`Sex`: sex of the patient [M: Male, F: Female]


`ChestPainType`: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]


`RestingBP`: resting blood pressure [mm Hg]


`Cholesterol`: serum cholesterol [mm/dl]


`FastingBS`: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]


`RestingECG`: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]


`MaxHR`: maximum heart rate achieved [Numeric value between 60 and 202]


`ExerciseAngina`: exercise-induced angina [Y: Yes, N: No]


`Oldpeak`: oldpeak = ST [Numeric value measured in depression]


`ST_Slope`: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]


`HeartDisease`: output class [1: heart disease, 0: Normal]

## EDA (Exploratory Data) Analysis:


```python
# Loading the data 
import pandas as pd

data = pd.read_csv('heart_disease_prediction.csv')

print(data.head(2))
```

       Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR  \
    0   40   M           ATA        140          289          0     Normal    172   
    1   49   F           NAP        160          180          0     Normal    156   
    
      ExerciseAngina  Oldpeak ST_Slope  HeartDisease  
    0              N      0.0       Up             0  
    1              N      1.0     Flat             1  



```python
# Checking the observations and features
print(f'Observations:{data.shape[0]}\n',f'Features:{data.shape[1]}')
```

    Observations:918
     Features:12



```python
# Checking for the presence of null values in the data:

print(data.isna().sum())
```

    Age               0
    Sex               0
    ChestPainType     0
    RestingBP         0
    Cholesterol       0
    FastingBS         0
    RestingECG        0
    MaxHR             0
    ExerciseAngina    0
    Oldpeak           0
    ST_Slope          0
    HeartDisease      0
    dtype: int64



```python
# Data types 
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 918 entries, 0 to 917
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Age             918 non-null    int64  
     1   Sex             918 non-null    object 
     2   ChestPainType   918 non-null    object 
     3   RestingBP       918 non-null    int64  
     4   Cholesterol     918 non-null    int64  
     5   FastingBS       918 non-null    int64  
     6   RestingECG      918 non-null    object 
     7   MaxHR           918 non-null    int64  
     8   ExerciseAngina  918 non-null    object 
     9   Oldpeak         918 non-null    float64
     10  ST_Slope        918 non-null    object 
     11  HeartDisease    918 non-null    int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 86.2+ KB
    None


### Categorical Data Analysis:

# Findings:

+ There are a total of 725 male and 193 female observations available for analysis, indicating that the dataset is skewed towards males. Approximately 63% of males had heart disease, while 26% of females did. This suggests that males may be more prone to heart disease compared to females



+ The 'ASY' category of Chest Pain Type has a higher count (392) of patients with heart disease. The Asymptotic type implies that chest pain is not reported by the patients but is detected in tests



+ Out of 918 observations, 508 patients reported of having Heart Disease


```python
# Target Variable:  Evenly distributed
#count of Individuals by HeartDisease

grouped_data_heartD= data.groupby('HeartDisease').agg(count=('HeartDisease','count')).reset_index()

print(f'''Individuals with & without Heart Disease:\n{data['HeartDisease'].value_counts()}''')
```

    Individuals with & without Heart Disease:
    HeartDisease
    1    508
    0    410
    Name: count, dtype: int64



```python
# Dataset is skewed towards Male patients
# Number of Male and Female observations:

normalized_val = data['Sex'].value_counts(normalize = True)*100
print(f'% of Male & Female patients:\n{normalized_val.round()}')
```

    % of Male & Female patients:
    Sex
    M    79.0
    F    21.0
    Name: proportion, dtype: float64


#### Males and Females with HD(Heart Disease)


```python
#26% of females and 63% of males have heart disease

grouped_data_sex= data.groupby('Sex').agg(Total_count=('Sex','count'),HD_count=('HeartDisease','sum')).reset_index()

grouped_data_sex['%_Normalized_HD'] = (grouped_data_sex['HD_count'] / grouped_data_sex['Total_count'] * 100).round()
print(grouped_data_sex)
```

      Sex  Total_count  HD_count  %_Normalized_HD
    0   F          193        50             26.0
    1   M          725       458             63.0


#### Individual with Heart Disease based on Chest pain type


```python
grouped_da= data.groupby(['ChestPainType','HeartDisease']).agg(HD_count=('HeartDisease','sum')).reset_index()
sorted_grouped_da = grouped_da.sort_values(by='HD_count', ascending=False)
print(sorted_grouped_da[sorted_grouped_da['HeartDisease']==1])

# Chest pain category and its counts
grouped_data_ChestPain = data.groupby('ChestPainType').agg(Total_count=('ChestPainType','count')).reset_index()
```

      ChestPainType  HeartDisease  HD_count
    1           ASY             1       392
    5           NAP             1        72
    3           ATA             1        24
    7            TA             1        20


#### Individuals with ECG Levels


```python
grouped_data_ECG = data.groupby('RestingECG').agg(Total_count=('RestingECG','count')).reset_index()

print(grouped_data_ECG)
```

      RestingECG  Total_count
    0        LVH          188
    1     Normal          552
    2         ST          178



```python
# Individuals with and without Exercise Angina
print(data['ExerciseAngina'].value_counts())
```

    ExerciseAngina
    N    547
    Y    371
    Name: count, dtype: int64



```python
# Individuals with ST_slope features
print(data['ST_Slope'].value_counts())
```

    ST_Slope
    Flat    460
    Up      395
    Down     63
    Name: count, dtype: int64


### Plotting Categorical Features Using Charts


```python
# importing the matplotlib library to prepare charts
import matplotlib.pyplot as plt

colors = ['lightblue', 'darkorange'] 

fig,(axs) =plt.subplots(2,2,figsize =(10,8))
axs[0,0].bar(grouped_data_heartD['HeartDisease'], grouped_data_heartD ['count'],color = colors)
axs[0,0].set_xticks([0, 1])  # Set the positions of the ticks
axs[0,0].set_xticklabels(['0', '1'])  # Set the labels for the ticks
axs[0,0].set_title('Individuals with and without Heart Disease')
axs[0,0].set_xlabel('HeartDisease')
axs[0,0].set_ylabel('count')
for index, value in enumerate(grouped_data_heartD['count']):
    axs[0,0].text(index, value + 5, str(value), ha='center')
    
axs[0,1].bar(grouped_data_sex['Sex'],grouped_data_sex['Total_count'],color = colors)
axs[0,1].set_title('Sex ratio')
axs[0,1].set_xlabel('Sex')
axs[0,1].set_ylabel('Total_count')
for index, value in enumerate(grouped_data_sex['Total_count']):
    axs[0,1].text(index, value + 5, str(value), ha='center')

colors = ['lightblue', 'lightgrey','darkorange','grey'] 
axs[1,0].bar(grouped_data_ChestPain['ChestPainType'],grouped_data_ChestPain['Total_count'],color = colors)
axs[1,0].set_title('Individuals with ChestPainType')
axs[1,0].set_xlabel('ChestPainType')
axs[1,0].set_ylabel('Total_count')
for index, value in enumerate(grouped_data_ChestPain['Total_count']):
    axs[1,0].text(index, value + 5, str(value), ha='center')

colors = ['lightblue', 'lightgrey','darkorange','grey'] 
axs[1,1].bar(grouped_data_ECG['RestingECG'],grouped_data_ECG['Total_count'],color = colors)
axs[1,1].set_title('Individuals with RestingECG types')
axs[1,1].set_xlabel('RestingECG')
axs[1,1].set_ylabel('Total_count')
for index, value in enumerate(grouped_data_ChestPain['Total_count']):
    axs[1,1].text(index, value + 1, str(value), ha='center')

plt.tight_layout()
plt.show()

```


    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/Categorical%20feature%20charts.png)
    


## Correlation Analysis

# Findings:
+ Heart disease is positively related to the features Oldpeak, Fasting Blood Sugar, Resting Blood Pressure, and Age
+ Heart disease is negatively related to Cholesterol and Maximum Heart Rate (HR)


```python
# High correlation might indicate that skewed distributions are affecting the outcome
# Correlation with the Target - Heart disease

cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
correlations = data[cols].apply(lambda x: x.corr(data['HeartDisease']))

# Print the correlations
print(correlations)
```

    Age            0.282039
    RestingBP      0.107589
    Cholesterol   -0.232741
    FastingBS      0.267291
    MaxHR         -0.400421
    Oldpeak        0.403951
    dtype: float64


## Numerical Data Analysis:

# Findings:

+ The `Age` ranges from 28 to 77 in the dataset
+ `Age` 54,55 & 58 tops the dataset
+ The `Cholesterol` and `Resting BP` fields have a minimum value of zero, which is not feasible for a healthy human body, indicating that the data needs to be handled
+ Mean age of dataset is 54.Meanwhile Median age of Heart Disease patients is 57
+ Out of 918 observations,383 patients of age above 50 have Heart Disease
+ There are 170 patients with heart disease who have blood sugar levels above 120 mg, and 44 patients without a diagnosis of heart disease
+ The level of depression is very high in individuals over the age of 70, high in those over 50, and moderate in those over 30
+ A minimum of 60 and a maximum of 202 have been recorded in terms of heart rate


```python
# Analyzing the `Age` data field
# Age range (28 -77)
description = data['Age'].describe()
print(description)
```

    count    918.000000
    mean      53.510893
    std        9.432617
    min       28.000000
    25%       47.000000
    50%       54.000000
    75%       60.000000
    max       77.000000
    Name: Age, dtype: float64



```python
# Age distribution by Heart Disease
import matplotlib.pyplot as plt
data.boxplot(column= 'Age',by ='HeartDisease',grid=True, figsize=(8,6))
plt.title('Box Plot of Age HD')
plt.suptitle('')
plt.xlabel('Heart Disease Status')
plt.ylabel('Age')
plt.show()
```


    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/Box%20Plot%20of%20Age%20HD.png)
    



```python
mean  = description['mean']
print(f'Mean_age of dataset:{mean:.0f}')
```

    Mean_age of dataset:54


#### Frequency of `Age` in the dataframe:


```python
grouped_data = data.groupby('Age').agg(count=('Age', 'count')).reset_index()

colors =['darkgrey']
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.bar(grouped_data['Age'], grouped_data['count'], width=0.9, edgecolor='black',color = colors)
plt.title('Count of Individuals by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(grouped_data['Age'])
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_31_0.png)
    


#### Mean and Median age of Individuals with Heart Disease


```python
grouped_data_age_median = data[data['HeartDisease']==1]['Age'].median()
grouped_data_age_mean = data[data['HeartDisease']==1]['Age'].mean()

print(f'Median age of Heart Disease patients:{grouped_data_age_median:.0f}')
print(f'Mean age of Heart Disease patients:{grouped_data_age_mean:.0f}')
```

    Median age of Heart Disease patients:57
    Mean age of Heart Disease patients:56


#### Individuals with Heart Disease above `Age` 50


```python
grouped_data_age_HD = data.groupby(['Age','HeartDisease']).agg(HD_count =('HeartDisease','count')).reset_index()

print('Total Heart Disease patients above age 50:',grouped_data_age_HD[(grouped_data_age_HD['Age'] > 50) & (grouped_data_age_HD['HeartDisease']==1)]['HD_count'].sum())
```

    Total Heart Disease patients above age 50: 383



```python
# Since the data is categorized between individuals having heartDisease and not having Heart Disease
# segmenting the data for visualization

HD_count = grouped_data_age_HD[grouped_data_age_HD['HeartDisease']==1]
Non_HD_count  = grouped_data_age_HD[grouped_data_age_HD['HeartDisease']==0]
```

#### Figure to show the frequency of Individuals with Heart Disease by `Age`


```python
colors =['lightblue']
plt.figure(figsize=(8, 6))
plt.bar(HD_count['Age'],HD_count['HD_count'],width=0.9, edgecolor='black',color = colors)
plt.title('Count of Individuals with Heart Disease by Age')
plt.xlabel('Age')
plt.ylabel('Heart_Disease count')
plt.xticks(grouped_data_age_HD['Age'])
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_38_0.png)
    


#### Figure to show the frequency of Individuals without Heart Disease by `Age`


```python
colors =['darkorange']
plt.figure(figsize=(8, 6))
plt.bar(Non_HD_count['Age'],Non_HD_count['HD_count'],width=0.9, edgecolor='black',color = colors)
plt.title('Count of Individuals with No Heart Disease by Age')
plt.xlabel('Age')
plt.ylabel('No Heart_Disease Diagnosis count')
plt.xticks(grouped_data_age_HD['Age'])
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_40_0.png)
    


#### Fasting Blood Sugar levels of Individuals with and without Heart Disease


```python
# 'FastingBS' in the dataframe:There are 170 patients with heart disease who have blood sugar levels above 120 mg
# 1 represents blood sugar above 120mg & 0 represents blood sugar below 120mg

grouped_data_heartFS = data.groupby(['FastingBS','HeartDisease']).agg(count=('HeartDisease', 'count')).reset_index()
print(f'''Fasting Blood Sugar levels of Individuals with and without Heart Disease:\n{grouped_data_heartFS[grouped_data_heartFS['FastingBS']==1][['FastingBS','HeartDisease','count']].sort_values(by ='count',ascending=False)}''')

```

    Fasting Blood Sugar levels of Individuals with and without Heart Disease:
       FastingBS  HeartDisease  count
    3          1             1    170
    2          1             0     44


#### `FastingBS` feature analysis:


```python
# Minimum blood sugar can't be zero but the blood sugar values has been converted to binary format using the condition:
# 1 represents blood sugar above 120mg & 0 represents blood sugar below 120mg

print(data['FastingBS'].describe())
```

    count    918.000000
    mean       0.233115
    std        0.423046
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000000
    max        1.000000
    Name: FastingBS, dtype: float64


#### `Cholesterol` feature analysis:


```python
# Body cannot have zero Cholesterol - Min value shows as zero
data['Cholesterol'].describe()
```




    count    918.000000
    mean     198.799564
    std      109.384145
    min        0.000000
    25%      173.250000
    50%      223.000000
    75%      267.000000
    max      603.000000
    Name: Cholesterol, dtype: float64



#### Cholestrol has a negative relation with Heart Disease


```python
# Correlation of Cholestrol with Target Heart Disease

print(f'Correlation of cholestrol with Heart Disease:{data['Cholesterol'].corr(data['HeartDisease'])}')
```

    Correlation of cholestrol with Heart Disease:-0.23274063892701105


### Distribution of Cholesterol
+ The distribution is heavily skewed to the right due to the presence of outliers


```python
# Distribution of Cholesterol field is skewed to the right
data['Cholesterol'].hist(figsize=(10,6),color='skyblue', edgecolor='black')
plt.title('Cholesterol Distribution')  
plt.xlabel('Cholesterol Level')         # X-axis label
plt.ylabel('Frequency')                  # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```


    
![png](output_50_0.png)
    



```python
# A skewness value close to 0 indicates a symmetric distribution, 
# while values greater than 1 or less than -1 indicate high skewness

skewness = data['Cholesterol'].skew()
print(f'Skewness: {skewness}')
```

    Skewness: -0.6100864307268192


#### Cholesterol with zero observations 


```python
# There were 172 observations with Cholesterol as 0

print(f"Data with Minimum Cholesterol level as 0:{data[data['Cholesterol']== 0]['Age'].count()}")
```

    Data with Minimum Cholesterol level as 0:172


#### Presence of Outliers - Rare Phenomena in Cholesterol Levels


```python
# Individuals with cholestrol level between 350 and 400

data[(data['Cholesterol']>350)&(data['Cholesterol']<400)]['Sex'].value_counts()
```




    Sex
    M    8
    F    6
    Name: count, dtype: int64




```python
# Individuals with cholestrol level between 0 and 150

data[(data['Cholesterol']>0)&(data['Cholesterol']<150)]['Sex'].value_counts()
```




    Sex
    M    18
    F     2
    Name: count, dtype: int64



### Grouping using 'HeartDisease' and replacing missing 'Cholesterol' values with only Median values 


```python
Cholesterol_median_HD = data.groupby('HeartDisease').agg(median_chl =('Cholesterol','median'))

print(Cholesterol_median_HD)
```

                  median_chl
    HeartDisease            
    0                  227.0
    1                  217.0



```python
# Function to replace the missing cholestrol levels

def Chloes1(x):
    HD =x['HeartDisease']
    CH =x['Cholesterol']
    if CH == 0:
        return Cholesterol_median_HD.loc[HD, 'median_chl']
    else:
        return CH
        
data['Cholesterol']=data.apply(Chloes1,axis =1)
```


```python
# Now the minimum value cholestrol is 85 instead of 0

data['Cholesterol'].describe()
```




    count    918.000000
    mean     239.675381
    std       54.328249
    min       85.000000
    25%      214.000000
    50%      225.000000
    75%      267.000000
    max      603.000000
    Name: Cholesterol, dtype: float64



#### Removing outliers from `Cholestrol` feature for better analysis


```python
data = data[(data['Cholesterol']>= 150) & (data['Cholesterol']<= 300)]
data['Cholesterol'].describe()
```




    count    795.000000
    mean     228.657862
    std       32.381871
    min      152.000000
    25%      213.000000
    50%      220.000000
    75%      252.000000
    max      300.000000
    Name: Cholesterol, dtype: float64




```python
# Distribution of Cholestrol levels by Heart Disease

import matplotlib.pyplot as plt
data.boxplot(column= 'Cholesterol',by ='HeartDisease',grid=True, figsize=(8,6))
plt.suptitle('')
plt.title('Box Plot of Cholesterol HD')
plt.xlabel('Heart Disease Status')
plt.ylabel('Cholestrol')
plt.show()
```


    
![png](output_63_0.png)
    


###  Oldpeak Analysis:
+  Numeric value measured in depression


```python
#`Oldpeak` levels of Individuals with `Age`
Oldpeak = data.groupby(['Age']).agg(dep_mean=('Oldpeak','median')).reset_index()
```

### Distribution of Oldpeak levels by Age

## Findings: 
+ The level of depression is very high in individuals over the age of 70, high in those over 50, and moderate in those over 30


```python
# Importing the Matplotlib library to plot the charts

import matplotlib.pyplot as plt
Oldpeak.plot(kind = 'bar',x = 'Age', y = 'dep_mean', figsize=(8,6))
plt.title('Distribution of Oldpeak levels by Age')
plt.show()
```


    
![png](output_67_0.png)
    


#### Resting BP Analysis
+ The feature has 0 as its minimum value, which is not feasible for the human body


```python
# Resting BP - No possibility for the presence of 0  - Min value
data['RestingBP'].describe()
```




    count    795.000000
    mean     131.877987
    std       18.597458
    min        0.000000
    25%      120.000000
    50%      130.000000
    75%      140.000000
    max      200.000000
    Name: RestingBP, dtype: float64




```python
# Observations with Minimum Cholesterol level as 0 
# There are 1 observations with RestingBP as 0

data[data['RestingBP'] == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>449</th>
      <td>55</td>
      <td>M</td>
      <td>NAP</td>
      <td>0</td>
      <td>217.0</td>
      <td>0</td>
      <td>Normal</td>
      <td>155</td>
      <td>N</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Dropping the index: 449 as it does not have meaningful data


```python
### Dropping the index: 449 as it does not have meaningful data in other fields as well
data.drop(index = 449,inplace = True)
```

    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_11638/536921324.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data.drop(index = 449,inplace = True)



```python
# There are 0 observations with RestingBP as 0 post dropping the index

print(f"Data with RestingBP 0:{data[data['RestingBP']== 0]['Age'].count()}")
```

    Data with RestingBP 0:0



```python
print(f'''Data with and without Heart Disease:\n {data['HeartDisease'].value_counts()}''')
```

    Data with and without Heart Disease:
     HeartDisease
    1    442
    0    352
    Name: count, dtype: int64


### Correlation analysis - Post replacing 'Cholesterol' values with median values


```python
# High correlation might indicate that skewed distributions are affecting the outcome
# Correlation with the Target - Heart disease

cols =['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak','HeartDisease']
     
correlation = data[cols].apply(lambda x:x.corr(data['HeartDisease']))
print(correlation)
```

    Age             0.304324
    RestingBP       0.124875
    Cholesterol     0.036623
    FastingBS       0.284905
    MaxHR          -0.388404
    Oldpeak         0.392248
    HeartDisease    1.000000
    dtype: float64


#### Maximum Heart Rate Analysis
+ A minimum of 60 and a maximum of 202 have been recorded in terms of heart rate
+ The distribution is skewed to the left


```python
# Heart rate - Minimum of 60 and Maximum of 202 recorded
print(data['MaxHR'].describe())
```

    count    794.000000
    mean     136.463476
    std       25.745908
    min       60.000000
    25%      118.250000
    50%      138.000000
    75%      155.000000
    max      202.000000
    Name: MaxHR, dtype: float64


### Median Heart Rate with & without Heart Disease


```python
# Grouping Individuals with Heart Disease and their Heart rate
# With Heart Disease (1), Without Heart Disease (0)

MaxHR  = data.groupby('HeartDisease').agg(median_chl =('MaxHR','median'))
print(MaxHR)
```

                  median_chl
    HeartDisease            
    0                  150.0
    1                  125.0


### Distribution of `MaxHR` field


```python
# Distribution of 'MaxHR' field is slightly skewed to the left
data['MaxHR'].hist(figsize=(10,6),color='lightgrey', edgecolor='black')
plt.title('Maximum Heart Rate Distribution')  
plt.xlabel('MaxHR Level')         # X-axis label
plt.ylabel('Frequency')           # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```


    
![png](output_82_0.png)
    


#### Distribution of data shown in a box plot, including outliers


```python
# Importing librariers to plot the chart
import matplotlib.pyplot as plt
data.boxplot(column= 'MaxHR',by ='HeartDisease',grid=True, figsize=(8,6))
plt.suptitle('')
plt.title('Box Plot of MaxHR by HD Category')
plt.xlabel('Heart Disease')                     # X-axis label
plt.ylabel('MaxHR Level')          # Y-axis label
plt.show()
```


    
![png](output_84_0.png)
    



```python
# skewness rate
skewness = data['MaxHR'].skew()
print(f'Skewness: {skewness}')
```

    Skewness: -0.12734070973397935


### Features to choose for our model:


+ I am chosing the below features to design my model. Because they are strongly correlating with my target variable (`Heart Disease`) and I believe that they add value to my model

    + `Oldpeak`                  
    + `MaxHR`
    + `ST_Slope_Flat`
    + `ST_Slope_Up`
    + `ExerciseAngina_Y`
    + `ChestPainType_ATA`

### Assigining dummy variables to the categorical column to include them in the Analysis:


```python
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 794 entries, 0 to 917
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Age             794 non-null    int64  
     1   Sex             794 non-null    object 
     2   ChestPainType   794 non-null    object 
     3   RestingBP       794 non-null    int64  
     4   Cholesterol     794 non-null    float64
     5   FastingBS       794 non-null    int64  
     6   RestingECG      794 non-null    object 
     7   MaxHR           794 non-null    int64  
     8   ExerciseAngina  794 non-null    object 
     9   Oldpeak         794 non-null    float64
     10  ST_Slope        794 non-null    object 
     11  HeartDisease    794 non-null    int64  
    dtypes: float64(2), int64(5), object(5)
    memory usage: 80.6+ KB
    None


## Using One hot encoding


```python
# Using one hot encoding replacing categorical values with dummy variables

# copying the orginal dataframe
dat = data.copy()

# Getting dummy variables
dat = pd.get_dummies(data = dat, columns =['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],drop_first = True)

# printing the new dataframe columns to view the changes
print(dat.columns)
```

    Index(['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
           'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
           'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
           'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'],
          dtype='object')


### Using the Pearson correlation coefficient to have range between -1 & 1


```python
correl_matrix = dat.corr()

print(correl_matrix['HeartDisease'])
```

    Age                  0.304324
    RestingBP            0.124875
    Cholesterol          0.036623
    FastingBS            0.284905
    MaxHR               -0.388404
    Oldpeak              0.392248
    HeartDisease         1.000000
    Sex_M                0.310463
    ChestPainType_ATA   -0.403482
    ChestPainType_NAP   -0.219577
    ChestPainType_TA    -0.049795
    RestingECG_Normal   -0.087840
    RestingECG_ST        0.100694
    ExerciseAngina_Y     0.476976
    ST_Slope_Flat        0.548192
    ST_Slope_Up         -0.621460
    Name: HeartDisease, dtype: float64


### Taking absolute value of the Pearson correlation coefficient


```python
correlation_mat = abs(dat.corr())

print(correlation_mat['HeartDisease'].sort_values())
```

    Cholesterol          0.036623
    ChestPainType_TA     0.049795
    RestingECG_Normal    0.087840
    RestingECG_ST        0.100694
    RestingBP            0.124875
    ChestPainType_NAP    0.219577
    FastingBS            0.284905
    Age                  0.304324
    Sex_M                0.310463
    MaxHR                0.388404
    Oldpeak              0.392248
    ChestPainType_ATA    0.403482
    ExerciseAngina_Y     0.476976
    ST_Slope_Flat        0.548192
    ST_Slope_Up          0.621460
    HeartDisease         1.000000
    Name: HeartDisease, dtype: float64


### Using the square of the Pearson correlation coefficient to have range between 0 & 1


```python
correlation_matrix = dat.corr()

correlation_matrix = correlation_matrix ** 2

print(correlation_matrix['HeartDisease'])
```

    Age                  0.092613
    RestingBP            0.015594
    Cholesterol          0.001341
    FastingBS            0.081171
    MaxHR                0.150858
    Oldpeak              0.153858
    HeartDisease         1.000000
    Sex_M                0.096387
    ChestPainType_ATA    0.162798
    ChestPainType_NAP    0.048214
    ChestPainType_TA     0.002480
    RestingECG_Normal    0.007716
    RestingECG_ST        0.010139
    ExerciseAngina_Y     0.227506
    ST_Slope_Flat        0.300514
    ST_Slope_Up          0.386213
    Name: HeartDisease, dtype: float64


### Heatmap - Correlation Matrix


```python
# Importing the packages for the heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='Reds',cbar=True)


# Add titles and labels
plt.title('Heatmap correlation values')
plt.xlabel('features')
plt.ylabel('features')

# Show the plot
plt.show()
```


    
![png](output_99_0.png)
    


### Highlighting feature with Correlation above 0.15


```python
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix[correlation_matrix > 0.15], annot=True, cmap="Reds")
plt.show()
```


    
![png](output_101_0.png)
    


### Segmenting the data and the target for ML models


```python
X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```

## Model 1 - KNeighborsClassifier

### Findings:

+ Out of all selected features, `ST_Slope_Up` performed well with a 84.28% accuracy based on numerical data


+ Under numerical features, the high performers are:
    + Age: Accuracy 67.30%
    + MaxHR: Accuracy 66.04%
     
+ Meanwhile under categorical features, the high performers are:
    + ST_Slope_Flat: Accuracy:82.39%, 
    + ST_Slope_Up: Accuracy:82.61%, 
    + ExerciseAngina_Y: Accuracy:74.21%

     
+ By using combined features like `FastingBS`,`Oldpeak`,`ST_Slope_Flat`,`ST_Slope_Up` & `ExerciseAngina_Y`, I achieved an accuracy of 90% on the validation set and 85% on the test set using KNN model

### Single feature model


```python
# Numerical Features to be used in the single feature model
features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
```

### Splitting the dataset into Training and Validation set


```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X[features],y,test_size = .20,random_state = 500)
```

### Scaling the Training and Validation Datasets for Numerical Data to Calculate Accuracy


```python
# Importing the libraries 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

scaler = MinMaxScaler()

X_train_scale = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)

X_val_scale = pd.DataFrame(scaler.fit_transform(X_val),columns=X_train.columns)

# Loop through each feature index
for i in features:
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scale[[i]],y_train)
    accuracy = model.score(X_val_scale[[i]],y_val)
    # Print accuracy results
    print(f'Feature {i}: Accuracy {accuracy*100:.2f}%')
```

    Feature Age: Accuracy 67.30%
    Feature RestingBP: Accuracy 57.23%
    Feature FastingBS: Accuracy 61.01%
    Feature Cholesterol: Accuracy 36.48%
    Feature Oldpeak: Accuracy 43.40%
    Feature MaxHR: Accuracy 66.04%


### Categorical data with dummy variables and their accuracy:


```python
# Segmenting the data and the Target
X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```


```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = .20,random_state = 500)
```


```python
# Categorical Features to be used in the single feature model
feature1 =['Sex_M','ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA']
```


```python
for i in feature1:
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train[[i]],y_train)
    accuracy = model.score(X_val[[i]],y_val)
    print(f' Feature {i}: Accuracy:{accuracy*100:.2f}%')
```

     Feature Sex_M: Accuracy:64.78%
     Feature ST_Slope_Flat: Accuracy:82.39%
     Feature ST_Slope_Up: Accuracy:84.28%
     Feature ExerciseAngina_Y: Accuracy:74.21%
     Feature ChestPainType_ATA: Accuracy:45.28%


### Multi feature model:


```python
# Segmenting the data and the Target

X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```


```python
# Combined features
feature3 =['FastingBS','Oldpeak','ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y'] 
```

### Data split:
    + Training set - 60%
    + Validation set - 20%
    + Test set - 20%


```python
# Splitting the dataset into Training and Validation set:
X_train, X_val, y_train, y_val = train_test_split(X[feature3],y,test_size = .20,random_state = 500)
```


```python
# Observations Train and validation set
print(f'Observations in Train set:{X_train.shape[0]}\n',f'Observations in the Validation set:{X_val.shape[0]}')
```

    Observations in Train set:635
     Observations in the Validation set:159



```python
# Splitting the dataset into Training and test set from the training set:

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size = .25,random_state = 500)
```


```python
# Observations in the Train and Test set

print(f'Observations in Train set:{X_train.shape[0]}\n',f'Observations in Test set:{X_test.shape[0]}')
```

    Observations in Train set:476
     Observations in Test set:159



```python
print(X_train.columns)
```

    Index(['FastingBS', 'Oldpeak', 'ST_Slope_Flat', 'ST_Slope_Up',
           'ExerciseAngina_Y'],
          dtype='object')


### Fitting and transforming the numerical fields to have the range between 0 and 1


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Using the categorical data as it is for the training set
X_train_sca = X_train

# Fit transforming the numerical data for the training set
X_train_sca[['Oldpeak','FastingBS']] = scaler.fit_transform(X_train[['Oldpeak','FastingBS']])


# Using the categorical data as it is for the Validation set
X_val_sca = X_val

# Fit transforming the numerical data for the Validation set
X_val_sca[['Oldpeak','FastingBS']] = scaler.transform(X_val[['Oldpeak','FastingBS']])
```

### Using hyperparameters to tune the model
+ Using GridsearchCV to find the ideal parameters that could yield the highest accuracy


```python
dic ={'n_neighbors':[1,2,3,4,5,10,20,30,40,50],'weights':['uniform','distance'],'p':[1,2,3,4,5]}

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
GridSearchCV =GridSearchCV(estimator = model,param_grid =dic ,scoring ='accuracy')

GridSearchCV.fit(X_train_sca,y_train)

GridSearchCV.score(X_val_sca,y_val)

best_params = GridSearchCV.best_params_
best_score = GridSearchCV.best_score_

print(best_params,best_score)
```

    {'n_neighbors': 5, 'p': 1, 'weights': 'uniform'} 0.8214692982456141


    /Users/nandhinimuthalraj/Documents/anaconda/anaconda3/lib/python3.12/site-packages/numpy/ma/core.py:2820: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,


#### KNN classifier


```python
# creating a KNN classifier & validating them on the Validation dataset: 

model = KNeighborsClassifier(n_neighbors= 5, p = 1, weights = 'uniform')
model.fit(X_train_sca,y_train)
accuracy = model.score(X_val_sca,y_val)
accuracy = accuracy*100
print(f'Accuracy of the validation data set:{accuracy:.0f}%')
```

    Accuracy of the validation data set:90%


### Test set Accuracy:


```python
# Using the categorical data as it is for the Validation set
X_test_sca = X_test

# Fit transforming the numerical data for the training set
X_test_sca[['Oldpeak','FastingBS']] = scaler.transform(X_test[['Oldpeak','FastingBS']])
```


```python
# creating a KNN classifier & validating them on the Validation dataset: 
model = KNeighborsClassifier(n_neighbors= 10, p = 1, weights = 'uniform')
model.fit(X_train_sca,y_train)
accuracy = model.score(X_test_sca,y_test)
accuracy = accuracy*100
print(f'Accuracy of the Test Set:{accuracy:.0f}%')
```

    Accuracy of the Test Set:85%


## Model 2 - Linear SVC model 

## Findings:
+ By using combined features like `FastingBS`,`Oldpeak`,`ST_Slope_Flat`,`ST_Slope_Up` & `ExerciseAngina_Y`, I achieved an accuracy of 84% on the validation set and 82% on the test set
+ As the KNN classifier is based on the nearest neighbours there might be a difference of margin classification compared to Linera SVC


```python
from sklearn.svm import LinearSVC

model = LinearSVC(penalty="l2",loss="squared_hinge",C=10,random_state=417)
```


```python
model.fit(X_train_sca,y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearSVC(C=10, random_state=417)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearSVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html">?<span>Documentation for LinearSVC</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearSVC(C=10, random_state=417)</pre></div> </div></div></div></div>




```python
val_accuracy = model.score(X_val_sca,y_val)
print(f'Accuracy of the Validation dataset via SVC Model:{val_accuracy*100:.0f}%')
```

    Accuracy of the Validation dataset via SVC Model:84%



```python
test_accuracy = model.score(X_test_sca,y_test)
print(f'Accuracy of the Test dataset via SVC Model:{test_accuracy*100:.0f}%')
```

    Accuracy of the Test dataset via SVC Model:82%


## Model 3 - Logistic Regression Model

### Findings:

+ I have used the following features to obtain the model accuracy, which are highly correlated with my target variable, 'HeartDisease':
   +  `FastingBS`
   +  `Oldpeak`
   +  `ST_Slope_Flat`
   +  `ExerciseAngina_Y`
 
+ The count of non-cases (368) exceeds the count of cases (331), indicating that there is more data on individuals without heart disease than on those with heart disease
+ The samples used in both the validation and test sets are balanced between the two classes (0's and 1's)
+ Different sampling methods resulted in varying accuracy levels on both the validation and test sets
+ Numerical features perform better for non-cases compared to cases
+ Categorical features, on the other hand, achieve higher accuracy with cases (80% correct predictions) as well as with non-cases (70% correct predictions)
+ By using the right combination of numerical and categorical features, along with effective sampling, I achieved an accuracy score of 84% on the training set, 81% on the validation set, and 85% on the test set


```python
X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```


```python
# Target 
print(f'Data with and without Heart Disease:\n{y.value_counts()}')
```

    Data with and without Heart Disease:
    HeartDisease
    1    442
    0    352
    Name: count, dtype: int64


### We'll use a 60-20 split of the dataset for the training and test sets.


```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.20,random_state = 500)

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size = 0.25,random_state = 500)
```

### Metrics used for evaluation of the model
+ ### Accuracy Score:
Accuracy = Correct predictions / Total predictions 

+ ### Sensitivity_score:
Sensitivity = True Positives / (True Positives + False Negatives)

+ ### Specificity_score 
Specificity = True Negatives / (True Negatives + False Positives)

## Numerical Features

## Findings:

#### Model using numerical features:

+ The cases (identifying individuals with heart disease) with certain features like Age, Oldpeak and MaxHR have accurate predictions above 50%
    + Oldpeak: Accuracy 75.47%
    + MaxHR: Accuracy 66.67%
    + Age: Accuracy 65.41%

+ The non-cases(identifying individuals without heart disease) have predictions above 65% for almost all features
+ The model performed better at identifying individuals without heart disease compared to those with the condition
+ All numerical features have positive coefficients, except for MaxHR and Cholesterol, which have coefficients of -3.120 and -0.019, respectively. This indicates that for every one-unit increase in MaxHR and Cholesterol, the response variable decreases by 3.120 units and 0.019 units
+ The test set performed similarly to the training set in terms of accuracy, specificity & sensitivity, and it did performed well on cases (Identifying individuals with heart disease)
+ Having balanced sensitivity and specificity can contribute to a higher accuracy score


```python
features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
```


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

scaler = MinMaxScaler()

X_sc_t = scaler.fit_transform(X_train)
X_sc_v = scaler.transform(X_val)
X_sc_te = scaler.transform(X_test)

# converting scaled data back into dataframe
X_sc_t_df = pd.DataFrame(X_sc_t, columns=X_train.columns)
X_sc_v_df = pd.DataFrame(X_sc_v, columns=X_val.columns)
X_sc_te_df = pd.DataFrame(X_sc_te, columns=X_test.columns)
```

### No feature Intercept and Success probability

+ The intercept (-0.35176709) when the features are not included in the model


```python
## Intercept of Model : 
model_l = LogisticRegression()
model_l.fit(X_sc_t_df[features], y_train)
b0 = model_l.intercept_
print(f'Intercept of Model:{b0}')

Z = b0

# Step 2: Calculate the success probability
success_probability = 1 / (1 + np.exp(-Z))    #Manual - z/(1+z)
print(f'Success probs with no predictors:{success_probability[0]}')
```

    Intercept of Model:[-0.35176709]
    Success probs with no predictors:0.41295397312744847


### Model Quality


```python
#Model fitting & Accuracy on validation set:
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    ac = model_l.score(X_sc_v_df[[i]], y_val)
    print(f'Feature {i}: Accuracy {ac * 100:.2f}%')
```

    Feature Age: Accuracy 65.41%
    Feature RestingBP: Accuracy 54.72%
    Feature FastingBS: Accuracy 61.01%
    Feature Cholesterol: Accuracy 54.72%
    Feature Oldpeak: Accuracy 75.47%
    Feature MaxHR: Accuracy 66.67%


### Slope co-efficients
+ With each individual feature, the variation in the slope


```python
# Slope co-efficients of Numerical features
# Age, RestingBP, FastingBS, Cholesterol and Oldpeak has positive coefficients
# MaxHR has negative coefficients

for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    slope_coff = model_l.coef_[0,0]
    log_odds = np.exp(slope_coff)
    print(f'Slope feature {i}:{slope_coff:.3f}')
    print(f'Log_odds {i}:{log_odds:.3f}')
```

    Slope feature Age:2.343
    Log_odds Age:10.410
    Slope feature RestingBP:1.005
    Log_odds RestingBP:2.731
    Slope feature FastingBS:1.482
    Log_odds FastingBS:4.400
    Slope feature Cholesterol:-0.019
    Log_odds Cholesterol:0.981
    Slope feature Oldpeak:3.971
    Log_odds Oldpeak:53.013
    Slope feature MaxHR:-3.435
    Log_odds MaxHR:0.032



```python
# Multi variate feature 
# The coefficients in this context reflect the relationship of each feature in the presence of the others.

mode = LogisticRegression()
mode.fit(X_sc_t_df[features], y_train)

coefs = ['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']

# Checking in terms of log-odds
for coef, val  in zip(coefs, mode.coef_[0]):
    print(coef, ":", round(val, 2))
```

    Age : 0.66
    RestingBP : 0.29
    FastingBS : 1.39
    Cholesterol : 0.06
    Oldpeak : 3.59
    MaxHR : -3.0


### To plot the odds against the probability for Cholestrol (linear realtion)


```python
# Using the logistic regression form :  Z=β0+β1X & EY=h(Z)

beta0 = -0.3555845
beta1 = -0.019

# Linear combination of predictors
dat['z'] = beta0 + dat['Cholesterol']*beta1

# Range between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

dat['hz'] = dat['z'].apply(sigmoid)

import matplotlib.pyplot as plt
dat.plot.scatter(x = 'z',y ='hz')
plt.show()
```


    
![png](output_160_0.png)
    


#### Various Models with Different Combinations of Predictors and Their Accuracies


```python
# Model fitting features:

X1 = X_train[['Oldpeak']]
X2 = X_train[['Oldpeak','MaxHR']]
X3 = X_train[['Oldpeak','MaxHR','Age']]
X4 = X_train[['Oldpeak','MaxHR','Age','FastingBS']]
X5 = X_train[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol']]
X6 = X_train[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol','RestingBP']]

X1_val = X_val[['Oldpeak']]
X2_val = X_val[['Oldpeak','MaxHR']]
X3_val = X_val[['Oldpeak','MaxHR','Age']]
X4_val = X_val[['Oldpeak','MaxHR','Age','FastingBS']]
X5_val = X_val[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol']]
X6_val = X_val[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol','RestingBP']]


X1_test = X_test[['Oldpeak']]
X2_test = X_test[['Oldpeak','MaxHR']]
X3_test = X_test[['Oldpeak','MaxHR','Age']]
X4_test = X_test[['Oldpeak','MaxHR','Age','FastingBS']]
X5_test = X_test[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol']]
X6_test = X_test[['Oldpeak','MaxHR','Age','FastingBS','Cholesterol','RestingBP']]


model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()
model4 = LogisticRegression()
model5 = LogisticRegression()
model6 = LogisticRegression()


model1.fit(X1, y_train)
model2.fit(X2, y_train)
model3.fit(X3, y_train)
model4.fit(X4, y_train)
model5.fit(X5, y_train)
model6.fit(X6, y_train)


train_accuracies = model1.score(X1,y_train),model2.score(X2,y_train),model3.score(X3,y_train),model4.score(X4,y_train),model5.score(X5,y_train),model6.score(X6,y_train)
print(f'Training Accuracy of models:{np.array(train_accuracies)}')

val_accuracies = model1.score(X1_val,y_val),model2.score(X2_val,y_val),model3.score(X3_val,y_val),model4.score(X4_val,y_val),model5.score(X5_val,y_val),model6.score(X6_val,y_val)
print(f'Validation Accuracy of models:{np.array(val_accuracies)}')
```

    /Users/nandhinimuthalraj/Documents/anaconda/anaconda3/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    Training Accuracy of models:[0.69747899 0.71638655 0.71218487 0.7394958  0.7394958  0.74369748]
    Validation Accuracy of models:[0.75471698 0.77987421 0.77987421 0.7672956  0.77987421 0.77987421]



```python
Test_accuracies = model1.score(X1_test,y_test),model2.score(X2_test,y_test),model3.score(X3_test,y_test),model4.score(X4_test,y_test),model5.score(X5_test,y_test),model6.score(X6_test,y_test)
print(f'Test Accuracy of models:{np.array(Test_accuracies)}')
```

    Test Accuracy of models:[0.67295597 0.7672956  0.76100629 0.77987421 0.78616352 0.77987421]



```python
#% of correct prediction for individuals with Heart Disease

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_t_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tp = sum((y_train == 1) & (predict == 1))
    fn = sum((y_train == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score:{i}:{sens:.2f}')
```

    sensitivity_score:Age:0.81
    sensitivity_score:RestingBP:0.95
    sensitivity_score:FastingBS:0.36
    sensitivity_score:Cholesterol:1.00
    sensitivity_score:Oldpeak:0.70
    sensitivity_score:MaxHR:0.76



```python
#% of correct prediction for individuals without Heart Disease

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']    
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_t_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tn = sum((y_train == 0) & (predict == 0))
    fp = sum((y_train == 0) & (predict == 1))
    spec = tn / ( tn + fp)
    print(f'specificity_score:{i}:{spec:.2f}')
```

    specificity_score:Age:0.42
    specificity_score:RestingBP:0.06
    specificity_score:FastingBS:0.90
    specificity_score:Cholesterol:0.00
    specificity_score:Oldpeak:0.70
    specificity_score:MaxHR:0.53


### Validation set


```python
# How well the validation set is classified:
# Class 1 (with heart disease)  

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_v_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tp = sum((y_val == 1) & (predict == 1))
    fn = sum((y_val == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score:{i}:{sens:.2f}')
```

    sensitivity_score:Age:0.86
    sensitivity_score:RestingBP:0.95
    sensitivity_score:FastingBS:0.39
    sensitivity_score:Cholesterol:1.00
    sensitivity_score:Oldpeak:0.75
    sensitivity_score:MaxHR:0.84



```python
# How well the validation set is classified:
# Class 0 (without heart disease) 

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_v_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tn = sum((y_val == 0) & (predict == 0))
    fp = sum((y_val == 0) & (predict == 1))
    spec = tn / ( tn + fp)
    print(f'specificity_score:{i}:{spec:.2f}')
```

    specificity_score:Age:0.40
    specificity_score:RestingBP:0.06
    specificity_score:FastingBS:0.88
    specificity_score:Cholesterol:0.00
    specificity_score:Oldpeak:0.76
    specificity_score:MaxHR:0.46


### Test Set


```python
# How well the test set is classified:
# Class 1 (with heart disease)  

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_te_df[[i]]) # No need to iterate already a 1-D array
    tp = sum((y_test == 1) & (predict == 1))
    fn = sum((y_test == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score_test:{i}:{sens:.2f}')
```

    sensitivity_score_test:Age:0.74
    sensitivity_score_test:RestingBP:0.97
    sensitivity_score_test:FastingBS:0.29
    sensitivity_score_test:Cholesterol:1.00
    sensitivity_score_test:Oldpeak:0.66
    sensitivity_score_test:MaxHR:0.75



```python
# How well the test set is classified:
# Class 0 (without heart disease) 

features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_te_df[[i]]) # No need to iterate already a 1-D array
    fp = sum((y_test == 0) & (predict == 1))
    tn = sum((y_test == 0) & (predict == 0))
    spec = fp / ( fp + tn)
    print(f'Specificity_score_test:{i}:{spec:.2f}')
```

    Specificity_score_test:Age:0.49
    Specificity_score_test:RestingBP:0.96
    Specificity_score_test:FastingBS:0.09
    Specificity_score_test:Cholesterol:1.00
    Specificity_score_test:Oldpeak:0.35
    Specificity_score_test:MaxHR:0.32


## Categorical Features

## Findings:

#### Model using categorical features:

+ The logistic regression model achieved correct predictions for all the categorical features at a rate exceeding 80%

+ The cases with individual features with correct predictions above 80%
+ The non-cases have features with correct predictions above 70%
+ The model performs well on both predicting individuals with and without heart disease for certain categorical features
+ ST_Slope_Flat and ExerciseAngina_Y, which have coefficients of 2.261 and 2.112, indicate that for every one-unit increase in ST_Slope_Flat and ExerciseAngina_Y, the response variable increases by 2.261 and 2.112 units, respectively
+ ST_Slope_Up, ChestPainType_ATA, and ChestPainType_NAP, which have coefficients of -2.747, -2.030, and -0.911, indicate that for every one-unit increase in ST_Slope_Up, ChestPainType_ATA, and ChestPainType_NAP, the response variable decreases by 2.747, 2.030, and 0.911 units, respectively
+ The test set performs similarly to the training set for cases, but they have opposite effects on non-cases. Certain features that performed well on the training set did not perform well on the test set for non-cases


```python
feature1 =['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']
```


```python
X_sc_t_df = pd.DataFrame(X_sc_t, columns=X_train.columns)
X_sc_v_df = pd.DataFrame(X_sc_v, columns=X_val.columns)
X_sc_te_df = pd.DataFrame(X_sc_te, columns=X_test.columns)
```


```python
#Model fitting & Accuracy:
for i in feature1:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    ac = model_l.score(X_sc_v_df[[i]], y_val)
    print(f'Feature {i}: Accuracy {ac * 100:.2f}%')
```

    Feature ST_Slope_Flat: Accuracy 82.39%
    Feature ST_Slope_Up: Accuracy 84.28%
    Feature ExerciseAngina_Y: Accuracy 74.21%
    Feature ChestPainType_ATA: Accuracy 69.81%
    Feature ChestPainType_NAP: Accuracy 62.89%



```python
# Slope co-efficients of categorical features
# ST_Slope_Flat and ExerciseAngina_Y has positive coefficients
# ST_Slope_Up, ChestPainType_ATA and ChestPainType_NAP has negative coefficients

for i in feature1:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    slope_coff = model_l.coef_[0,0]
    log_odds = np.exp(slope_coff)
    print(f'Slope feature {i}:{slope_coff:.3f}')
    print(f'Log_odds {i}:{log_odds:.3f}')
```

    Slope feature ST_Slope_Flat:2.261
    Log_odds ST_Slope_Flat:9.594
    Slope feature ST_Slope_Up:-2.747
    Log_odds ST_Slope_Up:0.064
    Slope feature ExerciseAngina_Y:2.112
    Log_odds ExerciseAngina_Y:8.268
    Slope feature ChestPainType_ATA:-2.029
    Log_odds ChestPainType_ATA:0.131
    Slope feature ChestPainType_NAP:-0.911
    Log_odds ChestPainType_NAP:0.402


### Model quality


```python
# Model fitting features:

X1 = X_train[['ST_Slope_Flat']]
X2 = X_train[['ST_Slope_Flat','ST_Slope_Up']]
X3 = X_train[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y']]
X4 = X_train[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA']]
X5 = X_train[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']]


X1_val = X_val[['ST_Slope_Flat']]
X2_val= X_val[['ST_Slope_Flat','ST_Slope_Up']]
X3_val= X_val[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y']]
X4_val = X_val[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA']]
X5_val = X_val[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']]


X1_test = X_test[['ST_Slope_Flat']]
X2_test = X_test[['ST_Slope_Flat','ST_Slope_Up']]
X3_test = X_test[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y']]
X4_test = X_test[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA']]
X5_test = X_test[['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']]




model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()
model4 = LogisticRegression()
model5 = LogisticRegression()


model1.fit(X1, y_train)
model2.fit(X2, y_train)
model3.fit(X3, y_train)
model4.fit(X4, y_train)
model5.fit(X5, y_train)


train_accuracies = model1.score(X1,y_train),model2.score(X2,y_train),model3.score(X3,y_train),model4.score(X4,y_train),model5.score(X5,y_train)
print(f'Training Accuracy of various models:{np.array(train_accuracies)}')

val_accuracies = model1.score(X1_val,y_val),model2.score(X2_val,y_val),model3.score(X3_val,y_val),model4.score(X4_val,y_val),model5.score(X5_val,y_val)
print(f'Test Accuracy of various models:{np.array(val_accuracies)}')
```

    Training Accuracy of various models:[0.7605042  0.81092437 0.81092437 0.82563025 0.83193277]
    Test Accuracy of various models:[0.82389937 0.8427673  0.8427673  0.8490566  0.87421384]



```python
Test_accuracies = model1.score(X1_test,y_test),model2.score(X2_test,y_test),model3.score(X3_test,y_test),model4.score(X4_test,y_test),model5.score(X5_test,y_test)
print(f'Test Accuracy of various models:{np.array(Test_accuracies)}')
```

    Test Accuracy of various models:[0.75471698 0.79245283 0.79245283 0.83018868 0.8427673 ]


### Training set Model quality



```python
#% of correct prediction for individuals with Heart Disease

for i in feature1:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_t_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tp = sum((y_train == 1) & (predict == 1))
    fn = sum((y_train == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score:{i}:{sens:.2f}')
```

    sensitivity_score:ST_Slope_Flat:0.73
    sensitivity_score:ST_Slope_Up:0.84
    sensitivity_score:ExerciseAngina_Y:0.62
    sensitivity_score:ChestPainType_ATA:0.94
    sensitivity_score:ChestPainType_NAP:0.85



```python
# Training set Model quality
#% of correct prediction for individuals without Heart Disease

for i in feature1:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_t_df[[i]]) # No need to iterate already a 1-D array
    y_val = y_val
    tn = sum((y_train == 0) & (predict == 0))
    fp = sum((y_train == 0) & (predict == 1))
    spec = tn / ( tn + fp)
    print(f'specificity_score:{i}:{spec:.2f}')
```

    specificity_score:ST_Slope_Flat:0.80
    specificity_score:ST_Slope_Up:0.77
    specificity_score:ExerciseAngina_Y:0.85
    specificity_score:ChestPainType_ATA:0.35
    specificity_score:ChestPainType_NAP:0.31


### Validation Set Model Quality


```python
# Class with Heart disease

features =['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_v_df[[i]]) # No need to iterate already a 1-D array
    tp = sum((y_test == 1) & (predict == 1))
    fn = sum((y_test == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score_test:{i}:{sens:.2f}')
```

    sensitivity_score_test:ST_Slope_Flat:0.48
    sensitivity_score_test:ST_Slope_Up:0.56
    sensitivity_score_test:ExerciseAngina_Y:0.37
    sensitivity_score_test:ChestPainType_ATA:0.82
    sensitivity_score_test:ChestPainType_NAP:0.82



```python
# Class without Heart disease

features =['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_v_df[[i]]) # No need to iterate already a 1-D array
    fp = sum((y_test == 0) & (predict == 1))
    tn = sum((y_test == 0) & (predict == 0))
    spec = fp / ( fp + tn)
    print(f'Specificity_score_test:{i}:{spec:.2f}')
```

    Specificity_score_test:ST_Slope_Flat:0.54
    Specificity_score_test:ST_Slope_Up:0.57
    Specificity_score_test:ExerciseAngina_Y:0.35
    Specificity_score_test:ChestPainType_ATA:0.82
    Specificity_score_test:ChestPainType_NAP:0.69


### Test Set Model Quality


```python
# Class with Heart disease

features =['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_te_df[[i]]) # No need to iterate already a 1-D array
    tp = sum((y_test == 1) & (predict == 1))
    fn = sum((y_test == 1) & (predict == 0))
    sens = tp / ( tp + fn)
    print(f'sensitivity_score_test:{i}:{sens:.2f}')
```

    sensitivity_score_test:ST_Slope_Flat:0.71
    sensitivity_score_test:ST_Slope_Up:0.81
    sensitivity_score_test:ExerciseAngina_Y:0.57
    sensitivity_score_test:ChestPainType_ATA:0.95
    sensitivity_score_test:ChestPainType_NAP:0.88



```python
# Class without Heart disease

features =['ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y','ChestPainType_ATA','ChestPainType_NAP']
for i in features:
    model_l = LogisticRegression()
    model_l.fit(X_sc_t_df[[i]], y_train)
    predict = model_l.predict(X_sc_te_df[[i]]) # No need to iterate already a 1-D array
    fp = sum((y_test == 0) & (predict == 1))
    tn = sum((y_test == 0) & (predict == 0))
    spec = fp / ( fp + tn)
    print(f'Specificity_score_test:{i}:{spec:.2f}')
```

    Specificity_score_test:ST_Slope_Flat:0.19
    Specificity_score_test:ST_Slope_Up:0.24
    Specificity_score_test:ExerciseAngina_Y:0.15
    Specificity_score_test:ChestPainType_ATA:0.57
    Specificity_score_test:ChestPainType_NAP:0.66


### Combined features - Numerical and Categorical


```python
feature_c = ['RestingBP','Age','FastingBS','Oldpeak','ST_Slope_Flat','ExerciseAngina_Y','ST_Slope_Up','ChestPainType_ATA']
```


```python
X_sc_t = X_train[feature_c] 
X_sc_v = X_val[feature_c]
X_sc_te = X_test[feature_c]
```


```python
# Using scaler function to fit transform training, val and test dataset with numerical features
scaler = MinMaxScaler()

X_sc_t[['RestingBP','Oldpeak','RestingBP','Age']] = scaler.fit_transform(X_train[['RestingBP','Oldpeak','RestingBP','Age']])
X_sc_v[['RestingBP','Oldpeak','RestingBP','Age']] = scaler.transform(X_val[['RestingBP','Oldpeak','RestingBP','Age']])
X_sc_te [['RestingBP','Oldpeak','RestingBP','Age']]= scaler.transform(X_test[['RestingBP','Oldpeak','RestingBP','Age']])
```

    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_11638/3256982120.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_sc_t[['RestingBP','Oldpeak','RestingBP','Age']] = scaler.fit_transform(X_train[['RestingBP','Oldpeak','RestingBP','Age']])
    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_11638/3256982120.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_sc_v[['RestingBP','Oldpeak','RestingBP','Age']] = scaler.transform(X_val[['RestingBP','Oldpeak','RestingBP','Age']])
    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_11638/3256982120.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_sc_te [['RestingBP','Oldpeak','RestingBP','Age']]= scaler.transform(X_test[['RestingBP','Oldpeak','RestingBP','Age']])


### Training set Model quality


```python
# Accuracy score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
accuracy = model_l.score(X_sc_t,y_train)
print(f'Accuracy_score:{accuracy:.2f}')
```

    Accuracy_score:0.83



```python
# Specificity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_t) 
fp = sum((y_train == 0) & (predict == 1))
tn = sum((y_train == 0) & (predict == 0))
spec = fp / ( fp + tn)
print(f'Specificity_score_test:{spec:.2f}')
```

    Specificity_score_test:0.19



```python
# sensitivity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_t) # No need to iterate already a 1-D array
tp = sum((y_train == 1) & (predict == 1))
fn = sum((y_train == 1) & (predict == 0))
sens = tp / ( tp + fn)
print(f'sensitivity_score_test:{sens:.2f}')
```

    sensitivity_score_test:0.86


### Validation set Model quality


```python
# Accuracy score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
accuracy = model_l.score(X_sc_v,y_val)
print(f'Accuracy_score:{accuracy:.2f}')
```

    Accuracy_score:0.85



```python
# Specificity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_v) # No need to iterate already a 1-D array
fp = sum((y_test == 0) & (predict == 1))
tn = sum((y_test == 0) & (predict == 0))
spec = fp / ( fp + tn)
print(f'Specificity_score_test:{spec:.2f}')
```

    Specificity_score_test:0.57



```python
# sensitivity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_v) # No need to iterate already a 1-D array
tp = sum((y_test == 1) & (predict == 1))
fn = sum((y_test == 1) & (predict == 0))
sens = tp / ( tp + fn)
print(f'sensitivity_score_test:{sens:.2f}')
```

    sensitivity_score_test:0.55


### Test Set Model Quality


```python
# Accuracy score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
accuracy = model_l.score(X_sc_te,y_test)
print(f'Accuracy_score:{accuracy:.2f}')
```

    Accuracy_score:0.82



```python
# Specificity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_te) # No need to iterate already a 1-D array
fp = sum((y_test == 0) & (predict == 1))
tn = sum((y_test == 0) & (predict == 0))
spec = fp / ( fp + tn)
print(f'Specificity_score_test:{spec:.2f}')
```

    Specificity_score_test:0.21



```python
# sensitivity score
model_l = LogisticRegression()
model_l.fit(X_sc_t, y_train)
predict = model_l.predict(X_sc_te) # No need to iterate already a 1-D array
tp = sum((y_test == 1) & (predict == 1))
fn = sum((y_test == 1) & (predict == 0))
sens = tp / ( tp + fn)
print(f'sensitivity_score_test:{sens:.2f}')
```

    sensitivity_score_test:0.85


# Summary:

+ I have used the following features to obtain the model accuracy, which are highly correlated with my target variable, 'HeartDisease':
    + `FastingBS`
    + `Oldpeak`
    + `ST_Slope_Flat`
    + `ExerciseAngina_Y`
    
    
+ The accuracy obtained using the KNN(k-Nearest Neighbour) Model:
    + Validation set : 90%
    + Test set       : 85%
    
+ The accuracy obtained using the Linear SVC Model:
    + Validation set : 84%
    + Test set       : 82%

+ The accuracy obtained using the Logistic Regression Model:
    + Validation set : 85%
    + Test set       : 82%

#### Cons:
   + Changes in the random state and tuning hyperparameters might influence the model's accuracy
   + KNN uses nearest neighbours to identify the class label for the input data point. If the nearest     neighbours belong to a different class, the data may be incorrectly classified
   + The predictors used in the model may also affect accuracy score of the outcome
   + Outliers present in the data may affect the relationship between the predictor and the target variable. For example, the predictor cholesterol has some outliers, which resulted in a low correlation value. However, once these outliers were replaced with median values, the relationship with the target variable improved
    
    
#### Potential improvements to performance:
   + Tuning hyperparameters to optimize the model
   + Exploring various random states could improve the accuracy of class labels
   + The training set comprises 60% of original dataset, while test set is 20% of the training dataset.If the test set does not have equal distributions of target and predictors variables, this could affect the accuracy of the model
   + Incorporating additional models, such as decision tree model, and identifying the best fit for the particular data type would be beneficial.
