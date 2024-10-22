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


## Findings:

+ There are a total of 725 male and 193 female observations available for analysis, indicating that the dataset is skewed towards males. Approximately 63% of males had heart disease, while 26% of females did. This suggests that males may be more prone to heart disease compared to females



+ The 'ASY' category of Chest Pain Type has a higher count (392) of patients with heart disease. The Asymptotic type implies that chest pain is not reported by the patients but is detected in tests



+ Out of 918 observations, 508 patients reported of having Heart Disease

### Categorical data


```python
# Target Variable:  Evenly distributed
#count of Individuals by HeartDisease

grouped_data_heartD= data.groupby('HeartDisease').agg(count=('HeartDisease','count')).reset_index()

print(f'''Individuals with & without Heart Disease:\n{data['HeartDisease'].value_counts()}''')
```

    Individuals with & without Heart Disease:
    1    508
    0    410
    Name: HeartDisease, dtype: int64



```python
# Dataset is skewed towards Male patients
# Number of Male and Female observations:

normalized_val = data['Sex'].value_counts(normalize = True)*100
print(f'% of Male & Female patients:\n{normalized_val.round()}')
```

    % of Male & Female patients:
    M    79.0
    F    21.0
    Name: Sex, dtype: float64


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

    N    547
    Y    371
    Name: ExerciseAngina, dtype: int64



```python
# Individuals with ST_slope features
print(data['ST_Slope'].value_counts())
```

    Flat    460
    Up      395
    Down     63
    Name: ST_Slope, dtype: int64


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
    


### Correlation Analysis

## Findings:
+ Heart disease is positively related to the features Oldpeak, Fasting Blood Sugar, Resting Blood Pressure, and Age
+ Heart disease is negatively related to Cholesterol and Maximum Heart Rate (HR)


```python
# High correlation might indicate that skewed distributions are affecting the outcome
# Correlation with the Target - Heart disease

print(data.corr()['HeartDisease'])
```

    Age             0.282039
    RestingBP       0.107589
    Cholesterol    -0.232741
    FastingBS       0.267291
    MaxHR          -0.400421
    Oldpeak         0.403951
    HeartDisease    1.000000
    Name: HeartDisease, dtype: float64


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_9448/3203825956.py:4: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      print(data.corr()['HeartDisease'])


## Numerical Data Analysis:

## Findings:

+ The `Age` ranges from 28 to 77 in the dataset
+ `Age` 54,55 & 58 tops the dataset
+ The `Cholesterol` and `Resting BP` fields have a minimum value of zero, which is not feasible for a healthy human body, indicating that the data needs to be handled
+ Mean age of dataset is 54.Meanwhile Median age of Heart Disease patients is 57
+ Out of 918 observations,383 patients of age above 50 have Heart Disease
+ There are 170 patients with heart disease who have blood sugar levels above 120 mg, and 44 patients without a diagnosis of heart disease


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


    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/Individual%20count%20by%20age.png)
    


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
grouped_data_age_HD = data.groupby(['Age','HeartDisease']).agg(HD_count=('HeartDisease', 'count')).reset_index()
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


    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/Individuals%20count%20by%20Heart%20Disease%20and%20Age.png)
    


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


    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/count%20of%20Individuals%20with%20no%20heart%20disease%20by%20Age.png)
    


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



### Distribution of Cholesterol


```python
# Distribution of Cholesterol field is skewed to the right
data['Cholesterol'].hist(figsize=(10,6),color='skyblue', edgecolor='black')
plt.title('Cholesterol Distribution')  
plt.xlabel('Cholesterol Level')         # X-axis label
plt.ylabel('Frequency')                  # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```




    <Axes: >




    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/Cholestrol%20Distribution.png)
    



```python
# A skewness value close to 0 indicates a symmetric distribution, 
# while values greater than 1 or less than -1 indicate high skewness

skewness = data['Cholesterol'].skew()
print(f'Skewness: {skewness}')
```

    Skewness: -0.6100864307268192



```python
# There were 172 observations with Cholesterol as 0

print(f"Data with Minimum Cholesterol level as 0:{data[data['Cholesterol']== 0]['Age'].count()}")
```

    Data with Minimum Cholesterol level as 0:172


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



#### Resting BP Analysis


```python
# Resting BP - No possibility for the presence of 0  - Min value
data['RestingBP'].describe()
```




    count    918.000000
    mean     132.396514
    std       18.514154
    min        0.000000
    25%      120.000000
    50%      130.000000
    75%      140.000000
    max      200.000000
    Name: RestingBP, dtype: float64




```python
# Observations with Minimum Cholesterol level as 0 

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



#### Dropping the index 449 as the row does not have meaningful Resting BP nor cholestrol


```python
# Dropping the index: 449 as it does not have meaningful data in other fields as well
data.drop(index = 449,inplace = True)
```


```python
# There are 1 observations with RestingBP as 0

print(f"Data with RestingBP 0:{data[data['RestingBP']== 0]['Age'].count()}")
```

    Data with RestingBP 0:0



```python
# Heart rate - Minimum of 60 and Maximum of 202 recorded
print(data['MaxHR'].describe())
```

    count    917.000000
    mean     136.789531
    std       25.467129
    min       60.000000
    25%      120.000000
    50%      138.000000
    75%      156.000000
    max      202.000000
    Name: MaxHR, dtype: float64



```python
print(data['Oldpeak'].describe())
```

    count    917.000000
    mean       0.886696
    std        1.066960
    min       -2.600000
    25%        0.000000
    50%        0.600000
    75%        1.500000
    max        6.200000
    Name: Oldpeak, dtype: float64


### Correlation analysis - Post replacing 'Cholesterol' values with median values


```python
# High correlation might indicate that skewed distributions are affecting the outcome
# Correlation with the Target - Heart disease

print(data.corr()['HeartDisease'])
```

    Age             0.282012
    RestingBP       0.117990
    Cholesterol     0.024914
    FastingBS       0.267994
    MaxHR          -0.401410
    Oldpeak         0.403638
    HeartDisease    1.000000
    Name: HeartDisease, dtype: float64


    /var/folders/16/3tzmlcc129z3xr007ykwjb640000gp/T/ipykernel_9448/3203825956.py:4: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      print(data.corr()['HeartDisease'])


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
    1                  126.0


### Distribution of `MaxHR` field


```python
# Distribution of 'MaxHR' field is slightly skewed to the left
data['MaxHR'].hist(figsize=(10,6),color='lightgrey', edgecolor='black')
plt.title('Maximum Heart Rate Distribution')  
plt.xlabel('MaxHR Level')         # X-axis label
plt.ylabel('Frequency')                  # Y-axis label
plt.grid(axis='y', alpha=0.75)         
plt.show()
```




    <Axes: >




    
![png](https://github.com/Muthaln1/Predicting-Heart-Disease--ML-model/blob/main/MaxHR%20Distribution.png)
    



```python
# skewness rate
skewness = data['MaxHR'].skew()
print(f'Skewness: {skewness}')
```

    Skewness: -0.14245852926814553


### Features to choose for our model:


+ I am chosing the below features to design my model. Because they are strongly correlating with my target variable (`Heart Disease`) and I believe that they add value to my model

    + `Oldpeak`                  
    + `MaxHR`
    + `ST_Slope_Flat`
    + `ST_Slope_Up`
    + `ExerciseAngina_Y`
    + `ChestPainType_ATA`

## Findings:

+ `Cholesterol` field has no significant relationship with HeartDisease based on the correlation analysis

### Assigining dummy variables to the categorical column to include them in the Analysis:


```python
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 917 entries, 0 to 917
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Age             917 non-null    int64  
     1   Sex             917 non-null    object 
     2   ChestPainType   917 non-null    object 
     3   RestingBP       917 non-null    int64  
     4   Cholesterol     917 non-null    float64
     5   FastingBS       917 non-null    int64  
     6   RestingECG      917 non-null    object 
     7   MaxHR           917 non-null    int64  
     8   ExerciseAngina  917 non-null    object 
     9   Oldpeak         917 non-null    float64
     10  ST_Slope        917 non-null    object 
     11  HeartDisease    917 non-null    int64  
    dtypes: float64(2), int64(5), object(5)
    memory usage: 93.1+ KB
    None


## One hot encoding:


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

    Age                  0.282012
    RestingBP            0.117990
    Cholesterol          0.024914
    FastingBS            0.267994
    MaxHR               -0.401410
    Oldpeak              0.403638
    HeartDisease         1.000000
    Sex_M                0.305118
    ChestPainType_ATA   -0.401680
    ChestPainType_NAP   -0.215311
    ChestPainType_TA    -0.054591
    RestingECG_Normal   -0.092452
    RestingECG_ST        0.103067
    ExerciseAngina_Y     0.495490
    ST_Slope_Flat        0.553700
    ST_Slope_Up         -0.621843
    Name: HeartDisease, dtype: float64


### Taking absolute value of the Pearson correlation coefficient


```python
correlation_mat = abs(dat.corr())

print(correlation_mat['HeartDisease'])
```

    Age                  0.282012
    RestingBP            0.117990
    Cholesterol          0.024914
    FastingBS            0.267994
    MaxHR                0.401410
    Oldpeak              0.403638
    HeartDisease         1.000000
    Sex_M                0.305118
    ChestPainType_ATA    0.401680
    ChestPainType_NAP    0.215311
    ChestPainType_TA     0.054591
    RestingECG_Normal    0.092452
    RestingECG_ST        0.103067
    ExerciseAngina_Y     0.495490
    ST_Slope_Flat        0.553700
    ST_Slope_Up          0.621843
    Name: HeartDisease, dtype: float64


### Using the square of the Pearson correlation coefficient to have range between 0 & 1


```python
correlation_matrix = dat.corr()

correlation_matrix = correlation_matrix ** 2

print(correlation_matrix['HeartDisease'])
```

    Age                  0.079531
    RestingBP            0.013922
    Cholesterol          0.000621
    FastingBS            0.071821
    MaxHR                0.161130
    Oldpeak              0.162924
    HeartDisease         1.000000
    Sex_M                0.093097
    ChestPainType_ATA    0.161346
    ChestPainType_NAP    0.046359
    ChestPainType_TA     0.002980
    RestingECG_Normal    0.008547
    RestingECG_ST        0.010623
    ExerciseAngina_Y     0.245510
    ST_Slope_Flat        0.306584
    ST_Slope_Up          0.386688
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


    
![png](output_83_0.png)
    


### Highlighting feature with Correlation above 0.15


```python
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix[correlation_matrix > 0.15], annot=True, cmap="Reds")
```




    <Axes: >




    
![png](output_85_1.png)
    


### Segmenting the data and the Target


```python
X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```

### Findings:

+ Out of all selected features, `Oldpeak` performed well with a 70.65% accuracy based on numerical data

+ Meanwhile under categorical features, the high performers are:
     + ST_Slope_Flat: Accuracy:78.80%, 
     + ST_Slope_Up: Accuracy:82.61%, 
     + ExerciseAngina_Y: Accuracy:70.11%
     
+ By using combined features like `FastingBS`,`Oldpeak`,`ST_Slope_Flat`,`ST_Slope_Up` & `ExerciseAngina_Y`, I achieved an accuracy of 85% on the validation set and 87% on the test set

### Single feature model


```python
# Numerical Features to be used in the single feature model
features =['Age','RestingBP','FastingBS','Cholesterol','Oldpeak','MaxHR']
```

### Splitting the dataset into Training and Validation set


```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X[features],y,test_size = .20,random_state = 439)
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

    Feature Age: Accuracy 60.87%
    Feature RestingBP: Accuracy 57.07%
    Feature FastingBS: Accuracy 54.35%
    Feature Cholesterol: Accuracy 44.02%
    Feature Oldpeak: Accuracy 67.39%
    Feature MaxHR: Accuracy 62.50%


### Categorical data with dummy variables and their accuracy:


```python
# Segmenting the data and the Target
X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```


```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = .20,random_state = 439)
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

     Feature Sex_M: Accuracy:53.80%
     Feature ST_Slope_Flat: Accuracy:77.17%
     Feature ST_Slope_Up: Accuracy:82.07%
     Feature ExerciseAngina_Y: Accuracy:70.11%
     Feature ChestPainType_ATA: Accuracy:46.20%


### Multi feature model:


```python
# Segmenting the data and the Target

X = dat.drop('HeartDisease',axis =1)
y = dat['HeartDisease']
```


```python
# Combined features
feature2 =['FastingBS','Oldpeak','ST_Slope_Flat','ST_Slope_Up','ExerciseAngina_Y']
```

### Data split:
    + Training set - 60%
    + Validation set - 20%
    + Test set - 20%


```python
# Splitting the dataset into Training and Validation set:
X_train, X_val, y_train, y_val = train_test_split(X[feature2],y,test_size = .20,random_state = 439)
```


```python
# Observations Train and validation set
print(f'Observations in Train set:{X_train.shape[0]}\n',f'Observations in the Validation set:{X_val.shape[0]}')
```

    Observations in Train set:733
     Observations in the Validation set:184



```python
# Splitting the dataset into Training and test set from the training set:

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size = .25,random_state = 439)
```


```python
# Observations in the Train and Test set

print(f'Observations in Train set:{X_train.shape[0]}\n',f'Observations in Test set:{X_test.shape[0]}')
```

    Observations in Train set:549
     Observations in Test set:184



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

    {'n_neighbors': 5, 'p': 1, 'weights': 'uniform'} 0.8378648874061719


#### KNN classifier


```python
# creating a KNN classifier & validating them on the Validation dataset: 

model = KNeighborsClassifier(n_neighbors= 5, p = 1, weights = 'uniform')
model.fit(X_train_sca,y_train)
accuracy = model.score(X_val_sca,y_val)
accuracy = accuracy*100
print(f'Accuracy of the validation data set:{accuracy:.0f}%')
```

    Accuracy of the validation data set:85%


### Test set Accuracy:


```python
# Using the categorical data as it is for the Validation set
X_test_sca = X_test

# Fit transforming the numerical data for the training set
X_test_sca[['Oldpeak','FastingBS']] = scaler.transform(X_test[['Oldpeak','FastingBS']])
```


```python
# creating a KNN classifier & validating them on the Validation dataset: 
model = KNeighborsClassifier(n_neighbors= 5, p = 1, weights = 'uniform')
model.fit(X_train_sca,y_train)
accuracy = model.score(X_test_sca,y_test)
accuracy = accuracy*100
print(f'Accuracy of the Test Set:{accuracy:.0f}%')
```

    Accuracy of the Test Set:87%


## Using Linear SVC model - Model 2

## Findings:
+ By using combined features like `FastingBS`,`Oldpeak`,`ST_Slope_Flat`,`ST_Slope_Up` & `ExerciseAngina_Y`, I achieved an accuracy of 82% on the validation set and 85% on the test set
+ As the KNN classifier is based on the nearest neighbours there might be a difference of margin calssification compared to Linera SVC


```python
from sklearn.svm import LinearSVC

model = LinearSVC(penalty="l2",loss="squared_hinge",C=10,random_state=417)
```


```python
model.fit(X_train_sca,y_train)
```

    /Users/nandhinimuthalraj/anaconda3/lib/python3.11/site-packages/sklearn/svm/_classes.py:32: FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.
      warnings.warn(
    /Users/nandhinimuthalraj/anaconda3/lib/python3.11/site-packages/sklearn/svm/_base.py:1242: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(





<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearSVC(C=10, random_state=417)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVC</label><div class="sk-toggleable__content"><pre>LinearSVC(C=10, random_state=417)</pre></div></div></div></div></div>




```python
val_accuracy = model.score(X_val_sca,y_val)
print(f'Accuracy of the Validation dataset via SVC Model:{val_accuracy*100:.0f}%')
```

    Accuracy of the Validation dataset via SVC Model:82%



```python
test_accuracy = model.score(X_test_sca,y_test)
print(f'Accuracy of the Test dataset via SVC Model:{test_accuracy*100:.0f}%')
```

    Accuracy of the Test dataset via SVC Model:85%


# Summary:

+ I have used the following features to obtain the model accuracy, which are highly correlated with my target variable, 'HeartDisease':
    + `FastingBS`
    + `Oldpeak`
    + `ST_Slope_Flat`
    + `ST_Slope_Up`
    + `ExerciseAngina_Y`
    
    
+ The accuracy obtained using the KNN model:
    + Validation set : 85%
    + Test set       : 87%
    
+ The accuracy obtained using the Linear SVC model:
    + Validation set : 82%
    + Test set       : 85%


+ cons:
    + Changes in the random state and tuning hyperparameters might influence the model's accuracy
    + KNN uses nearest neighbours to identify the class label for the input data point. If the nearest     neighbours belong to a different class, the data may be incorrectly classified.
    + The predictors used in the model may also affect accuracy. For example, We might assume cholestrol is associated with heart Disease, but the dataset shows a low correlation between Cholestrol (predictor) and heart Disease (target)
    

    
+ Potential improvements to performance:
     + Tuning hyperparameters to optimize the model
     + Exploring various random states could improve the accuracy of class labels
     + The training set comprises 60% of original dataset, while test set is 20% of the training dataset.If the test set does not have equal distributions of target and predictors variables, this could affect the accuracy of the model
     + Incorporating additional models, such as logistic regression, and identifying the best fit for the particular data type would be beneficial.
