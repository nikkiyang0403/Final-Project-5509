```python
pip install ucimlrepo
```

    Requirement already satisfied: ucimlrepo in c:\users\s209160\anaconda3\lib\site-packages (0.0.7)
    Requirement already satisfied: certifi>=2020.12.5 in c:\users\s209160\anaconda3\lib\site-packages (from ucimlrepo) (2021.10.8)
    Requirement already satisfied: pandas>=1.0.0 in c:\users\s209160\anaconda3\lib\site-packages (from ucimlrepo) (1.3.4)
    Requirement already satisfied: pytz>=2017.3 in c:\users\s209160\anaconda3\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2021.3)
    Requirement already satisfied: numpy>=1.17.3 in c:\users\s209160\anaconda3\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (1.20.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\s209160\anaconda3\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2.8.2)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: six>=1.5 in c:\users\s209160\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas>=1.0.0->ucimlrepo) (1.16.0)
    


```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 
```

    {'uci_id': 45, 'name': 'Heart Disease', 'repository_url': 'https://archive.ics.uci.edu/dataset/45/heart+disease', 'data_url': 'https://archive.ics.uci.edu/static/public/45/data.csv', 'abstract': '4 databases: Cleveland, Hungary, Switzerland, and the VA Long Beach', 'area': 'Health and Medicine', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 303, 'num_features': 13, 'feature_types': ['Categorical', 'Integer', 'Real'], 'demographics': ['Age', 'Sex'], 'target_col': ['num'], 'index_col': None, 'has_missing_values': 'yes', 'missing_values_symbol': 'NaN', 'year_of_dataset_creation': 1989, 'last_updated': 'Fri Nov 03 2023', 'dataset_doi': '10.24432/C52P4X', 'creators': ['Andras Janosi', 'William Steinbrunn', 'Matthias Pfisterer', 'Robert Detrano'], 'intro_paper': {'ID': 231, 'type': 'NATIVE', 'title': 'International application of a new probability algorithm for the diagnosis of coronary artery disease.', 'authors': 'R. Detrano, A. Jánosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, V. Froelicher', 'venue': 'American Journal of Cardiology', 'year': 1989, 'journal': None, 'DOI': None, 'URL': 'https://www.semanticscholar.org/paper/a7d714f8f87bfc41351eb5ae1e5472f0ebbe0574', 'sha': None, 'corpus': None, 'arxiv': None, 'mag': None, 'acl': None, 'pmid': '2756873', 'pmcid': None}, 'additional_info': {'summary': 'This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.  In particular, the Cleveland database is the only one that has been used by ML researchers to date.  The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).  \n   \nThe names and social security numbers of the patients were recently removed from the database, replaced with dummy values.\n\nOne file has been "processed", that one containing the Cleveland database.  All four unprocessed files also exist in this directory.\n\nTo see Test Costs (donated by Peter Turney), please see the folder "Costs" ', 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': 'Only 14 attributes used:\r\n      1. #3  (age)       \r\n      2. #4  (sex)       \r\n      3. #9  (cp)        \r\n      4. #10 (trestbps)  \r\n      5. #12 (chol)      \r\n      6. #16 (fbs)       \r\n      7. #19 (restecg)   \r\n      8. #32 (thalach)   \r\n      9. #38 (exang)     \r\n      10. #40 (oldpeak)   \r\n      11. #41 (slope)     \r\n      12. #44 (ca)        \r\n      13. #51 (thal)      \r\n      14. #58 (num)       (the predicted attribute)\r\n\r\nComplete attribute documentation:\r\n      1 id: patient identification number\r\n      2 ccf: social security number (I replaced this with a dummy value of 0)\r\n      3 age: age in years\r\n      4 sex: sex (1 = male; 0 = female)\r\n      5 painloc: chest pain location (1 = substernal; 0 = otherwise)\r\n      6 painexer (1 = provoked by exertion; 0 = otherwise)\r\n      7 relrest (1 = relieved after rest; 0 = otherwise)\r\n      8 pncaden (sum of 5, 6, and 7)\r\n      9 cp: chest pain type\r\n        -- Value 1: typical angina\r\n        -- Value 2: atypical angina\r\n        -- Value 3: non-anginal pain\r\n        -- Value 4: asymptomatic\r\n     10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)\r\n     11 htn\r\n     12 chol: serum cholestoral in mg/dl\r\n     13 smoke: I believe this is 1 = yes; 0 = no (is or is not a smoker)\r\n     14 cigs (cigarettes per day)\r\n     15 years (number of years as a smoker)\r\n     16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)\r\n     17 dm (1 = history of diabetes; 0 = no such history)\r\n     18 famhist: family history of coronary artery disease (1 = yes; 0 = no)\r\n     19 restecg: resting electrocardiographic results\r\n        -- Value 0: normal\r\n        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)\r\n        -- Value 2: showing probable or definite left ventricular hypertrophy by Estes\' criteria\r\n     20 ekgmo (month of exercise ECG reading)\r\n     21 ekgday(day of exercise ECG reading)\r\n     22 ekgyr (year of exercise ECG reading)\r\n     23 dig (digitalis used furing exercise ECG: 1 = yes; 0 = no)\r\n     24 prop (Beta blocker used during exercise ECG: 1 = yes; 0 = no)\r\n     25 nitr (nitrates used during exercise ECG: 1 = yes; 0 = no)\r\n     26 pro (calcium channel blocker used during exercise ECG: 1 = yes; 0 = no)\r\n     27 diuretic (diuretic used used during exercise ECG: 1 = yes; 0 = no)\r\n     28 proto: exercise protocol\r\n          1 = Bruce     \r\n          2 = Kottus\r\n          3 = McHenry\r\n          4 = fast Balke\r\n          5 = Balke\r\n          6 = Noughton \r\n          7 = bike 150 kpa min/min  (Not sure if "kpa min/min" is what was written!)\r\n          8 = bike 125 kpa min/min  \r\n          9 = bike 100 kpa min/min\r\n         10 = bike 75 kpa min/min\r\n         11 = bike 50 kpa min/min\r\n         12 = arm ergometer\r\n     29 thaldur: duration of exercise test in minutes\r\n     30 thaltime: time when ST measure depression was noted\r\n     31 met: mets achieved\r\n     32 thalach: maximum heart rate achieved\r\n     33 thalrest: resting heart rate\r\n     34 tpeakbps: peak exercise blood pressure (first of 2 parts)\r\n     35 tpeakbpd: peak exercise blood pressure (second of 2 parts)\r\n     36 dummy\r\n     37 trestbpd: resting blood pressure\r\n     38 exang: exercise induced angina (1 = yes; 0 = no)\r\n     39 xhypo: (1 = yes; 0 = no)\r\n     40 oldpeak = ST depression induced by exercise relative to rest\r\n     41 slope: the slope of the peak exercise ST segment\r\n        -- Value 1: upsloping\r\n        -- Value 2: flat\r\n        -- Value 3: downsloping\r\n     42 rldv5: height at rest\r\n     43 rldv5e: height at peak exercise\r\n     44 ca: number of major vessels (0-3) colored by flourosopy\r\n     45 restckm: irrelevant\r\n     46 exerckm: irrelevant\r\n     47 restef: rest raidonuclid (sp?) ejection fraction\r\n     48 restwm: rest wall (sp?) motion abnormality\r\n        0 = none\r\n        1 = mild or moderate\r\n        2 = moderate or severe\r\n        3 = akinesis or dyskmem (sp?)\r\n     49 exeref: exercise radinalid (sp?) ejection fraction\r\n     50 exerwm: exercise wall (sp?) motion \r\n     51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect\r\n     52 thalsev: not used\r\n     53 thalpul: not used\r\n     54 earlobe: not used\r\n     55 cmo: month of cardiac cath (sp?)  (perhaps "call")\r\n     56 cday: day of cardiac cath (sp?)\r\n     57 cyr: year of cardiac cath (sp?)\r\n     58 num: diagnosis of heart disease (angiographic disease status)\r\n        -- Value 0: < 50% diameter narrowing\r\n        -- Value 1: > 50% diameter narrowing\r\n        (in any major vessel: attributes 59 through 68 are vessels)\r\n     59 lmt\r\n     60 ladprox\r\n     61 laddist\r\n     62 diag\r\n     63 cxmain\r\n     64 ramus\r\n     65 om1\r\n     66 om2\r\n     67 rcaprox\r\n     68 rcadist\r\n     69 lvx1: not used\r\n     70 lvx2: not used\r\n     71 lvx3: not used\r\n     72 lvx4: not used\r\n     73 lvf: not used\r\n     74 cathef: not used\r\n     75 junk: not used\r\n     76 name: last name of patient  (I replaced this with the dummy string "name")', 'citation': None}}
            name     role         type demographic  \
    0        age  Feature      Integer         Age   
    1        sex  Feature  Categorical         Sex   
    2         cp  Feature  Categorical        None   
    3   trestbps  Feature      Integer        None   
    4       chol  Feature      Integer        None   
    5        fbs  Feature  Categorical        None   
    6    restecg  Feature  Categorical        None   
    7    thalach  Feature      Integer        None   
    8      exang  Feature  Categorical        None   
    9    oldpeak  Feature      Integer        None   
    10     slope  Feature  Categorical        None   
    11        ca  Feature      Integer        None   
    12      thal  Feature  Categorical        None   
    13       num   Target      Integer        None   
    
                                              description  units missing_values  
    0                                                None  years             no  
    1                                                None   None             no  
    2                                                None   None             no  
    3   resting blood pressure (on admission to the ho...  mm Hg             no  
    4                                   serum cholestoral  mg/dl             no  
    5                     fasting blood sugar > 120 mg/dl   None             no  
    6                                                None   None             no  
    7                         maximum heart rate achieved   None             no  
    8                             exercise induced angina   None             no  
    9   ST depression induced by exercise relative to ...   None             no  
    10                                               None   None             no  
    11  number of major vessels (0-3) colored by flour...   None            yes  
    12                                               None   None            yes  
    13                         diagnosis of heart disease   None             no  
    


```python
heart_disease.data
```




    {'ids': None,
     'features':      age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \
     0     63    1   1       145   233    1        2      150      0      2.3   
     1     67    1   4       160   286    0        2      108      1      1.5   
     2     67    1   4       120   229    0        2      129      1      2.6   
     3     37    1   3       130   250    0        0      187      0      3.5   
     4     41    0   2       130   204    0        2      172      0      1.4   
     ..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   
     298   45    1   1       110   264    0        0      132      0      1.2   
     299   68    1   4       144   193    1        0      141      0      3.4   
     300   57    1   4       130   131    0        0      115      1      1.2   
     301   57    0   2       130   236    0        2      174      0      0.0   
     302   38    1   3       138   175    0        0      173      0      0.0   
     
          slope   ca  thal  
     0        3  0.0   6.0  
     1        2  3.0   3.0  
     2        2  2.0   7.0  
     3        3  0.0   3.0  
     4        1  0.0   3.0  
     ..     ...  ...   ...  
     298      2  0.0   7.0  
     299      2  2.0   7.0  
     300      2  1.0   7.0  
     301      2  1.0   3.0  
     302      1  NaN   3.0  
     
     [303 rows x 13 columns],
     'targets':      num
     0      0
     1      2
     2      1
     3      0
     4      0
     ..   ...
     298    1
     299    2
     300    3
     301    1
     302    0
     
     [303 rows x 1 columns],
     'original':      age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \
     0     63    1   1       145   233    1        2      150      0      2.3   
     1     67    1   4       160   286    0        2      108      1      1.5   
     2     67    1   4       120   229    0        2      129      1      2.6   
     3     37    1   3       130   250    0        0      187      0      3.5   
     4     41    0   2       130   204    0        2      172      0      1.4   
     ..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   
     298   45    1   1       110   264    0        0      132      0      1.2   
     299   68    1   4       144   193    1        0      141      0      3.4   
     300   57    1   4       130   131    0        0      115      1      1.2   
     301   57    0   2       130   236    0        2      174      0      0.0   
     302   38    1   3       138   175    0        0      173      0      0.0   
     
          slope   ca  thal  num  
     0        3  0.0   6.0    0  
     1        2  3.0   3.0    2  
     2        2  2.0   7.0    1  
     3        3  0.0   3.0    0  
     4        1  0.0   3.0    0  
     ..     ...  ...   ...  ...  
     298      2  0.0   7.0    1  
     299      2  2.0   7.0    2  
     300      2  1.0   7.0    3  
     301      2  1.0   3.0    1  
     302      1  NaN   3.0    0  
     
     [303 rows x 14 columns],
     'headers': Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
           dtype='object')}




```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
```


```python
# Fetch dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features  # Features
y = heart_disease.data.targets   # Target variable
```


```python
print(X.isnull().sum()) # Check missing values in features
print(y.isnull().sum())  # Check missing values in target variable
```

    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    ca          4
    thal        2
    dtype: int64
    num    0
    dtype: int64
    


```python
# Drop rows with NaN values
X.dropna(inplace=True)
y = y.loc[X.index]  # Ensure target matches feature rows
```

    C:\Users\S209160\Anaconda3\lib\site-packages\pandas\util\_decorators.py:311: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return func(*args, **kwargs)
    


```python
print(X.isnull().sum().sum())  # Should print 0 if all NAs are handled
```

    0
    


```python
# Exploratory Data Analysis (EDA)
print("Dataset Overview:\n", X.info())
print("Summary Statistics:\n", X.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 297 entries, 0 to 301
    Data columns (total 13 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       297 non-null    int64  
     1   sex       297 non-null    int64  
     2   cp        297 non-null    int64  
     3   trestbps  297 non-null    int64  
     4   chol      297 non-null    int64  
     5   fbs       297 non-null    int64  
     6   restecg   297 non-null    int64  
     7   thalach   297 non-null    int64  
     8   exang     297 non-null    int64  
     9   oldpeak   297 non-null    float64
     10  slope     297 non-null    int64  
     11  ca        297 non-null    float64
     12  thal      297 non-null    float64
    dtypes: float64(3), int64(10)
    memory usage: 32.5 KB
    Dataset Overview:
     None
    Summary Statistics:
                   age         sex          cp    trestbps        chol         fbs  \
    count  297.000000  297.000000  297.000000  297.000000  297.000000  297.000000   
    mean    54.542088    0.676768    3.158249  131.693603  247.350168    0.144781   
    std      9.049736    0.468500    0.964859   17.762806   51.997583    0.352474   
    min     29.000000    0.000000    1.000000   94.000000  126.000000    0.000000   
    25%     48.000000    0.000000    3.000000  120.000000  211.000000    0.000000   
    50%     56.000000    1.000000    3.000000  130.000000  243.000000    0.000000   
    75%     61.000000    1.000000    4.000000  140.000000  276.000000    0.000000   
    max     77.000000    1.000000    4.000000  200.000000  564.000000    1.000000   
    
              restecg     thalach       exang     oldpeak       slope          ca  \
    count  297.000000  297.000000  297.000000  297.000000  297.000000  297.000000   
    mean     0.996633  149.599327    0.326599    1.055556    1.602694    0.676768   
    std      0.994914   22.941562    0.469761    1.166123    0.618187    0.938965   
    min      0.000000   71.000000    0.000000    0.000000    1.000000    0.000000   
    25%      0.000000  133.000000    0.000000    0.000000    1.000000    0.000000   
    50%      1.000000  153.000000    0.000000    0.800000    2.000000    0.000000   
    75%      2.000000  166.000000    1.000000    1.600000    2.000000    1.000000   
    max      2.000000  202.000000    1.000000    6.200000    3.000000    3.000000   
    
                 thal  
    count  297.000000  
    mean     4.730640  
    std      1.938629  
    min      3.000000  
    25%      3.000000  
    50%      3.000000  
    75%      7.000000  
    max      7.000000  
    


```python

```


```python
# Visualizing correlations
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()
```


    
![png](output_10_0.png)
    



```python
# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Build Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(max_depth=4, random_state=42)




```python
# Predictions
y_pred = clf.predict(X_test)
```


```python
# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

    Accuracy: 0.5833333333333334
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.83      0.94      0.88        36
               1       0.00      0.00      0.00         9
               2       0.06      0.20      0.10         5
               3       0.00      0.00      0.00         7
               4       0.00      0.00      0.00         3
    
        accuracy                           0.58        60
       macro avg       0.18      0.23      0.20        60
    weighted avg       0.50      0.58      0.54        60
    
    Confusion Matrix:
     [[34  0  2  0  0]
     [ 4  0  5  0  0]
     [ 2  0  1  2  0]
     [ 1  1  5  0  0]
     [ 0  0  3  0  0]]
    

    C:\Users\S209160\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\S209160\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\S209160\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['0', '1', '2', '3', '4'], 
            yticklabels=['0', '1', '2', '3', '4'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Plot Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(clf, proportion=True)
plt.title("Decision Tree Visualization")
plt.show()
```


    
![png](output_16_0.png)
    



```python
# Build Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train.values.ravel())
```




    RandomForestClassifier(random_state=42)




```python
# Predictions with Random Forest
y_pred_rf = rf_clf.predict(X_test)
```


```python
# Model Evaluation for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
```

    Random Forest Accuracy: 0.6
    Random Forest Classification Report:
                   precision    recall  f1-score   support
    
               0       0.80      0.97      0.88        36
               1       0.00      0.00      0.00         9
               2       0.14      0.20      0.17         5
               3       0.00      0.00      0.00         7
               4       0.00      0.00      0.00         3
    
        accuracy                           0.60        60
       macro avg       0.19      0.23      0.21        60
    weighted avg       0.49      0.60      0.54        60
    
    Random Forest Confusion Matrix:
     [[35  1  0  0  0]
     [ 5  0  3  1  0]
     [ 2  1  1  1  0]
     [ 1  2  3  0  1]
     [ 1  2  0  0  0]]
    


```python
# Plot Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', 
            xticklabels=['0', '1', '2', '3', '4'], 
            yticklabels=['0', '1', '2', '3', '4'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.show()
```


    
![png](output_20_0.png)
    



```python

```


      File "C:\Users\S209160\AppData\Local\Temp/ipykernel_12940/4069685416.py", line 1
        jupyter nbconvert --to rmarkdown your_notebook.ipynb
                ^
    SyntaxError: invalid syntax
    



```python

```
