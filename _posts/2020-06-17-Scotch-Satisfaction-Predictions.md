---
layout: post
title: Scotch Satisfaction Predictions
subtitle: Insights into good Scotches
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [school project, whisky, EDA, random forest, logistic regression, visualizations]
comments: true
---


# Predicting Scotch Satisfaction Ratings
*The dataset that was used in this project can be found [here](https://www.kaggle.com/koki25ando/22000-scotch-whisky-reviews) on Kaggle. It was collected from Whisky Advocate by Kaggle user Andy II.*

Project synopsis: Can we predict whether a whisky can be considered Poor, Fair, Good, or Excellent based off of it's category, Whisky Advocate score, price, and the reviewer's description?

*Note: This project uses scotch and whisky interchangably. While the author is aware that the wide world of whisky extends far beyond scotch alone, for this project we will just be looking at scotch whiskies and it is therefore inaccurate for use with bourbons, Japanese or Irish whiskies, or any other variation thereof.*

This project is designed with two major applications in mind, one for the consumer and one for the distiller.

For the consumer, it is my hope that this data can be useful in determining whether to give a whisky a try. While perhaps a bit unwieldy compared to directly consulting review sites such as Whisky Advocate, not all whiskies have reviews (or if they do there is a woefully small amount of them, which can bring into question how much partiality or bias went in to those scores.) Instead, by looking at the features used in these models and comparing them to prospective whiskies the consumer will be better armed in making an informed, and happily enjoyable, purchase.

Similarly, for the distiller, these models will show which key features lead more often than not towards Excellent ratings. This will allow for the tailoring of future mash bill recipes in ways to capitalize on the taste preferences of the consumer base.

However, these models are limited in scope. They do not take into account things like mash bill recpie percentages, barrel aging processes, or other key phases of the distillation process. Neither are these models good for whiskies that have extensive reviews already, if for no other reason than it would be easier to consult a review site or app than it would to apply these models to the whisky in question.

My target therefore initially was pegged to be the review point score from Whisky Advocate, but for reasons I will explain below I instead opted to create a new feature I called satisfaction rating. This feature is a categorical description of score bands with four separate classes: **Excellent**, **Good**, **Fair**, and **Poor**.

For the models themselves, I will be evaluating their performance on Accuracy, Precision, and Recall, as well as taking a look at sci-kit learn's classification report metric. As you will see shortly, Accuracy alone will prove inadequate due to the relative evenness of the classes (our baselines run from 20.1% to 27.7% on a random guess).

# Exploratory Analysis
In [1]:  
```python
# Imports and first dataset reading. Cleaned Index column and renamed column 
# 'review.point' to avoid any unnecessary issues with dot-notation.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

!pip install eli5 
!pip install category_encoders==2.*
!pip install pdpbox

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')
```

In [2]:  
``` python
scotch_df=pd.read_csv('scotch_review.csv').drop(columns='Unnamed: 0')
scotch_df=scotch_df.rename(columns={'review.point': 'review_point'})
scotch_df.head()
```
Out[2]:  
name	category	review_point	price	currency	description
0	Johnnie Walker Blue Label, 40%	Blended Scotch Whisky	97	225	$</td> <td>Magnificently powerful and intense. Caramels, ...</td> </tr> <tr> <th>1</th> <td>Black Bowmore, 1964 vintage, 42 year old, 40.5%</td> <td>Single Malt Scotch</td> <td>97</td> <td>4500.00</td> <td>$	What impresses me most is how this whisky evol...
2	Bowmore 46 year old (distilled 1964), 42.9%	Single Malt Scotch	97	13500.00	$</td> <td>There have been some legendary Bowmores from t...</td> </tr> <tr> <th>3</th> <td>Compass Box The General, 53.4%</td> <td>Blended Malt Scotch Whisky</td> <td>96</td> <td>325</td> <td>$	With a name inspired by a 1926 Buster Keaton m...
4	Chivas Regal Ultis, 40%	Blended Malt Scotch Whisky	96	160	$	Captivating, enticing, and wonderfully charmin...

In [3]:  
```python
# check for null values
scotch_df.isnull().sum()
```
Out[3]:  
```python
name            0
category        0
review_point    0
price           0
currency        0
description     0
dtype: int64
```

Since we are looking at predicting how well a scotch might be recieved, I initially pegged the `review_point` column as my target variable.

In [4]:  
```python
scotch_df['review_point'].value_counts(normalize=True)
```
Out[4]:  
```python
87    0.098353
86    0.097018
88    0.090788
85    0.088117
89    0.087672
90    0.083667
84    0.076992
83    0.062750
92    0.049844
91    0.047619
82    0.042724
93    0.037383
80    0.036938
81    0.029372
94    0.019137
79    0.013796
95    0.010681
78    0.008011
77    0.005340
96    0.004450
72    0.001780
76    0.001335
75    0.001335
97    0.001335
74    0.000890
70    0.000890
73    0.000890
71    0.000445
63    0.000445
Name: review_point, dtype: float64'
```

In [5]:  
```python
scotch_df['review_point'].describe()
```  
Out[5]:  
```python
count    2247.000000
mean       86.700045
std         4.054055
min        63.000000
25%        84.000000
50%        87.000000
75%        90.000000
max        97.000000
Name: review_point, dtype: float64
```

As you can see, we've a bunch of possible outcomes and not enough features to make good predictions from. At 29 posible outcomes, and only seven features, it just isn't feasible. At least, not with any significant accuracy.

Instead, I opted to combined multiple scores into one of four categories: **Poor**, **Fair**, **Good**, or **Excellent**. This moved my model from one of regression to classification.

Even though the ratings theoretically are from 0 to 100, there is actually a range from 63 to 97. Therefore, I decided to bin each category at the 25%, 50%, 75%, and max scores.

In [6]:  
``` python
scotch_df.loc[scotch_df['review_point'] >= 90, 'satisfaction_rating'] = 'Excellent'
scotch_df.loc[(scotch_df['review_point'] >= 87) & (scotch_df['review_point'] < 90), 'satisfaction_rating'] = 'Good'
scotch_df.loc[(scotch_df['review_point'] >= 84) & (scotch_df['review_point'] < 87), 'satisfaction_rating'] = 'Fair'
scotch_df.loc[scotch_df['review_point'] < 84, 'satisfaction_rating'] = 'Poor'

scotch_df.sample(7)
```
Out[6]:  
name	category	review_point	price	currency	description	satisfaction_rating
418	Blair Athol 23 year old (Diageo Special Releas...	Single Malt Scotch	90	460	$	This Perthshire single malt was distilled in 1...	Excellent
495	Adelphi (distilled at Macallan) 14 year old 19...	Single Malt Scotch	90	117.00	$&lt;/td&gt;
      &lt;td&gt;Here is Macallan in full-blown masculine mode....&lt;/td&gt;
      &lt;td&gt;Excellent&lt;/td&gt;
    &lt;/tr&gt;
    &lt;tr&gt;
      &lt;th&gt;1809&lt;/th&gt;
      &lt;td&gt;Bruichladdich Links, Torrey Pine, 15 year old,...&lt;/td&gt;
      &lt;td&gt;Single Malt Scotch&lt;/td&gt;
      &lt;td&gt;83&lt;/td&gt;
      &lt;td&gt;85.00&lt;/td&gt;
      &lt;td&gt;$	The 8th in a series of “Links” releases. This ...	Poor
1584	Sir Edward’s Smoky, 40%	Blended Scotch Whisky	85	22	$</td> <td>This blend delivers just enough peat turf to l...</td> <td>Fair</td> </tr> <tr> <th>492</th> <td>The Macallan 1861 Replica, 42.7%</td> <td>Single Malt Scotch</td> <td>90</td> <td>180.00</td> <td>$	Antique amber color. Aromas of toffee and malt...	Excellent
2156	Douglas Laing Premier Barrel (distilled at Gle...	Single Malt Scotch	80	95	$</td> <td>Another youthful offering from Douglas Laing i...</td> <td>Poor</td> </tr> <tr> <th>2095</th> <td>The BenRiach 12 year old, 46%</td> <td>Single Malt Scotch</td> <td>80</td> <td>63.00</td> <td>$	Clean, fresh and uncomplicated. Honeyed malt, ...	Poor

In [7]:
```python
# get normalized value_counts() to determine majority class
scotch_df['satisfaction_rating'].value_counts(normalize=True)
```  
Out[7]:  
```python
Good         0.276814
Fair         0.262127
Excellent    0.254117
Poor         0.206943
Name: satisfaction_rating, dtype: float64
```

This gave me a fairly even distribution of classes. The majority class here is **Good** at 27.68% accuracy, much better than the 9.83% accuracy if I had tried to predict the score itself.

This does, however, present an opportunity for data leakage. If I were to fit with the `review_point` feature still in the dataframe, the model will have an artificial 100% accuracy. You will see that I handle this later in the wrangle function, where I exclude this feature from the fitting.

Another potentially problematic feature is the currency feature. If the entire dataset has `price` recorded in USD, then I can ignore the feature because it has nothing substantial to give. If, instead, it happens to have multiple types of currency, I'd need to consider encoding techniques.

In [8]:  
```python
# make sure currency doesn't have multiple types, otherwise will need encoding
scotch_df['currency'].value_counts()
```
Out[8]:  
```python
$    2247  
Name: currency, dtype: int64
```

Since everything is recorded in USD, I will need to also drop this feature within the wrangle function. Speaking of prices, I also wanted to take a look at that particular feature's range of recorded prices.

In [9]:  
```python
scotch_df['price'].value_counts().sort_index()
```
Out[9]:  
```python
$15,000 or $60,000/set     1
1,100                      3
1,400                      1
1,500.00                   1
1,700                      1
                          ..  
98.00                      3
989                        1
99                        11
99.00                      2
995.00                     1
Name: price, Length: 632, dtype: int64
```

We can see here that price is recorded as a string, as indicative by not only a mixing of what appear to be integers and floats, but also from the entry at the top.

`Price` would do better as a numerical feature, but in order to convert it I first needed to deal with the entry with the value "15,000 or $60,000/set". I'm not sure why exactly this entry has a value for purchasing it as a set, but since everything else is cost in USD per bottle that's the value I decided to keep.

In [0]:  
```python
# prep price entries to be converted to floats
scotch_df = scotch_df.replace(to_replace ="$15,000 or $60,000/set", value ="15000")
scotch_df = scotch_df.replace(to_replace ="60000/set", value ="60000")
scotch_df = scotch_df.replace(to_replace ="60000/set", value ="60000")
scotch_df = scotch_df.replace(to_replace ="44/liter", value ="44")
scotch_df['price']=scotch_df['price'].str.replace(",","")
```

I found two additional instances of similar entries, so I cleaned those while I was at it.

In [0]:  
```python
scotch_df['price']=scotch_df['price'].astype(float)
```

## EDA Visualizations
Ok, so let's take a look at a few comparisions between the features.

In [23]:  
```python
sns.pairplot(scotch_df);
```

There seems to be a number of outliers in `price`, which make these visualizations hard to read.

I could altogether drop them, because these specific whiskies are unobtainable for the vast majorty of whisky drinkers anyway and therefore not useful for the intention of this project. But that seemed a significant number of entries to drop, well over 100.

Instead, I opted to try out two different model sets, one that cuts outliers with a price of one thousand dollars or more, and another that raises the cutoff to five thousand.

In [0]:  
```python
scotch_1000=scotch_df.drop(scotch_df.loc[scotch_df['price'] > 1000.000000].index)
scotch_5000=scotch_df.drop(scotch_df.loc[scotch_df['price'] > 5000.000000].index)
```

### EDA Visualizations for Scotch_1000
In [25]:  
```python
sns.pairplot(scotch_1000);
```

In [26]:  
```python
satisfaction_type = pd.crosstab(scotch_1000['category'], scotch_1000['satisfaction_rating'])
satisfaction_type.plot(kind='barh', stacked=True);
```

In [27]:  
```python
scotch_1000.boxplot(column='price', by='category', rot=45, figsize=(10,8));
```

In [28]:  
```python
scotch_1000.boxplot(column='review_point', by='category', rot=45, figsize=(10,8));
```

I noticed here that the first four boxes are similar, but Single Malt Scotch seems to have a disproportionately high number of outliers. Generally speaking, when people think of scotch they think of single malts. Perhaps there is just a disproportionately larger amount of these in our dataset to begin with? Looking at the original dataset, I got the following results:

In [29]:  
```python
scotch_df['category'].value_counts()
```
Out[29]:  
```python
Single Malt Scotch            1819
Blended Scotch Whisky          211
Blended Malt Scotch Whisky     132
Single Grain Whisky             57
Grain Scotch Whisky             28
Name: category, dtype: int64
```

Single Malts are almost nine times as numerous as the next largest category, Blended Scotch. That probably accounts for it.

In this case, I used the original scotch_df to see the distribution of categories, but I also wanted to see how it changed for scotch_1000 and scotch_5000. It looks like the majority of high-cost scotches were also Single Malts.

In [30]:  
```python
scotch_1000['category'].value_counts()
```
Out[30]:  
```python
Single Malt Scotch            1694
Blended Scotch Whisky          201
Blended Malt Scotch Whisky     130
Single Grain Whisky             55
Grain Scotch Whisky             27
Name: category, dtype: int64
```

In [31]:  
```python
scotch_5000['category'].value_counts()
```  
Out[31]:  
```python
Single Malt Scotch            1783
Blended Scotch Whisky          210
Blended Malt Scotch Whisky     132
Single Grain Whisky             57
Grain Scotch Whisky             28
Name: category, dtype: int64
```

### EDA Visualizations for Scotch_5000
I was curious to see how these same visualizations might change due to the raised cutoff for outlier values.

In [32]:   
```python
satisfaction_type = pd.crosstab(scotch_5000['category'], scotch_5000['satisfaction_rating'])
satisfaction_type.plot(kind='barh', stacked=True);
```

In [33]:  
```python
scotch_5000.boxplot(column='price', by='category', rot=45, figsize=(10,8));
```

In [34]:  
```python
scotch_5000.boxplot(column='review_point', by='category', rot=45, figsize=(10,8));
```

The first and third visualizations did not change by much, but we can see how much the outliers affect the price to category box-plot comparison.

# Train/Val/Test Split
*Note: here on out I will be primarily using the created scotch_1000 dataset, except where otherwise noted.*

In [35]:  
```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(scotch_1000, train_size=0.80, test_size=0.20, stratify=scotch_1000['satisfaction_rating'], random_state=55)

train.shape, test.shape
```
Out[35]:  
```python
((1685, 7), (422, 7))
```

Here, I opted to split the data into just training and test datasets. I will be using cross validation in my pipeline, which will provide my validation data.

In [36]:  
```python
train['satisfaction_rating'].value_counts(normalize=True)
```  
Out[36]:  
```python
Good         0.282493
Fair         0.271810
Excellent    0.228487
Poor         0.217211
Name: satisfaction_rating, dtype: float64
```

The satisfaction rating value_counts (normalized) are similar to what was represented by the original dataset. There is some difference, which is to be expected, but it is marginally skewed towards more **Good**, **Fair**, and **Poor** ratings and less **Excellent** ratings.

If we were to look at our training dataset alone, our majority class does not change it's class, but it does increase to 28.25%.

In [0]:  
```python
# train/val split for use with Logistic Regression and Random Forest without
# cross validation
trainLR, val = train_test_split(train, train_size=0.80, test_size=0.20, stratify=train['satisfaction_rating'], random_state=55)
```

## Feature Engineering
Two common characteristics of whiskies that are often sought are the alcohol by volume, or ABV, and whether or not the whisky has an age statement. Neither of these two are represented by a feature on their own, but rather are included in the name feature of the original dataset.

I extracted both of these characteristics and put them into thier own features. While ABV is required by law to be on the label (and is therefore included in each entry), age statements are not. Furthermore, not all whiskies have an identifiable age to state. In the cases where age statement was missing, I gave it a value of "No Age Statement".

In [0]:  
```python
def wrangle(X):
  ''' Wrangle train and test sets in the same way'''
  
  # Prevent SettingWithCopyWarning
  X = X.copy()
  
  # Currency is the same value for every entry, giving us no actionable information.
  # Drop review_point to prevent data leakage with the satisfaction_rating feature
  X=X.drop(columns=['currency', 'review_point'])
  
  # Extract the ABV from the name feature and put it into a new feature
  X['alcohol_by_volume']=X['name'].str.extract(pat = '([0-9][0-9.]+%)')
  X['alcohol_by_volume'] = X['alcohol_by_volume'].str.strip('%').astype(float)
  
  # Extract from name feature if there is an age statement
  X['age']=X['name'].str.extract(pat = '([0-9][0-9] year)')
  X['age']=X['age'].fillna('No Age Statement')
  
  # Return wrangled dataframe
  return X
  
train = wrangle(train)
test = wrangle(test)
trainLR = wrangle(trainLR)
val = wrangle(val)
```

In [39]:  
```python
# The target is the satisfaction_rating feature
target = 'satisfaction_rating'

# Get a dataframe with all train features except target
train_features = train.drop(columns=[target])

# Get a list of numeric features
numeric_features = train_features.select_dtypes(include='number').columns.tolist()

# Get a series with the cardinality of nonnumeric features, then a list of all
# with a cardinality <=225
cardinality = train_features.select_dtypes(exclude='number').nunique()
categorical_features = cardinality[cardinality<=225].index.tolist()

# Combine lists
features = numeric_features + categorical_features
print(features)
```
Out [39]  
```python
['price', 'alcohol_by_volume', 'category', 'age']
```

In [0]:
```python
# Arrange data into X feature matrix and y target vector
X_train = train[features]
X_test = test[features]
y_train=train[target]
y_test=test[target]

X_trainLR=trainLR[features]
X_val=val[features]
y_trainLR=trainLR[target]
y_val=val[target]
```

In [41]:
```python
X_train.shape
```
Out[41]:
```python
(1685, 4)
```

### Permutation Importance
Even though I only have four features, I decided to run a Permutation Importance as a just-in-case sort of thing.

In [42]:
```python
# transform train data for use with eli5 library's PermutationImportance
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

transformers = make_pipeline(
    ce.ordinal.OrdinalEncoder(),
    SimpleImputer(strategy='median')
)
X_trainLR_transformed = transformers.fit_transform(X_trainLR)
X_val_transformed = transformers.transform(X_val)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_trainLR_transformed, y_trainLR)
```
Out[42]:
```python
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
```

In [43]:
```python
# import, instantiate, and fit permuter
import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    model,
    scoring='accuracy',
    random_state=42
)

permuter.fit(X_val_transformed, y_val)
```
Out[43]:
```python
PermutationImportance(cv='prefit',
                      estimator=RandomForestClassifier(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       class_weight=None,
                                                       criterion='gini',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=-1,
                                                       oob_score=False,
                                                       random_state=42,
                                                       verbose=0,
                                                       warm_start=False),
                      n_iter=5, random_state=42, refit=True,
                      scoring='accuracy')
```
In [111]:
```python
feature_names = X_val.columns.tolist()
eli5.show_weights(permuter, top=None, feature_names=feature_names)
```
Out[111]:
```python
Weight	Feature
0.0623 ± 0.0168	price
0.0552 ± 0.0245	alcohol_by_volume
0.0344 ± 0.0233	age
0.0071 ± 0.0264	category
```

| **Weight** | **Feature** |
| --- | --- |
| 0.0623 ± 0.0168 | price |
| 0.0552 ± 0.0245 | alcohol_by_volume |
| 0.0344 ± 0.0233 | age |
| 0.0071 ± 0.0264 | category |

# Logistic Regression
My linear model utilizes Logistic Regression to predict the satisfaction rating of the test observations. Unlinke my later models, I needed to split the training data into training and validation datasets. I chose to split these randomly at 80%/20% train/validation.

## Model Pipeline: Logistic Regression
In [46]:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LR_pipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True),
                         SimpleImputer(), StandardScaler(),
                        LogisticRegression(multi_class='auto', solver='lbfgs', n_jobs=-1))

LR_pipe.fit(X_trainLR, y_trainLR)
print(f'Train accuracy: {LR_pipe.score(X_trainLR, y_trainLR)}')
print(f'Validation accuracy: {LR_pipe.score(X_val, y_val)}')
```
Out [46]:  
Train accuracy: 0.3879821958456973
Validation accuracy: 0.31750741839762614

## Logistic Regression Evaluation
In [47]:
```python
from sklearn.metrics import accuracy_score

y_pred_LR = LR_pipe.predict(X_test)
print(f'Prediction accuracy: {accuracy_score(y_test, y_pred_LR)}')
```
Out [47]:  
Prediction accuracy: 0.3033175355450237

In [48]:
```python
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(LR_pipe, X_test, y_test, normalize='true',
                      xticks_rotation='vertical', cmap='Blues');
```

In [49]:
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_LR))
```
Out [49]:
```python
              precision    recall  f1-score   support

   Excellent       0.40      0.37      0.38        97
        Fair       0.30      0.44      0.36       115
        Good       0.26      0.27      0.26       119
        Poor       0.26      0.10      0.14        91

    accuracy                           0.30       422
   macro avg       0.30      0.30      0.29       422
weighted avg       0.30      0.30      0.29       422
```

# Random Forest Classification with Cross Validation
For my tree-based model, I decided to utilize a Random Forest with Cross Validation. This classification model is ideal for this useage due to the multi-class nature of the classification. It also allows for a more accurate model compared to a single Decision Tree model.

Because of the use of cross validation, the training dataset did not need to be split into train and validation datasets.

## Model Pipeline: Random Forest with Cross Validation
In [50]:
```python
# Make a pipeline for the model
from sklearn.model_selection import cross_val_score

CVRF_pipe = make_pipeline(ce.ordinal.OrdinalEncoder(),
                          SimpleImputer(strategy='median'),
                          RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=55)
                          )

k=18
scores = cross_val_score(CVRF_pipe, X_train, y_train, cv=k, scoring='accuracy')
print(f'Accuracy for {k} folds:', scores)
```
Out [50]:  
Accuracy for 18 folds: [0.25531915 0.30851064 0.36170213 0.30851064 0.32978723 0.39361702
 0.34042553 0.38297872 0.38297872 0.38297872 0.34042553 0.31182796
 0.30107527 0.3655914  0.2688172  0.34408602 0.35483871 0.32258065]

In [51]:
```python
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = { 
    'simpleimputer__strategy': ['mean', 'median'], 
    'randomforestclassifier__n_estimators': randint(50, 500), 
    'randomforestclassifier__max_depth': [5, 10, 15, 20, None], 
    'randomforestclassifier__max_features': uniform(0, 1), 
}

search = RandomizedSearchCV(
    CVRF_pipe, 
    param_distributions=param_distributions, 
    n_iter=25, 
    cv=5, 
    scoring='accuracy', 
    verbose=10, 
    return_train_score=True, 
    n_jobs=-1
)

search.fit(X_train, y_train);
```

## Random Forest with CV Evaluation
In [53]:
```python
CV_best_pipe = search.best_estimator_

y_pred_RF = CV_best_pipe.predict(X_test)
print(f'Validation accuracy for {k} folds:', scores)
print(f'Prediction accuracy: {accuracy_score(y_test, y_pred_RF)}')
```
Out [53]:  
Validation accuracy for 18 folds: [0.25531915 0.30851064 0.36170213 0.30851064 0.32978723 0.39361702
 0.34042553 0.38297872 0.38297872 0.38297872 0.34042553 0.31182796
 0.30107527 0.3655914  0.2688172  0.34408602 0.35483871 0.32258065]
Prediction accuracy: 0.3459715639810427

In [54]:
```python
plot_confusion_matrix(CV_best_pipe, X_test, y_test, normalize='true',
                      xticks_rotation='vertical', cmap='Blues');
```

In [55]:
```python
print(classification_report (y_test, y_pred_RF))
```
Out [55]:  
```python
              precision    recall  f1-score   support

   Excellent       0.45      0.45      0.45        97
        Fair       0.33      0.39      0.36       115
        Good       0.29      0.34      0.32       119
        Poor       0.32      0.18      0.23        91

    accuracy                           0.35       422
   macro avg       0.35      0.34      0.34       422
weighted avg       0.35      0.35      0.34       422
```

In [56]:
```python
# Get feature importances
rf = CV_best_pipe.named_steps['randomforestclassifier']
importances = pd.Series(rf.feature_importances_, X_train.columns)

# Plot feature importances
%matplotlib inline

n = 20
plt.figure(figsize=(10,n/2))
plt.title(f'Top {n} features')
importances.sort_values()[-n:].plot.barh(color='grey');
```

# Random Forest Classification using Train/Val split
I was curious to see if the Cross Validation performed better than my own Train/Val split, so I also ran the model this way.

# Model Pipeline: Random Forest Train/Val Split
In [57]:
```python
# first, run the pipeline on the train/val split
RF_pipe = make_pipeline(ce.ordinal.OrdinalEncoder(),
                        SimpleImputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=55)
                        )

RF_pipe.fit(X_trainLR, y_trainLR)
print(f'Train accuracy: {RF_pipe.score(X_trainLR, y_trainLR)}')
print(f'Validation accuracy: {RF_pipe.score(X_val, y_val)}')
```
Out [57]:  
Train accuracy: 0.913946587537092
Validation accuracy: 0.32344213649851633

In [58]:
```python
# then fit it back to the full training data set
RF_pipe = make_pipeline(ce.ordinal.OrdinalEncoder(),
                        SimpleImputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=55)
                        )

RF_pipe.fit(X_train, y_train)
print(f'Train (full dataset) accuracy: {RF_pipe.score(X_train, y_train)}')
```
Out [58]:  
Train (full dataset) accuracy: 0.9014836795252226

## Random Forest Train/Val Split Evaluation
In [59]:
```python
y_pred_RFSplit = RF_pipe.predict(X_test)
print(f'Prediction accuracy: {accuracy_score(y_test, y_pred_RFSplit)}')
```
Out [59]:  
Prediction accuracy: 0.3175355450236967

In [60]:
```python
plot_confusion_matrix(RF_pipe, X_test, y_test, normalize='true',
                      xticks_rotation='vertical', cmap='Blues');
```

In [61]:
```python
print(classification_report(y_test, y_pred_RFSplit))
```
Out [61]:  
```python
              precision    recall  f1-score   support

   Excellent       0.38      0.39      0.39        97
        Fair       0.31      0.27      0.29       115
        Good       0.27      0.33      0.30       119
        Poor       0.32      0.29      0.30        91

    accuracy                           0.32       422
   macro avg       0.32      0.32      0.32       422
weighted avg       0.32      0.32      0.32       422
```

# Partial Dependency Plots
In [0]:
```python
from pdpbox.pdp import pdp_interact, pdp_interact_plot, pdp_isolate, pdp_plot
```
In [0]:
```python
X_test=X_test.fillna(method='ffill')
```

In [0]:
```python
features = ['price','alcohol_by_volume']

interact = pdp_interact(
    model=CV_best_pipe,
    dataset=X_test,
    model_features=X_test.columns,
    features=features
)
```

In [95]:
```python
pdp_interact_plot(interact, plot_type='grid', feature_names=features);
findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
```

In [107]:
```python
feature = 'price'

isolated = pdp_isolate(
    model=CV_best_pipe,
    dataset=X_test,
    model_features=X_test.columns,
    feature=feature
)

pdp_plot(isolated, feature_name=feature, plot_lines=True);
```

In [108]:
```python
feature = 'alcohol_by_volume'

isolated = pdp_isolate(
    model=CV_best_pipe,
    dataset=X_test,
    model_features=X_test.columns,
    feature=feature
)

pdp_plot(isolated, feature_name=feature, plot_lines=True);
```
