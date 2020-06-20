---
layout: post
title: First Analysis On Whiskies
subtitle: Interesting insights into high ranking whiskies
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [school project, whisky, EDA]
comments: true
---

*If you would like to follow along, you can find the python notebook on my github [here](https://github.com/StevenWestmoreland/Whisky_Data_Projects/blob/master/FirstDataAnalysisOfWhiskies.ipynb).*

## About the Data Source
The dataset being explored here is from [whiskyanalysis.com](https://whiskyanalysis.com/index.php/database/). It covers the Cost, Country, Type, and Class of an assortment of whiskies as well as the Cluster and Super Cluster of thier flavor profiles. It also includes the normalized Meta Critic score (as defined on his [methodology page](https://whiskyanalysis.com/index.php/methodology-introduction/methodology-metacritic-score-construction/)), the Standard Deviation of the normalized score, and the number of reviews that contributed to that score.

As stated on the webpage for the data itself: "The ultimate goal of this site is really one of arbitrage – taking advantage of the price differential between two or more whiskies of varying quality. You do this by comparing whiskies prices to whisky quality, among whiskies of similar flavour. The goal of this site is to objectively define those two key characteristics for you." This is handled via the normalization of Meta Critic scores for each whisky (as defined above), as well as providing key qualities for each whisky.

Class, Country, and Type give an idea towards overall profile qualities, while Cluster and Super Cluster are specific flavor profiles.

Cost values are representative of a range of costs in CAD. These costs can vary depending on the vendor, but generally can be found within the price range listed. The Cost value ranges are <$30, $30-$50, $50-$70, $70-$125, $125-$300, and >$300. These, once cleaned, are represented with a letter rating from A to F, where A is the most expensive and F is the least.

For more information on variables and their meanings, see the [how to read the database page](https://whiskyanalysis.com/index.php/interesting-correlations/how-to-read-the-database/) on the source site.

## Variables Kept and Variables Discarded
While there is a lot of good data in this dataset, I decided to focus on only a few of the variables presented. I wanted to make most of my comparisons against the Meta Critic ratings of the whiskies, but I was less interested in their Name, Number of Reviews, or Standard Deviation. I determined that Cost and Country are very valuable for such a comparison, as are Class and Type; these four variables I kept. Cluster was also usable, but Super Cluster seemed superflous for what I was doing seeing as how it was just a combination of flavor-adjacent Clusters.

Variables Kept: Meta Critic, Cost, Class, Cluster, Country, Type

Variables Discarded: Whisky, STDEV, #, Super Cluster

```python
# import libraries and read in dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

whisky_df = pd.read_csv("Whisky_Analysis.csv")
```

In []:
```python
# Check for missing values
whisky_df.isnull().sum()
```
Out []:
```python
Whisky             0
Meta Critic        0
STDEV              0
#                  0
Cost               2
Class              0
Super Cluster    570
Cluster          302
Country            0
Type               0
dtype: int64
```

In [ ]:
```python
whisky_df['Cost'].isnull().sum()
```
Out[ ]:
```python
2
```

# Null Values
There is a significant number of missing values for the Cluster variable. While I could have dropped these values, I instead decided to give them a new result of "U" for "Unknown." This will let me still use that data for other variable considerations, such as comparing Meta Critic to Cost.

We also have two missing Cost values, but these can simply be ignored or dropped. I opted to keep them in just in case they became useful.

In [ ]:
```python
# Replace all NaN values for Cluster column with "U" for unknown
whisky_df["Cluster"].fillna("U", inplace = True)
whisky_df.head()
```
Out[ ]:
```python
Whisky	Meta Critic	STDEV	#	Cost	Class	Super Cluster	Cluster	Country	Type
0	Ledaig 42yo Dusgadh	9.48	0.23	3	$$$$$+	SingleMalt-like	ABC	C	Scotland	Malt
1	Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...	9.42	0.22	4	$$$$$+	SingleMalt-like	ABC	C	Scotland	Malt
2	Glenfarclas 40yo	9.30	0.27	17	$$$$$+	SingleMalt-like	ABC	A	Scotland	Malt
3	Aberlour A'Bunadh (Batch 56)	9.25	0.24	3	$$$$	SingleMalt-like	ABC	A	Scotland	Malt
4	Glengoyne 25yo	9.24	0.22	21	$$$$$+	SingleMalt-like	ABC	A	Scotland	Malt
```

In [ ]:
# Check that the number of "U" values are equal to the previous 
# number of .isnull().sum() results for Cluster
whisky_df['Cluster'].value_counts().sort_index()
Out[ ]:
A      99
B      50
C     218
E     224
F      42
G     142
H      72
I     177
J     122
R0     28
R1     67
R2     67
R3     46
R4     60
U     302
Name: Cluster, dtype: int64
In [ ]:
# Double checking is never bad. Check new isnull().sum()
whisky_df['Cluster'].isnull().sum()
Out[ ]:
0
Restructuring the Data
Some of the data needed to be changed into an easier form to work with. Cost was an ideal candidate for this due to the unwieldyness of the "$" values it used. I opted to use a numerical "grade" instead, where "A" represented the most expensive value band (>300 CAD) and "F" represented the lowest (<30 CAD).

Country was another that could use some restructuring. There are significantly more Scotland entries than even the runner up, USA. In fact, anything below Ireland was almost neglible in comparison. Because of this, and considering the domain knowledge of general expected Country types, I combined everything from Sweden to France under the new entry "Other".

In [ ]:
# Checking the value counts for the most prime candidate to be restructured, Cost
whisky_df['Cost'].value_counts().sort_index()
Out[ ]:
$          87
$$        201
$$$       324
$$$$      580
$$$$$     343
$$$$$+    179
Name: Cost, dtype: int64
In [ ]:
# Checking the number of entries per country.
whisky_df['Country'].value_counts()
Out[ ]:
Scotland        928
USA             292
Canada          189
Ireland          81
Japan            75
Sweden           59
India            38
Taiwan           16
Wales             8
Switzerland       7
Finland           7
Tasmania          5
Netherlands       4
South Africa      3
England           2
Belgium           1
France            1
Name: Country, dtype: int64
In [ ]:
# Remap "Cost" to a more usable form
whisky_df["Cost Rank"] = whisky_df['Cost'].map({"$":"F", "$$":"E", "$$$":"D",
                                                "$$$$":"C", "$$$$$":"B",
                                                "$$$$$+":"A"})
In [ ]:
# Remap all other countries into one
whisky_df['Country Condensed'] = whisky_df['Country'].map({"Scotland":"Scotland",
                                                           "USA":"USA", "Canada":"Canada",
                                                           "Ireland":"Ireland", "Japan":"Japan",
                                                           "Sweden":"Other", "India":"Other",
                                                           "Taiwan":"Other", "Wales": "Other",
                                                           "Switzerland":"Other",
                                                           "Finland":"Other", "Tasmania":"Other",
                                                           "Netherlands": "Other",
                                                           "South Africa":"Other",
                                                           "England":"Other", "Belgium":"Other",
                                                           "France":"Other"})
In [ ]:
whisky_df.head()
Out[ ]:
Whisky	Meta Critic	STDEV	#	Cost	Class	Super Cluster	Cluster	Country	Type	Cost Rank	Country Condensed
0	Ledaig 42yo Dusgadh	9.48	0.23	3	$$$$$+	SingleMalt-like	ABC	C	Scotland	Malt	A	Scotland
1	Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...	9.42	0.22	4	$$$$$+	SingleMalt-like	ABC	C	Scotland	Malt	A	Scotland
2	Glenfarclas 40yo	9.30	0.27	17	$$$$$+	SingleMalt-like	ABC	A	Scotland	Malt	A	Scotland
3	Aberlour A'Bunadh (Batch 56)	9.25	0.24	3	$$$$	SingleMalt-like	ABC	A	Scotland	Malt	C	Scotland
4	Glengoyne 25yo	9.24	0.22	21	$$$$$+	SingleMalt-like	ABC	A	Scotland	Malt	A	Scotland
In [ ]:
whisky_df['Country Condensed'].value_counts()
Out[ ]:
Scotland    928
USA         292
Canada      189
Other       151
Ireland      81
Japan        75
Name: Country Condensed, dtype: int64
In [ ]:
whisky_df['Cost Rank'].value_counts().sort_index()
Out[ ]:
A    179
B    343
C    580
D    324
E    201
F     87
Name: Cost Rank, dtype: int64
Determining Useful Data Comparisons
One of the most important key skills of a Data Scientist is the ability to determine what data comparisons are ideal. And key to ths is understanding what the question or problem these comparisons are answering.

In our case, we are looking at the commonalities between the highest ranking whiskies. 'Meta Critic', the normalized score for each whisky, becomes the variable that we will compare everything else to.

In [ ]:
pd.crosstab(whisky_df['Meta Critic'], whisky_df['Country Condensed'])
Out[ ]:
Country Condensed	Canada	Ireland	Japan	Other	Scotland	USA
Meta Critic						
6.47	0	0	0	1	0	0
6.60	0	0	1	0	0	0
6.80	1	0	0	0	0	0
6.88	0	0	0	0	1	0
6.91	0	0	0	0	1	0
...	...	...	...	...	...	...
9.44	1	0	0	0	1	0
9.48	0	0	0	0	2	0
9.50	0	0	0	0	1	0
9.51	0	0	0	0	1	0
9.54	1	0	0	0	0	0
208 rows × 6 columns

In [ ]:
whisky_df.groupby(["Cost Rank"])["Meta Critic"].mean()
Out[ ]:
Cost Rank
A    8.961229
B    8.744927
C    8.618138
D    8.386883
E    8.224677
F    7.881609
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df["Meta Critic"].groupby(whisky_df["Cost Rank"]).mean()
Out[ ]:
Cost Rank
A    8.961229
B    8.744927
C    8.618138
D    8.386883
E    8.224677
F    7.881609
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df.groupby(["Cluster"])["Meta Critic"].mean()
Out[ ]:
Cluster
A     8.798788
B     8.588400
C     8.646697
E     8.528304
F     8.456190
G     8.436197
H     8.320833
I     8.713672
J     8.782213
R0    8.565357
R1    8.607761
R2    8.579851
R3    8.591522
R4    8.612833
U     8.313411
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df.groupby(["Country"])["Meta Critic"].mean()
Out[ ]:
Country
Belgium         8.310000
Canada          8.411005
England         8.210000
Finland         8.488571
France          7.290000
India           8.790526
Ireland         8.415556
Japan           8.528533
Netherlands     8.347500
Scotland        8.581940
South Africa    8.186667
Sweden          8.560847
Switzerland     7.748571
Taiwan          8.785625
Tasmania        8.534000
USA             8.595788
Wales           8.101250
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df.groupby(['Country Condensed'])["Meta Critic"].mean()
Out[ ]:
Country Condensed
Canada      8.411005
Ireland     8.415556
Japan       8.528533
Other       8.548411
Scotland    8.581940
USA         8.595788
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df.groupby(['Type'])["Meta Critic"].mean()
Out[ ]:
Type
Barley       8.550000
Blend        8.312712
Bourbon      8.598039
Flavoured    7.360000
Grain        8.287143
Malt         8.603295
Rye          8.645696
Wheat        8.500000
Whiskey      8.310000
Name: Meta Critic, dtype: float64
In [ ]:
whisky_df['Meta Critic'].describe()
Out[ ]:
count    1716.000000
mean        8.552331
std         0.396664
min         6.470000
25%         8.330000
50%         8.600000
75%         8.830000
max         9.540000
Name: Meta Critic, dtype: float64
In [ ]:
# Let's see what a scatterplot of the two looks like
plt.scatter(whisky_df["Country Condensed"], whisky_df['Meta Critic']);

In [ ]:
# Scatter is too difficult to determine anything specific. Boxplot might be better
whisky_df.boxplot(column='Meta Critic', by='Country Condensed');

In [ ]:
# Again, with different variables
whisky_df.boxplot(column="Meta Critic", by='Cost Rank');

In [ ]:
whisky_df.boxplot(column='Meta Critic', by='Cluster');

In [ ]:
whisky_df.boxplot(column='Meta Critic', by='Class');

In [ ]:
whisky_df.boxplot(column='Meta Critic', by='Type');

In [ ]:
# Are their any useful comparisons not utilizing 'Meta Critic'?
pd.crosstab(whisky_df['Cluster'], whisky_df['Type'])
Out[ ]:
Type	Barley	Blend	Bourbon	Flavoured	Grain	Malt	Rye	Wheat	Whiskey
Cluster									
A	0	0	0	0	0	99	0	0	0
B	0	0	0	0	0	50	0	0	0
C	0	3	0	0	0	215	0	0	0
E	0	10	0	0	0	214	0	0	0
F	0	1	0	0	0	41	0	0	0
G	0	7	0	0	0	135	0	0	0
H	0	0	0	0	0	72	0	0	0
I	0	0	0	0	0	177	0	0	0
J	0	0	0	0	0	122	0	0	0
R0	0	1	23	0	2	0	0	1	1
R1	0	0	67	0	0	0	0	0	0
R2	0	0	67	0	0	0	0	0	0
R3	0	0	46	0	0	0	0	0	0
R4	0	2	0	0	0	0	58	0	0
U	1	271	1	1	5	1	21	1	0
In [ ]:
# What does this look like as a stacked bar chart?
flavor_type = pd.crosstab(whisky_df['Cluster'], whisky_df['Type'])
flavor_type.plot(kind='barh', stacked=True);

In [ ]:
# This next one is not really useful information because it is literally 
# defined by the Cluster code definitions already
pd.crosstab(whisky_df['Cluster'], whisky_df['Class'])
Out[ ]:
Class	Bourbon-like	Rye-like	Scotch-like	SingleMalt-like
Cluster				
A	0	0	0	99
B	0	0	0	50
C	0	0	0	218
E	0	0	0	224
F	0	0	0	42
G	0	0	0	142
H	0	0	0	72
I	0	0	0	177
J	0	0	0	122
R0	28	0	0	0
R1	67	0	0	0
R2	67	0	0	0
R3	46	0	0	0
R4	0	60	0	0
U	12	155	135	0
In [ ]:
# Similar to previous, but more useful
pd.crosstab(whisky_df['Type'], whisky_df['Class'])
Out[ ]:
Class	Bourbon-like	Rye-like	Scotch-like	SingleMalt-like
Type				
Barley	0	1	0	0
Blend	11	134	129	21
Bourbon	204	0	0	0
Flavoured	1	0	0	0
Grain	2	0	5	0
Malt	0	0	1	1125
Rye	0	79	0	0
Wheat	1	1	0	0
Whiskey	1	0	0	0
In [ ]:
# Lets see this as a stacked bar too
# The Malt value skews the graph and makes it hard to read 
type_class = pd.crosstab(whisky_df['Type'], whisky_df['Class'])
type_class.plot(kind='barh', stacked=True);

In [ ]:
pd.crosstab(whisky_df['Cluster'], whisky_df['Country Condensed'])
Out[ ]:
Country Condensed	Canada	Ireland	Japan	Other	Scotland	USA
Cluster						
A	0	1	2	7	89	0
B	3	3	1	8	34	1
C	1	12	5	29	170	1
E	9	16	12	39	144	4
F	1	1	5	4	31	0
G	15	6	18	13	88	2
H	1	4	6	11	50	0
I	5	1	11	16	142	2
J	0	1	1	17	103	0
R0	1	0	0	0	0	27
R1	0	0	0	0	0	67
R2	0	0	0	0	0	67
R3	0	0	0	0	0	46
R4	0	0	0	0	0	60
U	153	36	14	7	77	15
In [ ]:
type_class = pd.crosstab(whisky_df['Country Condensed'], whisky_df['Cluster'])
type_class.plot(kind='barh', stacked=True);

Final Notes
What I want to do better
I wanted to try looking at correlation heat maps utilizing dython, but I ran out of time.

I wasn't able to clean up the visualizations as much as I would like.

Takeaways
Scotch Single Malts seem to be the preffered type of whisky for the data gatherer. This meant the results were weighted towards these categories.

Country of Origin has little effect on the overall score of a whisky.

The Cost of a whisky seems to have a proportional correlation to the score of a whisky. Could be that higher costing whiskies are objectively better, but could also partially be the consumer rationalizing the higher expenditure as "better". Would be interesting to try this with a blind group who did not know the cost of the whiskies they were trying.

Scotland has a more homogenous flavor profile than I initially would have thought. In retrospect, it makes sense when considering the breadth of their sub-regions.
