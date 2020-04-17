---
layout: post
title: First Analysis On Whiskies
subtitle: Interesting insights into high ranking whiskies
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
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
