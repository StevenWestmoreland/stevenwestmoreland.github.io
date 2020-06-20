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


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "DSPT6_Unit1_Build_StevenWestmoreland",
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyOK1DtNysGa8YUFvn2oSczR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/StevenWestmoreland/Whisky_Data_Projects/blob/master/DSPT6_Unit1_Build_StevenWestmoreland.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sJqSnfOBUlbu",
        "colab_type": "text"
      },
      "source": [
        "# Qualities of Top Whiskies\n",
        "\n",
        "The dataset being explored here is from https://whiskyanalysis.com/index.php/database/. It covers the Cost, Country, Type, and Class of an assortment of whiskies as well as the Cluster and Super Cluster of thier flavor profiles. It also includes the normalized Meta Critic score (as defined at https://whiskyanalysis.com/index.php/methodology-introduction/methodology-metacritic-score-construction/), the Standard Deviation of the normalized score, and the number of reviews that contributed to that score. \n",
        "\n",
        "### About the Data Source\n",
        "\n",
        "As stated on the webpage for the data itself: \"The ultimate goal of this site is really one of arbitrage – taking advantage of the price differential between two or more whiskies of varying quality. You do this by comparing whiskies prices to whisky quality, among whiskies of similar flavour. The goal of this site is to objectively define those two key characteristics for you.\" \n",
        "This is handled via the normalization of Meta Critic scores for each whisky (as defined above), as well as providing key qualities for each whisky. \n",
        "\n",
        "Class, Country, and Type give an idea towards overall profile qualities, while Cluster and Super Cluster are specific flavor profiles.\n",
        "\n",
        "Cost values are representative of a range of costs in CAD. These costs can vary depending on the vendor, but generally can be found for these amounts. The Cost value ranges are <30 CAD, 30-50 CAD, 50-70 CAD, 70-125 CAD, 125-300 CAD, and >300 CAD. These are represented, within the cleaned column, a letter rating from A to F, where A is the most expensive and F is the least.\n",
        "\n",
        "For more information on variables and their meanings, see https://whiskyanalysis.com/index.php/interesting-correlations/how-to-read-the-database/.\n",
        "\n",
        "### Variables Kept and Variables Discarded\n",
        "\n",
        "While there is a lot of good data in this dataset, I decided to focus on only a few of the variables presented. I wanted to make most of my comparisons against the Meta Critic ratings of the whiskies, but I was less interested in their Name, Number of Reviews, or Standard Deviation. \n",
        "\n",
        "Cost and Country are very valuable for such a comparison, as are Class and Type; these four variables I kept. Cluster was also usable, but Super Cluster seemed superflous for what I was doing. \n",
        "\n",
        "Variables Kept: Meta Critic, Cost, Class, Cluster, Country, Type\n",
        "\n",
        "Variables Discarded: Whisky, STDEV, #, Super Cluster"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ogziIVuOPqLo",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "73e72ceb-df03-413f-d9c1-033b0ea26afd"
      },
      "source": [
        "# import libraries and read in dataset\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "whisky_df = pd.read_csv(\"Whisky_Analysis.csv\")\n",
        "whisky_df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.\n",
            "  import pandas.util.testing as tm\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Whisky</th>\n",
              "      <th>Meta Critic</th>\n",
              "      <th>STDEV</th>\n",
              "      <th>#</th>\n",
              "      <th>Cost</th>\n",
              "      <th>Class</th>\n",
              "      <th>Super Cluster</th>\n",
              "      <th>Cluster</th>\n",
              "      <th>Country</th>\n",
              "      <th>Type</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Ledaig 42yo Dusgadh</td>\n",
              "      <td>9.48</td>\n",
              "      <td>0.23</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...</td>\n",
              "      <td>9.42</td>\n",
              "      <td>0.22</td>\n",
              "      <td>4</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Glenfarclas 40yo</td>\n",
              "      <td>9.30</td>\n",
              "      <td>0.27</td>\n",
              "      <td>17</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Aberlour A'Bunadh (Batch 56)</td>\n",
              "      <td>9.25</td>\n",
              "      <td>0.24</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Glengoyne 25yo</td>\n",
              "      <td>9.24</td>\n",
              "      <td>0.22</td>\n",
              "      <td>21</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                              Whisky  ...  Type\n",
              "0                                Ledaig 42yo Dusgadh  ...  Malt\n",
              "1  Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...  ...  Malt\n",
              "2                                   Glenfarclas 40yo  ...  Malt\n",
              "3                       Aberlour A'Bunadh (Batch 56)  ...  Malt\n",
              "4                                     Glengoyne 25yo  ...  Malt\n",
              "\n",
              "[5 rows x 10 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 1
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yGtib_ReSUi6",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "bc8601a0-cb73-416d-904a-dd60c6db8007"
      },
      "source": [
        "# Check for missing values\n",
        "whisky_df.isnull().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Whisky             0\n",
              "Meta Critic        0\n",
              "STDEV              0\n",
              "#                  0\n",
              "Cost               2\n",
              "Class              0\n",
              "Super Cluster    570\n",
              "Cluster          302\n",
              "Country            0\n",
              "Type               0\n",
              "dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aKl8t_y0ld1G",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "2d7fca75-6d7f-46ba-eba7-450279be28a0"
      },
      "source": [
        "whisky_df['Cost'].isnull().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "y4TdGz0YjidM",
        "colab_type": "text"
      },
      "source": [
        "# Null Values\n",
        "\n",
        "There is a significant number of missing values for the Cluster variable. While I could have dropped these values, I instead decided to give them a new result of \"U\" for \"Unknown.\" This will let me still use that data for other variable considerations, such as comparing Meta Critic to Cost.\n",
        "\n",
        "We also have two missing Cost values, but these can simply be ignored or dropped. I opted to keep them in just in case they became useful. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zi_N78rZrdTr",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "0badf345-febe-44f5-8bf5-e89f46575974"
      },
      "source": [
        "# Replace all NaN values for Cluster column with \"U\" for unknown\n",
        "whisky_df[\"Cluster\"].fillna(\"U\", inplace = True)\n",
        "whisky_df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Whisky</th>\n",
              "      <th>Meta Critic</th>\n",
              "      <th>STDEV</th>\n",
              "      <th>#</th>\n",
              "      <th>Cost</th>\n",
              "      <th>Class</th>\n",
              "      <th>Super Cluster</th>\n",
              "      <th>Cluster</th>\n",
              "      <th>Country</th>\n",
              "      <th>Type</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Ledaig 42yo Dusgadh</td>\n",
              "      <td>9.48</td>\n",
              "      <td>0.23</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...</td>\n",
              "      <td>9.42</td>\n",
              "      <td>0.22</td>\n",
              "      <td>4</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Glenfarclas 40yo</td>\n",
              "      <td>9.30</td>\n",
              "      <td>0.27</td>\n",
              "      <td>17</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Aberlour A'Bunadh (Batch 56)</td>\n",
              "      <td>9.25</td>\n",
              "      <td>0.24</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Glengoyne 25yo</td>\n",
              "      <td>9.24</td>\n",
              "      <td>0.22</td>\n",
              "      <td>21</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                              Whisky  ...  Type\n",
              "0                                Ledaig 42yo Dusgadh  ...  Malt\n",
              "1  Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...  ...  Malt\n",
              "2                                   Glenfarclas 40yo  ...  Malt\n",
              "3                       Aberlour A'Bunadh (Batch 56)  ...  Malt\n",
              "4                                     Glengoyne 25yo  ...  Malt\n",
              "\n",
              "[5 rows x 10 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UT33PhSsr8_S",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 289
        },
        "outputId": "8310e8fc-9f0d-4e4f-f6dc-842380a9cc11"
      },
      "source": [
        "# Check that the number of \"U\" values are equal to the previous \n",
        "# number of .isnull().sum() results for Cluster\n",
        "whisky_df['Cluster'].value_counts().sort_index()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "A      99\n",
              "B      50\n",
              "C     218\n",
              "E     224\n",
              "F      42\n",
              "G     142\n",
              "H      72\n",
              "I     177\n",
              "J     122\n",
              "R0     28\n",
              "R1     67\n",
              "R2     67\n",
              "R3     46\n",
              "R4     60\n",
              "U     302\n",
              "Name: Cluster, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mEdzGp5YsE6t",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "c7398c83-f935-4e4a-820e-c4f1cbcfa7cb"
      },
      "source": [
        "# Double checking is never bad. Check new isnull().sum()\n",
        "whisky_df['Cluster'].isnull().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-obYum3uxiDg",
        "colab_type": "text"
      },
      "source": [
        "# Restructuring the Data\n",
        "\n",
        "Some of the data needed to be changed into an easier form to work with. Cost was an ideal candidate for this due to the unwieldyness of the \"$\" values it used. I opted to use a numerical \"grade\" instead, where \"A\" represented the most expensive value band (>300 CAD) and \"F\" represented the lowest (<30 CAD). \n",
        "\n",
        "Country was another that could use some restructuring. There are significantly more Scotland entries than even the runner up, USA. In fact, anything below Ireland was almost neglible in comparison. Because of this, and considering the domain knowledge of general expected Country types, I combined everything from Sweden to France under the new entry \"Other\"."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0_3hk8XJd1KT",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "caa7ea53-cc96-44db-9013-7e4218936965"
      },
      "source": [
        "# Checking the value counts for the most prime candidate to be restructured, Cost\n",
        "whisky_df['Cost'].value_counts().sort_index()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "$          87\n",
              "$$        201\n",
              "$$$       324\n",
              "$$$$      580\n",
              "$$$$$     343\n",
              "$$$$$+    179\n",
              "Name: Cost, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lIvvbW4jyYig",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 323
        },
        "outputId": "adfe4e03-e15f-420c-cd60-a14bc91d7ff4"
      },
      "source": [
        "# Checking the number of entries per country.\n",
        "whisky_df['Country'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Scotland        928\n",
              "USA             292\n",
              "Canada          189\n",
              "Ireland          81\n",
              "Japan            75\n",
              "Sweden           59\n",
              "India            38\n",
              "Taiwan           16\n",
              "Wales             8\n",
              "Switzerland       7\n",
              "Finland           7\n",
              "Tasmania          5\n",
              "Netherlands       4\n",
              "South Africa      3\n",
              "England           2\n",
              "Belgium           1\n",
              "France            1\n",
              "Name: Country, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8eX6uzTKd_bI",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Remap \"Cost\" to a more usable form\n",
        "whisky_df[\"Cost Rank\"] = whisky_df['Cost'].map({\"$\":\"F\", \"$$\":\"E\", \"$$$\":\"D\",\n",
        "                                                \"$$$$\":\"C\", \"$$$$$\":\"B\",\n",
        "                                                \"$$$$$+\":\"A\"})"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Bvutt7tZzEZJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Remap all other countries into one\n",
        "whisky_df['Country Condensed'] = whisky_df['Country'].map({\"Scotland\":\"Scotland\",\n",
        "                                                           \"USA\":\"USA\", \"Canada\":\"Canada\",\n",
        "                                                           \"Ireland\":\"Ireland\", \"Japan\":\"Japan\",\n",
        "                                                           \"Sweden\":\"Other\", \"India\":\"Other\",\n",
        "                                                           \"Taiwan\":\"Other\", \"Wales\": \"Other\",\n",
        "                                                           \"Switzerland\":\"Other\",\n",
        "                                                           \"Finland\":\"Other\", \"Tasmania\":\"Other\",\n",
        "                                                           \"Netherlands\": \"Other\",\n",
        "                                                           \"South Africa\":\"Other\",\n",
        "                                                           \"England\":\"Other\", \"Belgium\":\"Other\",\n",
        "                                                           \"France\":\"Other\"})"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9iqEKgJ5exOf",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "7b03edeb-3b64-4b6e-9033-68826b740419"
      },
      "source": [
        "whisky_df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Whisky</th>\n",
              "      <th>Meta Critic</th>\n",
              "      <th>STDEV</th>\n",
              "      <th>#</th>\n",
              "      <th>Cost</th>\n",
              "      <th>Class</th>\n",
              "      <th>Super Cluster</th>\n",
              "      <th>Cluster</th>\n",
              "      <th>Country</th>\n",
              "      <th>Type</th>\n",
              "      <th>Cost Rank</th>\n",
              "      <th>Country Condensed</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Ledaig 42yo Dusgadh</td>\n",
              "      <td>9.48</td>\n",
              "      <td>0.23</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...</td>\n",
              "      <td>9.42</td>\n",
              "      <td>0.22</td>\n",
              "      <td>4</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Glenfarclas 40yo</td>\n",
              "      <td>9.30</td>\n",
              "      <td>0.27</td>\n",
              "      <td>17</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Aberlour A'Bunadh (Batch 56)</td>\n",
              "      <td>9.25</td>\n",
              "      <td>0.24</td>\n",
              "      <td>3</td>\n",
              "      <td>$$$$</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "      <td>C</td>\n",
              "      <td>Scotland</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Glengoyne 25yo</td>\n",
              "      <td>9.24</td>\n",
              "      <td>0.22</td>\n",
              "      <td>21</td>\n",
              "      <td>$$$$$+</td>\n",
              "      <td>SingleMalt-like</td>\n",
              "      <td>ABC</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "      <td>Malt</td>\n",
              "      <td>A</td>\n",
              "      <td>Scotland</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                              Whisky  ...  Country Condensed\n",
              "0                                Ledaig 42yo Dusgadh  ...           Scotland\n",
              "1  Laphroaig 27yo 57.4% 1980-2007 (OB, 5 Oloroso ...  ...           Scotland\n",
              "2                                   Glenfarclas 40yo  ...           Scotland\n",
              "3                       Aberlour A'Bunadh (Batch 56)  ...           Scotland\n",
              "4                                     Glengoyne 25yo  ...           Scotland\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4Q8_BLJnIBXd",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "5d80692a-f813-4cf1-8de2-2b4e86a7632b"
      },
      "source": [
        "whisky_df['Country Condensed'].value_counts()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Scotland    928\n",
              "USA         292\n",
              "Canada      189\n",
              "Other       151\n",
              "Ireland      81\n",
              "Japan        75\n",
              "Name: Country Condensed, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XMl-YT7WIJFm",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        },
        "outputId": "da9bd936-b6ba-4b46-802a-42d39ccd5691"
      },
      "source": [
        "whisky_df['Cost Rank'].value_counts().sort_index()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "A    179\n",
              "B    343\n",
              "C    580\n",
              "D    324\n",
              "E    201\n",
              "F     87\n",
              "Name: Cost Rank, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kBtOVizKtzJn",
        "colab_type": "text"
      },
      "source": [
        "# Determining Useful Data Comparisons\n",
        "\n",
        "One of the most important key skills of a Data Scientist is the ability to determine what data comparisons are ideal. And key to ths is understanding what the question or problem these comparisons are answering. \n",
        "\n",
        "In our case, we are looking at the commonalities between the highest ranking whiskies. 'Meta Critic', the normalized score for each whisky, becomes the variable that we will compare everything else to. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ewR5BkxXe6-o",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 450
        },
        "outputId": "09bc7b7f-ef6c-4d84-ad37-7e33122090ee"
      },
      "source": [
        "pd.crosstab(whisky_df['Meta Critic'], whisky_df['Country Condensed'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th>Country Condensed</th>\n",
              "      <th>Canada</th>\n",
              "      <th>Ireland</th>\n",
              "      <th>Japan</th>\n",
              "      <th>Other</th>\n",
              "      <th>Scotland</th>\n",
              "      <th>USA</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Meta Critic</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>6.47</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6.60</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6.80</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6.88</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6.91</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9.44</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9.48</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9.50</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9.51</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9.54</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>208 rows × 6 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "Country Condensed  Canada  Ireland  Japan  Other  Scotland  USA\n",
              "Meta Critic                                                    \n",
              "6.47                    0        0      0      1         0    0\n",
              "6.60                    0        0      1      0         0    0\n",
              "6.80                    1        0      0      0         0    0\n",
              "6.88                    0        0      0      0         1    0\n",
              "6.91                    0        0      0      0         1    0\n",
              "...                   ...      ...    ...    ...       ...  ...\n",
              "9.44                    1        0      0      0         1    0\n",
              "9.48                    0        0      0      0         2    0\n",
              "9.50                    0        0      0      0         1    0\n",
              "9.51                    0        0      0      0         1    0\n",
              "9.54                    1        0      0      0         0    0\n",
              "\n",
              "[208 rows x 6 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bVGfCqYbfIzN",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        },
        "outputId": "e287cf05-e26f-4585-a1f1-65afa79fe193"
      },
      "source": [
        "whisky_df.groupby([\"Cost Rank\"])[\"Meta Critic\"].mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Cost Rank\n",
              "A    8.961229\n",
              "B    8.744927\n",
              "C    8.618138\n",
              "D    8.386883\n",
              "E    8.224677\n",
              "F    7.881609\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qOURkQv5fuY2",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        },
        "outputId": "5b001a57-8885-4ce9-df35-4b47db5c2755"
      },
      "source": [
        "whisky_df[\"Meta Critic\"].groupby(whisky_df[\"Cost Rank\"]).mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Cost Rank\n",
              "A    8.961229\n",
              "B    8.744927\n",
              "C    8.618138\n",
              "D    8.386883\n",
              "E    8.224677\n",
              "F    7.881609\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XoCm4q9wgF09",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 306
        },
        "outputId": "4907b1a2-03f9-45ab-c7c1-cebffc53da10"
      },
      "source": [
        "whisky_df.groupby([\"Cluster\"])[\"Meta Critic\"].mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Cluster\n",
              "A     8.798788\n",
              "B     8.588400\n",
              "C     8.646697\n",
              "E     8.528304\n",
              "F     8.456190\n",
              "G     8.436197\n",
              "H     8.320833\n",
              "I     8.713672\n",
              "J     8.782213\n",
              "R0    8.565357\n",
              "R1    8.607761\n",
              "R2    8.579851\n",
              "R3    8.591522\n",
              "R4    8.612833\n",
              "U     8.313411\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9-zaKxT6gTr8",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 340
        },
        "outputId": "901e2881-092d-4bed-ba01-9ccb3a211b94"
      },
      "source": [
        "whisky_df.groupby([\"Country\"])[\"Meta Critic\"].mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Country\n",
              "Belgium         8.310000\n",
              "Canada          8.411005\n",
              "England         8.210000\n",
              "Finland         8.488571\n",
              "France          7.290000\n",
              "India           8.790526\n",
              "Ireland         8.415556\n",
              "Japan           8.528533\n",
              "Netherlands     8.347500\n",
              "Scotland        8.581940\n",
              "South Africa    8.186667\n",
              "Sweden          8.560847\n",
              "Switzerland     7.748571\n",
              "Taiwan          8.785625\n",
              "Tasmania        8.534000\n",
              "USA             8.595788\n",
              "Wales           8.101250\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GsFdm_Kx1Fnr",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 153
        },
        "outputId": "a390cc7b-9833-48bd-9342-52fa7b1485fe"
      },
      "source": [
        "whisky_df.groupby(['Country Condensed'])[\"Meta Critic\"].mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Country Condensed\n",
              "Canada      8.411005\n",
              "Ireland     8.415556\n",
              "Japan       8.528533\n",
              "Other       8.548411\n",
              "Scotland    8.581940\n",
              "USA         8.595788\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Xphwknz4I443",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "6469922a-4086-4722-a397-8e3d1313c1fc"
      },
      "source": [
        "whisky_df.groupby(['Type'])[\"Meta Critic\"].mean()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Type\n",
              "Barley       8.550000\n",
              "Blend        8.312712\n",
              "Bourbon      8.598039\n",
              "Flavoured    7.360000\n",
              "Grain        8.287143\n",
              "Malt         8.603295\n",
              "Rye          8.645696\n",
              "Wheat        8.500000\n",
              "Whiskey      8.310000\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5ZUsxXcRgZID",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 170
        },
        "outputId": "22e9adfa-fbfc-45c5-dccc-5756068b478d"
      },
      "source": [
        "whisky_df['Meta Critic'].describe()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "count    1716.000000\n",
              "mean        8.552331\n",
              "std         0.396664\n",
              "min         6.470000\n",
              "25%         8.330000\n",
              "50%         8.600000\n",
              "75%         8.830000\n",
              "max         9.540000\n",
              "Name: Meta Critic, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ylSe2EQw2FjI",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "14f9d857-d73d-4b56-a120-7e889280147b"
      },
      "source": [
        "# Let's see what a scatterplot of the two looks like\n",
        "plt.scatter(whisky_df[\"Country Condensed\"], whisky_df['Meta Critic']);"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QAjFXVh-2mli",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "outputId": "c5e3a22a-03c6-46fa-fff7-d57d1267215d"
      },
      "source": [
        "# Scatter is too difficult to determine anything specific. Boxplot might be better\n",
        "whisky_df.boxplot(column='Meta Critic', by='Country Condensed');"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\ngg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PmRqUroJ35nw",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "outputId": "56cb5719-14be-4a74-f322-80092bd220cf"
      },
      "source": [
        "# Again, with different variables\n",
        "whisky_df.boxplot(column=\"Meta Critic\", by='Cost Rank');"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qghKTOSs4I19",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "outputId": "cf375a52-077a-404b-cff3-8465cea4966e"
      },
      "source": [
        "whisky_df.boxplot(column='Meta Critic', by='Cluster');"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\na4O+/+L3qg06YUpGMyuNo2bSSm3+KZN/ab1161aWLl3K0qVLU9dBlgaa/XVmpXHUTFqp5p8yDQ0N\n9PX1jbrk7uvro6GhodxFq0otLS10dnbS1tbGkiVLaGtrS00ySusopbSNmkkr1fxTRm3QMy+NUzBo\nlJJMZtKav5m92szuibyeNrNz87ZZbGZPRbb5YnJFzrak26DVn1AdNEpJJjNpzd/d/wAcB2Bms4CH\ngB8X2PSX7v7ueIsn+ZJsg1Z/QvXQFaJMptQ2/2bgz+5+fxKFkckl2Qat/oTqoVFKMhkrpUZnZlcC\nd7n7mrzli4FrgO3ADuCz7r61wM8vB5YD1NXVNa5bt67ofQ8NDVFbW1v09qVQ7EBzczM333wzs2fP\nHom9d+9elixZQm9vb2z7SdMxSWvs3t5eurq6WLlyJQsWLGBwcJDVq1fT2tpKc3NzbPtJ0zGphtjL\nbhpm7TvmFL19U1PTZnc/vuDK6A0aE72AfYGdQF2BdQcCteH7U4A/TRavsbHRS9Hf31/S9uWO3d3d\n7Q0NDV5TU+MNDQ3e3d0d+z7iLndDQ4P39fWNit3X1+cNDQ2x7idtv8u0xk7jd1CxJ3Z0+/UlbQ9s\n8nFycCnNPu8kqPU/UuAE8rS7D4XvbwT2MbPDS4hdVdL4SDfQDTbj0ZBJqUalDPVsAQp+683sZcAj\n7u5mdgJBX8JjMZQvldL4SDfQDTaFaMikVKuiav5mNgf4O+DayLIVZrYi/HgasMXMfg1cCpwRXnJk\nUppHWqi2OJqGTEq1Kqrm7+7DwGF5yy6PvF8DrMn/uazKjbRoamoaWaaRFumU5hO5yEQ0vUMC1HZe\nPTRkUqqVpndIgNrOq4cmdpNqpeSfkDTOByNj6UQu1UrJX2QSOpFLNVKbv4hIBin5i4hkkJK/SBml\n9e5hST+1+YuUie4elnJSzV+kTHT3sJSTkr9ImejuYSknJX+RMqmvr+eiiy4a1eZ/0UUXZf7uYfWD\nzAy1+YuUSVNTE6tWrWLVqlUce+yx/O53v6O9vZ0VK1ZM/sNVSv0gY+U/XtVWvfB+OvNnquYvUib9\n/f20t7dz5ZVX8q53vYsrr7yS9vZ2+vv7y120slE/yFjRB7DkHq/qLzxIa8pU8xcpk4GBAe6++26+\n8pWvjNw9vGfPHr761a+Wu2hlo36QmaOav0iZaMbQsXRMZo6Sv0iZaOrvsXRMCkuiE1zNPiJlohlD\n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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yqutPcmO4g78",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "outputId": "789be047-ab5d-425c-e453-e9e714800690"
      },
      "source": [
        "whisky_df.boxplot(column='Meta Critic', by='Class');"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\ngd/MLGf+P9T+pv94+BKhAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "69ugM_eU4zcD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "outputId": "cfd2b3f6-1a7e-48b4-9332-b93a1ef74a13"
      },
      "source": [
        "whisky_df.boxplot(column='Meta Critic', by='Type');"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uWxFk9uy5P2e",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "outputId": "b3cca1c7-c2d0-4281-9df6-274ae880a1d4"
      },
      "source": [
        "# Are their any useful comparisons not utilizing 'Meta Critic'?\n",
        "pd.crosstab(whisky_df['Cluster'], whisky_df['Type'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th>Type</th>\n",
              "      <th>Barley</th>\n",
              "      <th>Blend</th>\n",
              "      <th>Bourbon</th>\n",
              "      <th>Flavoured</th>\n",
              "      <th>Grain</th>\n",
              "      <th>Malt</th>\n",
              "      <th>Rye</th>\n",
              "      <th>Wheat</th>\n",
              "      <th>Whiskey</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cluster</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>A</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>99</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>B</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>50</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>C</th>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>215</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>E</th>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>214</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>F</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>41</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>G</th>\n",
              "      <td>0</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>135</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>H</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>72</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>I</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>177</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>J</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>122</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R0</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>23</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R1</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R2</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R3</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>46</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R4</th>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>58</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>U</th>\n",
              "      <td>1</td>\n",
              "      <td>271</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>5</td>\n",
              "      <td>1</td>\n",
              "      <td>21</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Type     Barley  Blend  Bourbon  Flavoured  Grain  Malt  Rye  Wheat  Whiskey\n",
              "Cluster                                                                     \n",
              "A             0      0        0          0      0    99    0      0        0\n",
              "B             0      0        0          0      0    50    0      0        0\n",
              "C             0      3        0          0      0   215    0      0        0\n",
              "E             0     10        0          0      0   214    0      0        0\n",
              "F             0      1        0          0      0    41    0      0        0\n",
              "G             0      7        0          0      0   135    0      0        0\n",
              "H             0      0        0          0      0    72    0      0        0\n",
              "I             0      0        0          0      0   177    0      0        0\n",
              "J             0      0        0          0      0   122    0      0        0\n",
              "R0            0      1       23          0      2     0    0      1        1\n",
              "R1            0      0       67          0      0     0    0      0        0\n",
              "R2            0      0       67          0      0     0    0      0        0\n",
              "R3            0      0       46          0      0     0    0      0        0\n",
              "R4            0      2        0          0      0     0   58      0        0\n",
              "U             1    271        1          1      5     1   21      1        0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s36Ecx1izHCD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "f1738c12-30d2-4579-e8a1-edc128357bd8"
      },
      "source": [
        "# What does this look like as a stacked bar chart?\n",
        "flavor_type = pd.crosstab(whisky_df['Cluster'], whisky_df['Type'])\n",
        "flavor_type.plot(kind='barh', stacked=True);"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nfdxg1iB5hVC",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "outputId": "23f7a92d-6245-4ce6-c49b-9dfa4c450646"
      },
      "source": [
        "# This next one is not really useful information because it is literally \n",
        "# defined by the Cluster code definitions already\n",
        "pd.crosstab(whisky_df['Cluster'], whisky_df['Class'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th>Class</th>\n",
              "      <th>Bourbon-like</th>\n",
              "      <th>Rye-like</th>\n",
              "      <th>Scotch-like</th>\n",
              "      <th>SingleMalt-like</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cluster</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>A</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>99</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>B</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>50</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>C</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>218</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>E</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>224</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>F</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>42</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>G</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>142</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>H</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>72</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>I</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>177</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>J</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>122</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R0</th>\n",
              "      <td>28</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R1</th>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R2</th>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R3</th>\n",
              "      <td>46</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R4</th>\n",
              "      <td>0</td>\n",
              "      <td>60</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>U</th>\n",
              "      <td>12</td>\n",
              "      <td>155</td>\n",
              "      <td>135</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Class    Bourbon-like  Rye-like  Scotch-like  SingleMalt-like\n",
              "Cluster                                                      \n",
              "A                   0         0            0               99\n",
              "B                   0         0            0               50\n",
              "C                   0         0            0              218\n",
              "E                   0         0            0              224\n",
              "F                   0         0            0               42\n",
              "G                   0         0            0              142\n",
              "H                   0         0            0               72\n",
              "I                   0         0            0              177\n",
              "J                   0         0            0              122\n",
              "R0                 28         0            0                0\n",
              "R1                 67         0            0                0\n",
              "R2                 67         0            0                0\n",
              "R3                 46         0            0                0\n",
              "R4                  0        60            0                0\n",
              "U                  12       155          135                0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yMxnQSMl5n_b",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 359
        },
        "outputId": "17d1536c-0359-40dd-e7f6-e0ae8297930b"
      },
      "source": [
        "# Similar to previous, but more useful\n",
        "pd.crosstab(whisky_df['Type'], whisky_df['Class'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th>Class</th>\n",
              "      <th>Bourbon-like</th>\n",
              "      <th>Rye-like</th>\n",
              "      <th>Scotch-like</th>\n",
              "      <th>SingleMalt-like</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Type</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Barley</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Blend</th>\n",
              "      <td>11</td>\n",
              "      <td>134</td>\n",
              "      <td>129</td>\n",
              "      <td>21</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Bourbon</th>\n",
              "      <td>204</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Flavoured</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Grain</th>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Malt</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1125</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Rye</th>\n",
              "      <td>0</td>\n",
              "      <td>79</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Wheat</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Whiskey</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Class      Bourbon-like  Rye-like  Scotch-like  SingleMalt-like\n",
              "Type                                                           \n",
              "Barley                0         1            0                0\n",
              "Blend                11       134          129               21\n",
              "Bourbon             204         0            0                0\n",
              "Flavoured             1         0            0                0\n",
              "Grain                 2         0            5                0\n",
              "Malt                  0         0            1             1125\n",
              "Rye                   0        79            0                0\n",
              "Wheat                 1         1            0                0\n",
              "Whiskey               1         0            0                0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wfdCGqNkUvBj",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "cb81e703-db51-4fa7-e51a-00fda76c64f2"
      },
      "source": [
        "# Lets see this as a stacked bar too\n",
        "# The Malt value skews the graph and makes it hard to read \n",
        "type_class = pd.crosstab(whisky_df['Type'], whisky_df['Class'])\n",
        "type_class.plot(kind='barh', stacked=True);"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nwzn00EOrfZ6SkhI++ugjjj/++C3WK1tWYmvy8vIoLi6mXbt29O/fnylTpvDKK69w++23M2HChGrH\n1RA0qmQUETlVlC8m84QcEfE18J/JJ9sryafsmMuzvv8ZOKhWgzWzJmXq1KlMmDCBGTNm0KJFC1au\nXMn69etr1EZJSQnFxcVbTUbbYsqUKbXeZn1qtNN0Zmb1afny5bRr144WLVoA0K5dOzp16sS0adPo\n378/Bx98MIcccghr1qxh7dq1nHfeefTo0YNevXoxefJk1q9fz69+9SvGjBlDQUEBY8aM4fPPPy+v\n17NnT55++uny811zzTUcfPDB9OvXj48//nir8bVq1WqzsmnTptGrVy8WLVpU4yUt6lujGhk1JD32\nbEPxLd9LttJ9NUlVqjfxUI/OSTsAs203cOBArr/+eg444ACOOeYYBg8eTFFREYMHD2bMmDH07duX\nzz77jJYtW/K73/0OScyZM4f58+czcOBAFixYwPXXX7/JCq6//OUvadOmDXPmZP5r/cc/Mr86+cUX\nX9CvXz9uuukmfvGLX/DAAw9w7bXX1ijeKVOmMHToUJ555hk6duzIWWedxTPPPEP79u0ZM2YM11xz\nDQ8//HDt/pC2g5ORmVk1tGrViunTp/Paa68xefJkBg8ezDXXXEPHjh3p27cvAK1btwbg9ddfZ+jQ\noUDm/XKdO3cuf/t2tpdeeoknn3yyfHv33XcHMu+2O+GEzBvJ+vTpw4svvlijWN99910uuugiJk6c\nSKdOnZg7d+52L2lR15yMzMyqKScnhwEDBjBgwAB69OhR60uPl2nevDllD2bl5OSwceNGSktL6dOn\nDwAnnngi119/fZXHd+zYkbVr1zJz5kw6depUvqTF1KlT6yTe2uB7RmZm1fDee++xcOHC8u2SkhK6\ndevG8uXLmTZtGgBr1qxh48aNHHHEEYwePRqABQsW8H//938ceOCB7LrrrqxZs6a8jWOPPXaThFY2\nTVeZnJyc8rWMtpSIAHbbbTeef/55rr76al555ZVNlrQA2LBhA++8807Nfwh1yCMjM2t0qvsodm36\n/PPPGTp0KJ9++inNmjVjv/32Y8SIEZx33nkMHTqUr776ipYtW/LSSy9x6aWXcskll9CjRw+aNWvG\nyJEjadGiBd/+9re55ZZbKCgo4Oqrr+baa6/lsssuIz8/n5ycHK677jp+8IMf1Eq8HTp0YMKECXz3\n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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RC1P_CpnLpRQ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "outputId": "0efbe74f-6f93-42c7-9cfc-ee3656405c4f"
      },
      "source": [
        "pd.crosstab(whisky_df['Cluster'], whisky_df['Country Condensed'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th>Country Condensed</th>\n",
              "      <th>Canada</th>\n",
              "      <th>Ireland</th>\n",
              "      <th>Japan</th>\n",
              "      <th>Other</th>\n",
              "      <th>Scotland</th>\n",
              "      <th>USA</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cluster</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>A</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>7</td>\n",
              "      <td>89</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>B</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>8</td>\n",
              "      <td>34</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>C</th>\n",
              "      <td>1</td>\n",
              "      <td>12</td>\n",
              "      <td>5</td>\n",
              "      <td>29</td>\n",
              "      <td>170</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>E</th>\n",
              "      <td>9</td>\n",
              "      <td>16</td>\n",
              "      <td>12</td>\n",
              "      <td>39</td>\n",
              "      <td>144</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>F</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>5</td>\n",
              "      <td>4</td>\n",
              "      <td>31</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>G</th>\n",
              "      <td>15</td>\n",
              "      <td>6</td>\n",
              "      <td>18</td>\n",
              "      <td>13</td>\n",
              "      <td>88</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>H</th>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>6</td>\n",
              "      <td>11</td>\n",
              "      <td>50</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>I</th>\n",
              "      <td>5</td>\n",
              "      <td>1</td>\n",
              "      <td>11</td>\n",
              "      <td>16</td>\n",
              "      <td>142</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>J</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>17</td>\n",
              "      <td>103</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>27</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R1</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>67</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R2</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>67</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R3</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>46</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>R4</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>60</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>U</th>\n",
              "      <td>153</td>\n",
              "      <td>36</td>\n",
              "      <td>14</td>\n",
              "      <td>7</td>\n",
              "      <td>77</td>\n",
              "      <td>15</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Country Condensed  Canada  Ireland  Japan  Other  Scotland  USA\n",
              "Cluster                                                        \n",
              "A                       0        1      2      7        89    0\n",
              "B                       3        3      1      8        34    1\n",
              "C                       1       12      5     29       170    1\n",
              "E                       9       16     12     39       144    4\n",
              "F                       1        1      5      4        31    0\n",
              "G                      15        6     18     13        88    2\n",
              "H                       1        4      6     11        50    0\n",
              "I                       5        1     11     16       142    2\n",
              "J                       0        1      1     17       103    0\n",
              "R0                      1        0      0      0         0   27\n",
              "R1                      0        0      0      0         0   67\n",
              "R2                      0        0      0      0         0   67\n",
              "R3                      0        0      0      0         0   46\n",
              "R4                      0        0      0      0         0   60\n",
              "U                     153       36     14      7        77   15"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qSSJuVGALs9l",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "outputId": "504c92f2-d13a-484e-eb16-642185965ad1"
      },
      "source": [
        "type_class = pd.crosstab(whisky_df['Country Condensed'], whisky_df['Cluster'])\n",
        "type_class.plot(kind='barh', stacked=True);"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sSLK8x56MHvS",
        "colab_type": "text"
      },
      "source": [
        "# Final Notes\n",
        "\n",
        "## What I want to do better\n",
        "\n",
        "I wanted to try looking at correlation heat maps utilizing dython, but I ran out of time. \n",
        "\n",
        "I wasn't able to clean up the visualizations as much as I would like. \n",
        "\n",
        "## Takeaways\n",
        "\n",
        "Scotch Single Malts seem to be the preffered type of whisky for the data gatherer. This meant the results were weighted towards these categories.\n",
        "\n",
        "Country of Origin has little effect on the overall score of a whisky.\n",
        "\n",
        "The Cost of a whisky seems to have a proportional correlation to the score of a whisky. Could be that higher costing whiskies are objectively better, but could also partially be the consumer rationalizing the higher expenditure as \"better\". Would be interesting to try this with a blind group who did not know the cost of the whiskies they were trying. \n",
        "\n",
        "Scotland has a more homogenous flavor profile than I initially would have thought. In retrospect, it makes sense when considering the breadth of their sub-regions."
      ]
    }
  ]
}
