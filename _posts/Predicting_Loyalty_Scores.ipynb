{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "749f2928",
   "metadata": {},
   "source": [
    "# Predicting Loyalty Scores Using Machine Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "578155de",
   "metadata": {},
   "source": [
    "## Description \n",
    "\n",
    "ABC Gocery Store recently defined a key performance metric (KPI) called `loyalty score` that ranges from 0-1: this is the percentage of the customer's grocery budget spent at ABC Grocery Store relative to other grocery stores. A consulting company sent out cold emails to all customers to obtain this information: approximately 50% of customers responded, the other 50% did not.   \n",
    "\n",
    "The marketing team has deemed it critical to `obtain all loyalty scores` to better evaluate their customers for future marketing campaigns. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1d948e9",
   "metadata": {},
   "source": [
    "## Problem\n",
    "\n",
    "**classic regression problem**:\n",
    "                               Given varying attributes about a customer, can we predict missing loyalty scores?    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0995c467",
   "metadata": {},
   "source": [
    "## Data\n",
    "\n",
    "The grocery database contains six schemas, and they are described as follows;\n",
    "\n",
    "1. customer_details: Contains information about the customer.  Table is at customer level\n",
    "2. transactions: Contains all transaction information for customers, including date of transaction, a unique transaction id, and the product area id.  Sales and number of items for each product area are aggregated.\n",
    "3. product_areas: A lookup table mapping product_area_id in the transaction table, to the product area name.\n",
    "4. delivery_club_campaign: A table showing which customers received mailers (mailer type and control group) as well as those who signed up for the July 1st 2019 campaign promoting a $100 per year membership which offers free grocery deliveries.\n",
    "5. loyalty_scores: A table containing a loyalty score for 400 customers that a consulting company were able to match to their loyalty database that measures the percentage of grocery spend that a customer allocates to this supermarket.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cae0f900",
   "metadata": {},
   "source": [
    "## Creating our Regression Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1c6aabf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required packages \n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Plots to appear in notebook\n",
    "%matplotlib inline\n",
    "\n",
    "## Models\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor, plot_tree\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "## Model evaluators\n",
    "from sklearn.utils import shuffle\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, KFold\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.feature_selection import RFECV\n",
    "from sklearn.inspection import permutation_importance\n",
    "\n",
    "import warnings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "70ab6112",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the data\n",
    "loyalty_scores = pd.read_excel(\"data/grocery_database.xlsx\", sheet_name = \"loyalty_scores\")\n",
    "customer_details = pd.read_excel(\"data/grocery_database.xlsx\", sheet_name = \"customer_details\")\n",
    "transactions = pd.read_excel(\"data/grocery_database.xlsx\", sheet_name = \"transactions\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61e63b35",
   "metadata": {},
   "source": [
    "## Visualize DataFrames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fad78f2a",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>customer_loyalty_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>104</td>\n",
       "      <td>0.587</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>69</td>\n",
       "      <td>0.156</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>525</td>\n",
       "      <td>0.959</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>181</td>\n",
       "      <td>0.418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>796</td>\n",
       "      <td>0.570</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  customer_loyalty_score\n",
       "0          104                   0.587\n",
       "1           69                   0.156\n",
       "2          525                   0.959\n",
       "3          181                   0.418\n",
       "4          796                   0.570"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Visualize loyalty_scores DataFrame\n",
    "loyalty_scores.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0d61207",
   "metadata": {},
   "source": [
    "Column Description\n",
    "1. customer_id: customer unique identifier.\n",
    "2. customer_loyalty_scores: percentage of grocery spend that a customer allocates to this supermarket."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d9b174ea",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>74</td>\n",
       "      <td>3.38</td>\n",
       "      <td>F</td>\n",
       "      <td>0.59</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>524</td>\n",
       "      <td>4.76</td>\n",
       "      <td>F</td>\n",
       "      <td>0.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>607</td>\n",
       "      <td>4.45</td>\n",
       "      <td>F</td>\n",
       "      <td>0.49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>343</td>\n",
       "      <td>0.91</td>\n",
       "      <td>M</td>\n",
       "      <td>0.54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>322</td>\n",
       "      <td>3.02</td>\n",
       "      <td>F</td>\n",
       "      <td>0.63</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  distance_from_store gender  credit_score\n",
       "0           74                 3.38      F          0.59\n",
       "1          524                 4.76      F          0.52\n",
       "2          607                 4.45      F          0.49\n",
       "3          343                 0.91      M          0.54\n",
       "4          322                 3.02      F          0.63"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Visualize loyalty_scores DataFrame\n",
    "customer_details.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28ff0b1e",
   "metadata": {},
   "source": [
    "Column Description\n",
    "1. customer_id: customer unique identifier.\n",
    "2. distance_from_store: distance a customer covers to store.\n",
    "3. gender: sexual orientation.\n",
    "4. credit_score: financial worthiness."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2e4dd55a",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>transaction_date</th>\n",
       "      <th>transaction_id</th>\n",
       "      <th>product_area_id</th>\n",
       "      <th>num_items</th>\n",
       "      <th>sales_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2020-04-10</td>\n",
       "      <td>435657533999</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>19.16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2020-04-10</td>\n",
       "      <td>435657533999</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>7.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>2020-06-02</td>\n",
       "      <td>436189770685</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>26.97</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>2020-06-02</td>\n",
       "      <td>436189770685</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>38.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>2020-06-10</td>\n",
       "      <td>436265380298</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>22.13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id transaction_date  transaction_id  product_area_id  num_items  \\\n",
       "0            1       2020-04-10    435657533999                3          7   \n",
       "1            1       2020-04-10    435657533999                2          5   \n",
       "2            1       2020-06-02    436189770685                4          4   \n",
       "3            1       2020-06-02    436189770685                1          2   \n",
       "4            1       2020-06-10    436265380298                4          4   \n",
       "\n",
       "   sales_cost  \n",
       "0       19.16  \n",
       "1        7.71  \n",
       "2       26.97  \n",
       "3       38.52  \n",
       "4       22.13  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Visualize transactions DataFrame\n",
    "transactions.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e32ef7f3",
   "metadata": {},
   "source": [
    "Column Description\n",
    "1. customer_id: customer unique identifier.\n",
    "2. transaction_date: date a transaction occurred.\n",
    "3. transaction_id: transaction unique identifier.\n",
    "4. product_area_id: product name unique identifier.\n",
    "5. num_items: number of items bought per transaction.\n",
    "6. sales_cost: amount spent by customer per transaction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "20c47372",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>customer_loyalty_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>74</td>\n",
       "      <td>3.38</td>\n",
       "      <td>F</td>\n",
       "      <td>0.59</td>\n",
       "      <td>0.263</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>524</td>\n",
       "      <td>4.76</td>\n",
       "      <td>F</td>\n",
       "      <td>0.52</td>\n",
       "      <td>0.298</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>607</td>\n",
       "      <td>4.45</td>\n",
       "      <td>F</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.337</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>343</td>\n",
       "      <td>0.91</td>\n",
       "      <td>M</td>\n",
       "      <td>0.54</td>\n",
       "      <td>0.873</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>322</td>\n",
       "      <td>3.02</td>\n",
       "      <td>F</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.350</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  distance_from_store gender  credit_score  \\\n",
       "0           74                 3.38      F          0.59   \n",
       "1          524                 4.76      F          0.52   \n",
       "2          607                 4.45      F          0.49   \n",
       "3          343                 0.91      M          0.54   \n",
       "4          322                 3.02      F          0.63   \n",
       "\n",
       "   customer_loyalty_score  \n",
       "0                   0.263  \n",
       "1                   0.298  \n",
       "2                   0.337  \n",
       "3                   0.873  \n",
       "4                   0.350  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating a customer level dataset\n",
    "data_for_regression = pd.merge(customer_details, loyalty_scores, how = \"left\", on = \"customer_id\")\n",
    "data_for_regression.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "27757ae5",
   "metadata": {},
   "source": [
    "By joining the customer_details table and loyalty_scores table, we effectively introduced our `customer_loyalty_scores` attribute into our new DataFrame."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09ba976d",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8adbbb09",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>sales_cost</th>\n",
       "      <th>num_items</th>\n",
       "      <th>transaction_id</th>\n",
       "      <th>product_area_id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2056.91</td>\n",
       "      <td>213</td>\n",
       "      <td>52</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>324.22</td>\n",
       "      <td>65</td>\n",
       "      <td>12</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>3499.39</td>\n",
       "      <td>278</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>6609.19</td>\n",
       "      <td>987</td>\n",
       "      <td>106</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  sales_cost  num_items  transaction_id  product_area_id\n",
       "0            1     3980.49        424              51                5\n",
       "1            2     2056.91        213              52                5\n",
       "2            3      324.22         65              12                4\n",
       "3            4     3499.39        278              47                5\n",
       "4            5     6609.19        987             106                5"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Aggregate customer sales data\n",
    "\n",
    "sales_summary = transactions.groupby([\"customer_id\"]).agg({\"sales_cost\" : \"sum\",\n",
    "                                                           \"num_items\" : \"sum\",\n",
    "                                                           \"transaction_id\" : \"count\",\n",
    "                                                           \"product_area_id\" : \"nunique\"}).reset_index()\n",
    "\n",
    "sales_summary.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d976b9ef",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2056.91</td>\n",
       "      <td>213</td>\n",
       "      <td>52</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>324.22</td>\n",
       "      <td>65</td>\n",
       "      <td>12</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>3499.39</td>\n",
       "      <td>278</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>6609.19</td>\n",
       "      <td>987</td>\n",
       "      <td>106</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  total_sales  total_items  transaction_count  \\\n",
       "0            1      3980.49          424                 51   \n",
       "1            2      2056.91          213                 52   \n",
       "2            3       324.22           65                 12   \n",
       "3            4      3499.39          278                 47   \n",
       "4            5      6609.19          987                106   \n",
       "\n",
       "   product_area_count  \n",
       "0                   5  \n",
       "1                   5  \n",
       "2                   4  \n",
       "3                   5  \n",
       "4                   5  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Rename columns for ease of reference\n",
    "\n",
    "sales_summary.columns = [\"customer_id\", \"total_sales\", \"total_items\", \"transaction_count\", \"product_area_count\"]\n",
    "sales_summary.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a852703",
   "metadata": {},
   "source": [
    "Column Description\n",
    "1. customer_id: customer unique identifier.\n",
    "2. sales_cost: amount spent by customer per transaction.\n",
    "3. total_items: aggregate number of purchased items.\n",
    "4. transaction_count: aggregate number of transaction per customer.\n",
    "5. product_area_count: aggregate number of unique product areas purchased"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7b1db855",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>78.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2056.91</td>\n",
       "      <td>213</td>\n",
       "      <td>52</td>\n",
       "      <td>5</td>\n",
       "      <td>39.56</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>324.22</td>\n",
       "      <td>65</td>\n",
       "      <td>12</td>\n",
       "      <td>4</td>\n",
       "      <td>27.02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>3499.39</td>\n",
       "      <td>278</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "      <td>74.46</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>6609.19</td>\n",
       "      <td>987</td>\n",
       "      <td>106</td>\n",
       "      <td>5</td>\n",
       "      <td>62.35</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  total_sales  total_items  transaction_count  \\\n",
       "0            1      3980.49          424                 51   \n",
       "1            2      2056.91          213                 52   \n",
       "2            3       324.22           65                 12   \n",
       "3            4      3499.39          278                 47   \n",
       "4            5      6609.19          987                106   \n",
       "\n",
       "   product_area_count  average_basket_value  \n",
       "0                   5                 78.05  \n",
       "1                   5                 39.56  \n",
       "2                   4                 27.02  \n",
       "3                   5                 74.46  \n",
       "4                   5                 62.35  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# More feature engineering. Add another column... average basket value\n",
    "\n",
    "sales_summary[\"average_basket_value\"] = round(sales_summary[\"total_sales\"] / sales_summary[\"transaction_count\"], 2)\n",
    "sales_summary.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83cfa106",
   "metadata": {},
   "source": [
    "Column Description\n",
    "1. average_basket_value: average dollar amount spent per store visit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "633d2e40",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>customer_loyalty_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>74</td>\n",
       "      <td>3.38</td>\n",
       "      <td>F</td>\n",
       "      <td>0.59</td>\n",
       "      <td>0.263</td>\n",
       "      <td>2563.71</td>\n",
       "      <td>297</td>\n",
       "      <td>44</td>\n",
       "      <td>5</td>\n",
       "      <td>58.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>524</td>\n",
       "      <td>4.76</td>\n",
       "      <td>F</td>\n",
       "      <td>0.52</td>\n",
       "      <td>0.298</td>\n",
       "      <td>2996.02</td>\n",
       "      <td>357</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>61.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>607</td>\n",
       "      <td>4.45</td>\n",
       "      <td>F</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.337</td>\n",
       "      <td>2853.82</td>\n",
       "      <td>350</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>58.24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>343</td>\n",
       "      <td>0.91</td>\n",
       "      <td>M</td>\n",
       "      <td>0.54</td>\n",
       "      <td>0.873</td>\n",
       "      <td>2388.31</td>\n",
       "      <td>272</td>\n",
       "      <td>54</td>\n",
       "      <td>5</td>\n",
       "      <td>44.23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>322</td>\n",
       "      <td>3.02</td>\n",
       "      <td>F</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.350</td>\n",
       "      <td>2401.64</td>\n",
       "      <td>278</td>\n",
       "      <td>50</td>\n",
       "      <td>5</td>\n",
       "      <td>48.03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  distance_from_store gender  credit_score  \\\n",
       "0           74                 3.38      F          0.59   \n",
       "1          524                 4.76      F          0.52   \n",
       "2          607                 4.45      F          0.49   \n",
       "3          343                 0.91      M          0.54   \n",
       "4          322                 3.02      F          0.63   \n",
       "\n",
       "   customer_loyalty_score  total_sales  total_items  transaction_count  \\\n",
       "0                   0.263      2563.71          297                 44   \n",
       "1                   0.298      2996.02          357                 49   \n",
       "2                   0.337      2853.82          350                 49   \n",
       "3                   0.873      2388.31          272                 54   \n",
       "4                   0.350      2401.64          278                 50   \n",
       "\n",
       "   product_area_count  average_basket_value  \n",
       "0                   5                 58.27  \n",
       "1                   5                 61.14  \n",
       "2                   5                 58.24  \n",
       "3                   5                 44.23  \n",
       "4                   5                 48.03  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Merge data_for_regression tabe and sales_summary table\n",
    "data_for_regression = pd.merge(data_for_regression, sales_summary, how = \"inner\", on = \"customer_id\")\n",
    "data_for_regression.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "af9ab94e",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>customer_loyalty_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>74</td>\n",
       "      <td>3.38</td>\n",
       "      <td>F</td>\n",
       "      <td>0.59</td>\n",
       "      <td>0.263</td>\n",
       "      <td>2563.71</td>\n",
       "      <td>297</td>\n",
       "      <td>44</td>\n",
       "      <td>5</td>\n",
       "      <td>58.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>524</td>\n",
       "      <td>4.76</td>\n",
       "      <td>F</td>\n",
       "      <td>0.52</td>\n",
       "      <td>0.298</td>\n",
       "      <td>2996.02</td>\n",
       "      <td>357</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>61.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>607</td>\n",
       "      <td>4.45</td>\n",
       "      <td>F</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.337</td>\n",
       "      <td>2853.82</td>\n",
       "      <td>350</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>58.24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>343</td>\n",
       "      <td>0.91</td>\n",
       "      <td>M</td>\n",
       "      <td>0.54</td>\n",
       "      <td>0.873</td>\n",
       "      <td>2388.31</td>\n",
       "      <td>272</td>\n",
       "      <td>54</td>\n",
       "      <td>5</td>\n",
       "      <td>44.23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>322</td>\n",
       "      <td>3.02</td>\n",
       "      <td>F</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.350</td>\n",
       "      <td>2401.64</td>\n",
       "      <td>278</td>\n",
       "      <td>50</td>\n",
       "      <td>5</td>\n",
       "      <td>48.03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_id  distance_from_store gender  credit_score  \\\n",
       "0           74                 3.38      F          0.59   \n",
       "1          524                 4.76      F          0.52   \n",
       "2          607                 4.45      F          0.49   \n",
       "3          343                 0.91      M          0.54   \n",
       "4          322                 3.02      F          0.63   \n",
       "\n",
       "   customer_loyalty_score  total_sales  total_items  transaction_count  \\\n",
       "0                   0.263      2563.71          297                 44   \n",
       "1                   0.298      2996.02          357                 49   \n",
       "2                   0.337      2853.82          350                 49   \n",
       "3                   0.873      2388.31          272                 54   \n",
       "4                   0.350      2401.64          278                 50   \n",
       "\n",
       "   product_area_count  average_basket_value  \n",
       "0                   5                 58.27  \n",
       "1                   5                 61.14  \n",
       "2                   5                 58.24  \n",
       "3                   5                 44.23  \n",
       "4                   5                 48.03  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create DataFrame for modelling: contains customer loyalty scores.\n",
    "data_for_modelling = data_for_regression.loc[data_for_regression[\"customer_loyalty_score\"].notna()]\n",
    "data_for_modelling.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "53f4e2f2",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>4.78</td>\n",
       "      <td>F</td>\n",
       "      <td>0.66</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>78.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>120</td>\n",
       "      <td>3.49</td>\n",
       "      <td>F</td>\n",
       "      <td>0.38</td>\n",
       "      <td>2887.20</td>\n",
       "      <td>253</td>\n",
       "      <td>45</td>\n",
       "      <td>5</td>\n",
       "      <td>64.16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>52</td>\n",
       "      <td>14.91</td>\n",
       "      <td>F</td>\n",
       "      <td>0.68</td>\n",
       "      <td>3342.75</td>\n",
       "      <td>335</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "      <td>71.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>435</td>\n",
       "      <td>0.25</td>\n",
       "      <td>M</td>\n",
       "      <td>0.62</td>\n",
       "      <td>2326.71</td>\n",
       "      <td>267</td>\n",
       "      <td>48</td>\n",
       "      <td>5</td>\n",
       "      <td>48.47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>679</td>\n",
       "      <td>4.74</td>\n",
       "      <td>F</td>\n",
       "      <td>0.58</td>\n",
       "      <td>3448.59</td>\n",
       "      <td>370</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>70.38</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    customer_id  distance_from_store gender  credit_score  total_sales  \\\n",
       "6             1                 4.78      F          0.66      3980.49   \n",
       "7           120                 3.49      F          0.38      2887.20   \n",
       "8            52                14.91      F          0.68      3342.75   \n",
       "10          435                 0.25      M          0.62      2326.71   \n",
       "12          679                 4.74      F          0.58      3448.59   \n",
       "\n",
       "    total_items  transaction_count  product_area_count  average_basket_value  \n",
       "6           424                 51                   5                 78.05  \n",
       "7           253                 45                   5                 64.16  \n",
       "8           335                 47                   5                 71.12  \n",
       "10          267                 48                   5                 48.47  \n",
       "12          370                 49                   5                 70.38  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "warnings.filterwarnings(\"ignore\")\n",
    "# Create DataFrame for scoring: missing customer loyalty scores.\n",
    "data_for_scoring = data_for_regression.loc[data_for_regression[\"customer_loyalty_score\"].isna()]\n",
    "\n",
    "# Drop empty customer_loyalty_score column\n",
    "data_for_scoring.drop(\"customer_loyalty_score\", axis = 1, inplace = True)\n",
    "data_for_scoring.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7cf0027d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save files\n",
    "pickle.dump(data_for_modelling, open(\"data/abc_regression_modelling.p\", \"wb\"))\n",
    "pickle.dump(data_for_scoring, open(\"data/abc_regression_scoring.p\", \"wb\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1124c7ba",
   "metadata": {},
   "source": [
    "## Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7c57afba",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>customer_loyalty_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.38</td>\n",
       "      <td>F</td>\n",
       "      <td>0.59</td>\n",
       "      <td>0.263</td>\n",
       "      <td>2563.71</td>\n",
       "      <td>297</td>\n",
       "      <td>44</td>\n",
       "      <td>5</td>\n",
       "      <td>58.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.76</td>\n",
       "      <td>F</td>\n",
       "      <td>0.52</td>\n",
       "      <td>0.298</td>\n",
       "      <td>2996.02</td>\n",
       "      <td>357</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>61.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.45</td>\n",
       "      <td>F</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.337</td>\n",
       "      <td>2853.82</td>\n",
       "      <td>350</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>58.24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.91</td>\n",
       "      <td>M</td>\n",
       "      <td>0.54</td>\n",
       "      <td>0.873</td>\n",
       "      <td>2388.31</td>\n",
       "      <td>272</td>\n",
       "      <td>54</td>\n",
       "      <td>5</td>\n",
       "      <td>44.23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.02</td>\n",
       "      <td>F</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.350</td>\n",
       "      <td>2401.64</td>\n",
       "      <td>278</td>\n",
       "      <td>50</td>\n",
       "      <td>5</td>\n",
       "      <td>48.03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   distance_from_store gender  credit_score  customer_loyalty_score  \\\n",
       "0                 3.38      F          0.59                   0.263   \n",
       "1                 4.76      F          0.52                   0.298   \n",
       "2                 4.45      F          0.49                   0.337   \n",
       "3                 0.91      M          0.54                   0.873   \n",
       "4                 3.02      F          0.63                   0.350   \n",
       "\n",
       "   total_sales  total_items  transaction_count  product_area_count  \\\n",
       "0      2563.71          297                 44                   5   \n",
       "1      2996.02          357                 49                   5   \n",
       "2      2853.82          350                 49                   5   \n",
       "3      2388.31          272                 54                   5   \n",
       "4      2401.64          278                 50                   5   \n",
       "\n",
       "   average_basket_value  \n",
       "0                 58.27  \n",
       "1                 61.14  \n",
       "2                 58.24  \n",
       "3                 44.23  \n",
       "4                 48.03  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Import Sample Data\n",
    "###############################################################################\n",
    "\n",
    "# Import\n",
    "data_for_model = pickle.load(open(\"data/abc_regression_modelling.p\", \"rb\"))\n",
    "\n",
    "# Drop unnecessary columns\n",
    "data_for_model.drop([\"customer_id\"], axis = 1, inplace = True)\n",
    "data_for_model.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7b2da92e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Shuffle data\n",
    "data_for_model = shuffle(data_for_model, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c1e20067",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_from_store       2\n",
       "gender                    3\n",
       "credit_score              2\n",
       "customer_loyalty_score    0\n",
       "total_sales               0\n",
       "total_items               0\n",
       "transaction_count         0\n",
       "product_area_count        0\n",
       "average_basket_value      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Deal with Missing Values\n",
    "###############################################################################\n",
    "\n",
    "data_for_model.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2f102afb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_from_store       0\n",
       "gender                    0\n",
       "credit_score              0\n",
       "customer_loyalty_score    0\n",
       "total_sales               0\n",
       "total_items               0\n",
       "transaction_count         0\n",
       "product_area_count        0\n",
       "average_basket_value      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_for_model.dropna(how = \"any\", inplace = True)\n",
    "data_for_model.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f82bc3ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2 outliers detected in column distance_from_store\n",
      "23 outliers detected in column total_sales\n",
      "20 outliers detected in column total_items\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Deal with Outliers\n",
    "###############################################################################\n",
    "\n",
    "outlier_investigation = data_for_model.describe()\n",
    "\n",
    "outlier_columns = [\"distance_from_store\", \"total_sales\", \"total_items\"]\n",
    "\n",
    "# Boxplot approach\n",
    "\n",
    "for column in outlier_columns:\n",
    "    \n",
    "    lower_quartile = data_for_model[column].quantile(0.25)\n",
    "    upper_quartile = data_for_model[column].quantile(0.75)\n",
    "    iqr = upper_quartile - lower_quartile\n",
    "    iqr_extended = iqr * 2\n",
    "    min_border = lower_quartile - iqr_extended\n",
    "    max_border = upper_quartile + iqr_extended\n",
    "    \n",
    "    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index\n",
    "    print(f\"{len(outliers)} outliers detected in column {column}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "567e816c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove outliers\n",
    "data_for_model.drop(outliers, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f1848722",
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################################################################\n",
    "# Split Input Varibles and Output Variables\n",
    "###############################################################################\n",
    "\n",
    "X = data_for_model.drop([\"customer_loyalty_score\"], axis = 1)\n",
    "y = data_for_model[\"customer_loyalty_score\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "32e6dd4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((299, 8), (75, 8), (299,), (75,))"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Split Train and Test Datesets\n",
    "###############################################################################\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)\n",
    "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "95988b75",
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################################################################\n",
    "# Deal with Categorical Variables\n",
    "###############################################################################\n",
    "\n",
    "# Create an object to store categorical variables\n",
    "categorical_vars = [\"gender\"]\n",
    "\n",
    "# Instantiate OneHotEncoder\n",
    "\"\"\"sparse ensure that output is an array making it easy to visualize. Drop parameter removes one dummy variable to mitigate \n",
    "dummy variable trap - input variables perfectly predicting each other\"\"\"\n",
    "one_hot_encoder = OneHotEncoder(sparse = False, drop = \"first\")\n",
    "\n",
    "X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])\n",
    "X_test_encoded = one_hot_encoder.fit_transform(X_test[categorical_vars])\n",
    "\n",
    "\n",
    "encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)\n",
    "\n",
    "# Create dataframe to hold categorical variables\n",
    "X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)\n",
    "\n",
    "# Concatenate dummy variables back to our original dataframe\n",
    "X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1 )\n",
    "\n",
    "X_train.drop(categorical_vars, axis = 1, inplace= True)\n",
    "\n",
    "# Create dataframe to hold categorical variables\n",
    "X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)\n",
    "\n",
    "# Concatenate dummy variables back to our original dataframe\n",
    "X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1 )\n",
    "\n",
    "X_test.drop(categorical_vars, axis = 1, inplace= True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a9604b1",
   "metadata": {},
   "source": [
    "## Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "bee55b76",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Put models in a dictionary\n",
    "models = {\"Linear Regression\": LinearRegression(),\n",
    "          \"Decision Tree\": DecisionTreeRegressor(random_state = 42, max_depth = 4), \n",
    "          \"Random Forest\": RandomForestRegressor(random_state = 42)}\n",
    "\n",
    "# Create function to fit and score models\n",
    "def fit_and_score(models, X_train, X_test, y_train, y_test):\n",
    "    \"\"\"\n",
    "    Fits and evaluates given machine learning models.\n",
    "    models : a dict of different Scikit-Learn machine learning models\n",
    "    X_train : training data\n",
    "    X_test : testing data\n",
    "    y_train : labels assosciated with training data\n",
    "    y_test : labels assosciated with test data\n",
    "    \"\"\"\n",
    "    # Random seed for reproducible results\n",
    "    np.random.seed(42)\n",
    "    # Make a list to keep model scores\n",
    "    model_scores = {}\n",
    "    # Loop through models\n",
    "    for name, model in models.items():\n",
    "        # Fit the model to the data\n",
    "        model.fit(X_train, y_train)\n",
    "        # Evaluate the model and append its score to model_scores\n",
    "        model_scores[name] = model.score(X_test, y_test)\n",
    "    return model_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "df907dbc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Linear Regression': 0.7106110367882751,\n",
       " 'Decision Tree': 0.8970395837191489,\n",
       " 'Random Forest': 0.9325146360291109}"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_scores = fit_and_score(models = models,\n",
    "                             X_train = X_train,\n",
    "                             X_test = X_test,\n",
    "                             y_train = y_train,\n",
    "                             y_test = y_test)\n",
    "model_scores"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37f358a5",
   "metadata": {},
   "source": [
    "## Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "12bf6361",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nz7Evs+lt9T54y6QHqJ3b+srwGVJZtO7z3bPUItdoqqeSPLfgBOAGcAZwO3ddeYDfw6cBVzQLWvOobcsuWAZfd2V5GTgyi7UnqI3s3sCOK/bB/C39JZV/3eSCfRmap+rqseWmlQeB3w5yce713zEqr4+SdKakaq1cwUtyRhgXFUtTPIaejO67arqD30ubcjW3Xzb2vywM/pdhqTVNH/mAf0uoSlJbq2qXZd1bLju6Y1G6wE/6JZXA3zwxRR4kqTht9aGXlX9Flhm0kuS2uRnb0qSmmHoSZKaYehJkpph6EmSmmHoSZKasda+e1OSRqOnnnqKgYEBFi5c2O9SXvTGjx/Plltuybhx44Z8jqEnSSNoYGCADTfckEmTJrHijwjWilQVjzzyCAMDA0yePHnI57m8KUkjaOHChWy66aYG3mpKwqabbrrKM2ZDT5JGmIE3PF7IOBp6kqRmeE9Pkvpo0omXD2t/frj1ihl6o9hOW0xgtn+AJa1Bjz/+ONOnT+f+++9nzJgxvOMd72DmzJn9LmuNcXlTkhpWVXzsYx/jnnvuYc6cOVx//fV897vfHbHrL168eMSuBYaeJDVn/vz5bL/99nzoQx9i7733ZptttgFgnXXWYZdddmFgYGC5515yySXsuOOO7Lzzzuyzzz5AL7hOOOEEdtppJ6ZOncoXvvAFAGbNmsXrXvc6dtppJ4488kiefPJJACZNmsQpp5zC3nvvzSWXXMKVV17JnnvuyS677ML06dP53e9+B8CJJ57IDjvswNSpUznhhBOG5bW7vClJDZo3bx7nnXceZ5111jP7HnvsMS677DKOP/745Z53yimncMUVV7DFFlvw2GOPAXDOOefw4IMPMmfOHMaOHcujjz7KwoULOfzww5k1axbbbbcdhx56KF/84hf5yEc+AvR+sfy6667j4Ycf5j3veQ9XXXUV66+/Pqeddhqnn346M2bM4Jvf/Cb33HMPSZ651upypidJDdpqq63YY489nnm+aNEiDj74YI477ji23nrr5Z631157cfjhh/OlL33pmaXJq666imOOOYaxY3vzqE022YR58+YxefJktttuOwAOO+wwfvjDHz7Tz3vf+14AbrzxRu666y722msvpk2bxgUXXMDPfvYzNtpoI8aPH89RRx3FpZdeynrrrTcsr9uZniQ1aP3113/O86OPPpptt932mZnY8px99tncdNNNXH755UybNo25c+dSVc/7nbmqGtL1q4q3vvWtfPWrX31em5tvvplZs2Zx8cUXc+aZZ3L11VcP4ZWtmKEnSX00Gn7F4OSTT2bBggWce+65K217//33s/vuu7P77rtz2WWX8fOf/5y3ve1tnH322ey7777PLG9OmTKF+fPnc99997HNNttw0UUX8aY3vel5/e2xxx4ce+yxz7R7/PHHGRgY4JWvfCWPP/44+++/P3vssccz9x1Xl6EnSQ0bGBjg1FNPZcqUKeyyyy4AzJgxg6OOOmqZ7T/+8Y/z05/+lKpiv/32Y+edd2bHHXfk3nvvZerUqYwbN44PfOADzJgxg/POO4/p06ezaNEidtttN4455pjn9Tdx4kTOP/98Dj744Gfe6PKpT32KDTfckAMPPJCFCxdSVXzuc58blteblU1B1T+77rprzZ49u99lSBpGd999N9tvv32/y1hrLGs8k9xaVbsuq71vZJEkNcPlTUnS85x66qlccsklz9k3ffp0TjrppD5VNDwMPUkaYct6t+Noc9JJJ436gHsht+dc3pSkETR+/HgeeeSRF/QPtp615Etkx48fv0rnOdOTpBG05ZZbMjAwwEMPPdTvUl70xo8fz5ZbbrlK5xh6kjSCxo0bx+TJk/tdRrNc3pQkNcPQkyQ1w9CTJDXDT2QZxZL8FpjX7zpGkc2Ah/tdxCjjmDyX4/F8LY7JVlU1cVkHfCPL6DZveR+l06Iksx2P53JMnsvxeD7H5Llc3pQkNcPQkyQ1w9Ab3c7pdwGjjOPxfI7Jczkez+eYDOIbWSRJzXCmJ0lqhqEnSWqGoTcKJHl7knlJ7kty4jKOJ8nnu+O3J9mlH3WOlCGMxyHdONye5IYkO/ejzpGysvEY1G63JIuTHDSS9fXDUMYkyb5J5ia5M8m1I13jSBrC35kJSS5Lcls3Hkf0o85Roar86eMPMAa4H9gaWAe4DdhhqTb7A98FAuwB3NTvuvs8Hn8MvKzb/rPWx2NQu6uB7wAH9bvufo8JsDFwF/Dq7vnL+113n8fj74DTuu2JwKPAOv2uvR8/zvT67w3AfVX1QFX9AbgYOHCpNgcCF1bPjcDGSTYf6UJHyErHo6puqKpfd09vBFbtu0VeXIby5wPgw8A3gF+NZHF9MpQx+Uvg0qr6D4CqWpvHZSjjUcCG6X1z7Qb0Qm/RyJY5Ohh6/bcF8PNBzwe6favaZm2xqq/1r+jNgtdWKx2PJFsA7wbOHsG6+mkof0a2A16W5JoktyY5dMSqG3lDGY8zge2B/wvcARxfVU+PTHmjix9D1n9Zxr6lf49kKG3WFkN+rUn+hF7o7b1GK+qvoYzHGcAnqmpx7z/ya72hjMlY4PXAfsBLgR8lubGq7l3TxfXBUMbjT4G5wJuB1wDfT/LvVfWbNVzbqGPo9d8A8KpBz7ek97+xVW2zthjSa00yFTgX+LOqemSEauuHoYzHrsDFXeBtBuyfZFFVfWtEKhx5Q/0783BV/R74fZIfAjsDa2PoDWU8jgBmVu+m3n1JHgSmADePTImjh8ub/XcLsG2SyUnWAd4HfHupNt8GDu3exbkHsKCqfjnShY6QlY5HklcDlwLvX0v/5z7YSsejqiZX1aSqmgR8HfjQWhx4MLS/M/8KvDHJ2CTrAbsDd49wnSNlKOPxH/RmvSR5BfBa4IERrXKUcKbXZ1W1KMkM4Ap678L6clXdmeSY7vjZ9N6Rtz9wH/A4vf+1rZWGOB7/AGwKnNXNbhbVWvop8kMcj6YMZUyq6u4k3wNuB54Gzq2qn/Sv6jVniH9G/hk4P8kd9JZDP1FVrX3dEODHkEmSGuLypiSpGYaeJKkZhp4kqRmGniSpGYaeJKkZhp4kqRmGniSpGf8fds7Tytj42JQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_compare = pd.DataFrame(model_scores, index=['r2_scores'])\n",
    "model_compare.T.plot.barh();"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c26bedb",
   "metadata": {},
   "source": [
    "From our preliminary results, `RandomForestRegressor` scored highest."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f2f9454",
   "metadata": {},
   "source": [
    "## Random Forest Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3482ec21",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(random_state=42)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Model Training\n",
    "###############################################################################\n",
    "\n",
    "regressor = RandomForestRegressor(random_state = 42)\n",
    "regressor.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "36e2141d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9325146360291109"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###############################################################################\n",
    "# Model Assessment\n",
    "###############################################################################\n",
    "\n",
    "# Predict on the test set\n",
    "y_pred = regressor.predict(X_test)\n",
    "\n",
    "# Calculate R-Squared\n",
    "r_squared = r2_score(y_test, y_pred)\n",
    "r_squared"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "b5690038",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9192168158295811"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cross Validation\n",
    "cv = KFold(n_splits = 4, shuffle = True, random_state = 42)\n",
    "cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = \"r2\")\n",
    "cv_scores.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6f1b170",
   "metadata": {},
   "source": [
    "`cross_val_score()` works by taking an estimator (machine learning model) along with data and labels. It then evaluates the machine learning model on the data and labels using cross-validation and a defined scoring parameter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "cea48850",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9243345919114274"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculate Adjusted R-Squared\n",
    "num_data_points, num_input_vars = X_test.shape\n",
    "adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)\n",
    "adjusted_r_squared"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e78dfc1a",
   "metadata": {},
   "source": [
    "## Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b25a44f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Permutation Importance\n",
    "result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)\n",
    "\n",
    "# To DataFrame\n",
    "permutation_importance = pd.DataFrame(result[\"importances_mean\"])\n",
    "feature_names = pd.DataFrame(X.columns)\n",
    "permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)\n",
    "permutation_importance_summary.columns = [\"input_variable\", \"permutation_importance\"]\n",
    "\n",
    "permutation_importance_summary.sort_values(by = \"permutation_importance\", inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3f5daf2",
   "metadata": {},
   "source": [
    "Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular. In other words, it measures which features are contributing most to the outcome of the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "d8d81e4f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot \n",
    "plt.barh(permutation_importance_summary[\"input_variable\"], permutation_importance_summary[\"permutation_importance\"])\n",
    "plt.title(\"Permutation Importance of Random Forest\")\n",
    "plt.xlabel(\"Permutation Importance\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94dc38b5",
   "metadata": {},
   "source": [
    "From the plot above, `distance_from_store` contributes the most to the model's decision."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b8069c5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save Objects to Pickle File\n",
    "pickle.dump(regressor, open(\"data/random_forest_regressor_model.p\", \"wb\"))\n",
    "pickle.dump(one_hot_encoder, open(\"data/random_forest_regressor_ohe.p\", \"wb\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bcae7752",
   "metadata": {},
   "source": [
    "## Predicting Missing Loyalty Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "0aaefe65",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_id</th>\n",
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>4.78</td>\n",
       "      <td>F</td>\n",
       "      <td>0.66</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>78.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>120</td>\n",
       "      <td>3.49</td>\n",
       "      <td>F</td>\n",
       "      <td>0.38</td>\n",
       "      <td>2887.20</td>\n",
       "      <td>253</td>\n",
       "      <td>45</td>\n",
       "      <td>5</td>\n",
       "      <td>64.16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>52</td>\n",
       "      <td>14.91</td>\n",
       "      <td>F</td>\n",
       "      <td>0.68</td>\n",
       "      <td>3342.75</td>\n",
       "      <td>335</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "      <td>71.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>435</td>\n",
       "      <td>0.25</td>\n",
       "      <td>M</td>\n",
       "      <td>0.62</td>\n",
       "      <td>2326.71</td>\n",
       "      <td>267</td>\n",
       "      <td>48</td>\n",
       "      <td>5</td>\n",
       "      <td>48.47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>679</td>\n",
       "      <td>4.74</td>\n",
       "      <td>F</td>\n",
       "      <td>0.58</td>\n",
       "      <td>3448.59</td>\n",
       "      <td>370</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>70.38</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    customer_id  distance_from_store gender  credit_score  total_sales  \\\n",
       "6             1                 4.78      F          0.66      3980.49   \n",
       "7           120                 3.49      F          0.38      2887.20   \n",
       "8            52                14.91      F          0.68      3342.75   \n",
       "10          435                 0.25      M          0.62      2326.71   \n",
       "12          679                 4.74      F          0.58      3448.59   \n",
       "\n",
       "    total_items  transaction_count  product_area_count  average_basket_value  \n",
       "6           424                 51                   5                 78.05  \n",
       "7           253                 45                   5                 64.16  \n",
       "8           335                 47                   5                 71.12  \n",
       "10          267                 48                   5                 48.47  \n",
       "12          370                 49                   5                 70.38  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import customers for scoring\n",
    "to_be_scored = pickle.load(open(\"data/abc_regression_scoring.p\", \"rb\"))\n",
    "to_be_scored.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b2e76f3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import model and model objects\n",
    "regressor = pickle.load(open(\"data/random_forest_regression_model.p\", \"rb\"))\n",
    "one_hot_encoder = pickle.load(open(\"data/random_forest_regression_ohe.p\", \"rb\"))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "ff05d9fc",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>distance_from_store</th>\n",
       "      <th>gender</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>4.78</td>\n",
       "      <td>F</td>\n",
       "      <td>0.66</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>78.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.49</td>\n",
       "      <td>F</td>\n",
       "      <td>0.38</td>\n",
       "      <td>2887.20</td>\n",
       "      <td>253</td>\n",
       "      <td>45</td>\n",
       "      <td>5</td>\n",
       "      <td>64.16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>14.91</td>\n",
       "      <td>F</td>\n",
       "      <td>0.68</td>\n",
       "      <td>3342.75</td>\n",
       "      <td>335</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "      <td>71.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>0.25</td>\n",
       "      <td>M</td>\n",
       "      <td>0.62</td>\n",
       "      <td>2326.71</td>\n",
       "      <td>267</td>\n",
       "      <td>48</td>\n",
       "      <td>5</td>\n",
       "      <td>48.47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.74</td>\n",
       "      <td>F</td>\n",
       "      <td>0.58</td>\n",
       "      <td>3448.59</td>\n",
       "      <td>370</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>70.38</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    distance_from_store gender  credit_score  total_sales  total_items  \\\n",
       "6                  4.78      F          0.66      3980.49          424   \n",
       "7                  3.49      F          0.38      2887.20          253   \n",
       "8                 14.91      F          0.68      3342.75          335   \n",
       "10                 0.25      M          0.62      2326.71          267   \n",
       "12                 4.74      F          0.58      3448.59          370   \n",
       "\n",
       "    transaction_count  product_area_count  average_basket_value  \n",
       "6                  51                   5                 78.05  \n",
       "7                  45                   5                 64.16  \n",
       "8                  47                   5                 71.12  \n",
       "10                 48                   5                 48.47  \n",
       "12                 49                   5                 70.38  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Drop unused columns (customer_id)\n",
    "to_be_scored.drop([\"customer_id\"], axis = 1, inplace = True)\n",
    "to_be_scored.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "f6dc7dc9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_from_store     3\n",
       "gender                  2\n",
       "credit_score            6\n",
       "total_sales             0\n",
       "total_items             0\n",
       "transaction_count       0\n",
       "product_area_count      0\n",
       "average_basket_value    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Find missing values\n",
    "to_be_scored.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0c94741c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "distance_from_store     0\n",
       "gender                  0\n",
       "credit_score            0\n",
       "total_sales             0\n",
       "total_items             0\n",
       "transaction_count       0\n",
       "product_area_count      0\n",
       "average_basket_value    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Drop missing values\n",
    "to_be_scored.dropna(how = \"any\", inplace = True)\n",
    "to_be_scored.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "48a3f2fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply One Hot Encoding\n",
    "\n",
    "# Create an object to store categorical variables\n",
    "categorical_vars = [\"gender\"]\n",
    "\n",
    "# Instantiate OneHotEncoder\n",
    "encoder_var_array = one_hot_encoder.transform(to_be_scored[categorical_vars])\n",
    "encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "107350bd",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>gender_M</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   gender_M\n",
       "0       0.0\n",
       "1       0.0\n",
       "2       0.0\n",
       "3       1.0\n",
       "4       0.0"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create dataframe to hold categorical variables\n",
    "encoder_vars_df = pd.DataFrame(encoder_var_array, columns = encoder_feature_names)\n",
    "encoder_vars_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "c81e151d",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>distance_from_store</th>\n",
       "      <th>credit_score</th>\n",
       "      <th>total_sales</th>\n",
       "      <th>total_items</th>\n",
       "      <th>transaction_count</th>\n",
       "      <th>product_area_count</th>\n",
       "      <th>average_basket_value</th>\n",
       "      <th>gender_M</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.78</td>\n",
       "      <td>0.66</td>\n",
       "      <td>3980.49</td>\n",
       "      <td>424</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>78.05</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.49</td>\n",
       "      <td>0.38</td>\n",
       "      <td>2887.20</td>\n",
       "      <td>253</td>\n",
       "      <td>45</td>\n",
       "      <td>5</td>\n",
       "      <td>64.16</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14.91</td>\n",
       "      <td>0.68</td>\n",
       "      <td>3342.75</td>\n",
       "      <td>335</td>\n",
       "      <td>47</td>\n",
       "      <td>5</td>\n",
       "      <td>71.12</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.25</td>\n",
       "      <td>0.62</td>\n",
       "      <td>2326.71</td>\n",
       "      <td>267</td>\n",
       "      <td>48</td>\n",
       "      <td>5</td>\n",
       "      <td>48.47</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.74</td>\n",
       "      <td>0.58</td>\n",
       "      <td>3448.59</td>\n",
       "      <td>370</td>\n",
       "      <td>49</td>\n",
       "      <td>5</td>\n",
       "      <td>70.38</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   distance_from_store  credit_score  total_sales  total_items  \\\n",
       "0                 4.78          0.66      3980.49          424   \n",
       "1                 3.49          0.38      2887.20          253   \n",
       "2                14.91          0.68      3342.75          335   \n",
       "3                 0.25          0.62      2326.71          267   \n",
       "4                 4.74          0.58      3448.59          370   \n",
       "\n",
       "   transaction_count  product_area_count  average_basket_value  gender_M  \n",
       "0                 51                   5                 78.05       0.0  \n",
       "1                 45                   5                 64.16       0.0  \n",
       "2                 47                   5                 71.12       0.0  \n",
       "3                 48                   5                 48.47       1.0  \n",
       "4                 49                   5                 70.38       0.0  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Concatenate dummy variables back to our original dataframe\n",
    "to_be_scored = pd.concat([to_be_scored.reset_index(drop = True), encoder_vars_df.reset_index(drop = True)], axis = 1 )\n",
    "to_be_scored.drop(categorical_vars, axis = 1, inplace= True)\n",
    "to_be_scored.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "c98b306e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make our predictions\n",
    "loyalty_predictions = regressor.predict(to_be_scored)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "6259f517",
   "metadata": {},
   "outputs": [
    {
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
       "      <th>customer_loyalty_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.42641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.32992</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.34719</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.93166</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.38490</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   customer_loyalty_score\n",
       "0                 0.42641\n",
       "1                 0.32992\n",
       "2                 0.34719\n",
       "3                 0.93166\n",
       "4                 0.38490"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Loyalty predictions to a DataFrame\n",
    "scored = pd.DataFrame({\"customer_loyalty_score\" : loyalty_predictions})\n",
    "scored.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ea14965",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
