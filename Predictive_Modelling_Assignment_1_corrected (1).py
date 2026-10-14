#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Predictive Modelling: Assignment 1

#****************QUESTION 1**************
#import modules
#from google.colab import drive
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import statsmodels.api as sm

#mount drive (for google colab)
#drive.mount('/content/drive')

#file_path = '/content/drive/MyDrive/sales_transactions (2).csv'


# In[10]:


import os

print(os.getcwd())
print(os.listdir())


# In[11]:


#read csv file and see first 5 rows
df = pd.read_csv('sales_transactions (1).csv')
df.head()


# In[12]:


#Make copy of df to keep original df intact
df_copy = df.copy()

#convert to datetime values
df_copy["order_date"] = pd.to_datetime(df_copy["order_date"])
df_copy["delivery_date"] = pd.to_datetime(df_copy["delivery_date"])

#create year-2 week periods within each calendar year.
#We multiply by 100 to create a number like "202001" (this corresponds to a specific 2 week period within a calendar year)
order_period = df_copy["order_date"].dt.year * 100 + (df_copy["order_date"].dt.dayofyear - 1) // 14 + 1

delivery_period = df_copy["delivery_date"].dt.year * 100 + (df_copy["delivery_date"].dt.dayofyear - 1) // 14 + 1

# NEW: continuous two-week period numbers for calculations across years
order_period_num = df_copy["order_date"].dt.year * 27 + (df_copy["order_date"].dt.dayofyear - 1) // 14
delivery_period_num = df_copy["delivery_date"].dt.year * 27 + (df_copy["delivery_date"].dt.dayofyear - 1) // 14

# NEW: include periods with no transactions so lag counts actual periods
periods = np.arange(delivery_period_num.min(), delivery_period_num.max() + 1)

is_advance = order_period_num < delivery_period_num





# In[13]:


# --- Build NEW aggregated DataFrame ---
df_2week_demand = (
pd.DataFrame({
"delivery_period_num": delivery_period_num,
"quantity": df_copy["quantity"],
"advance_quantity": df_copy["quantity"].where(is_advance, 0)
})
.groupby("delivery_period_num")
.sum()
.reindex(periods, fill_value=0)
.rename_axis("delivery_period_num")
.reset_index()
)


# NEW: calendar-year label for display
# The continuous number is used for calculations, not displayed
# Reconstruct year and period within year
df_2week_demand["delivery_period"] = (df_2week_demand["delivery_period_num"] // 27) * 100 + (df_2week_demand["delivery_period_num"] % 27) + 1

# Format 2week-year nicely
df_2week_demand["2 week period"] = df_2week_demand["delivery_period"].astype(str).str[:4] + "-" + df_2week_demand["delivery_period"].astype(str).str[4:]

# Final shape & order
df_2week_demand = df_2week_demand[
["2 week period", "advance_quantity", "quantity"]
].rename(columns={
"advance_quantity": "advance_demand",
"quantity": "total_demand"
})

print(df_2week_demand.head())



# In[14]:


#Develop OLS models for forecast horizons ranging from one to four two-week periods.
h = 0
while h != 4:
  #must modify is_advance depending on value of h
  is_advance = order_period_num < delivery_period_num - h # CHANGED: handles year boundaries

  # --- Build NEW aggregated DataFrame ---
  df_2week_demand = (
  pd.DataFrame({
  "delivery_period_num": delivery_period_num,
  "quantity": df_copy["quantity"],
  "advance_quantity": df_copy["quantity"].where(is_advance, 0)
  })
  .groupby("delivery_period_num")
  .sum()
  .reindex(periods, fill_value=0) # NEW: include empty periods for accurate lag
  .rename_axis("delivery_period_num")
  .reset_index()
  )

  # NEW: convert continuous number back to calendar-year label for display
  df_2week_demand["delivery_period"] = (df_2week_demand["delivery_period_num"] // 27) * 100 + (df_2week_demand["delivery_period_num"] % 27) + 1

  # Format 2week-year nicely
  df_2week_demand["2 week period"] = df_2week_demand["delivery_period"].astype(str).str[:4] + "-" + df_2week_demand["delivery_period"].astype(str).str[4:]

  # Final shape & order
  df_2week_demand = df_2week_demand[
  ["2 week period", "advance_quantity", "quantity"]
  ].rename(columns={
  "advance_quantity": "advance_demand",
  "quantity": "total_demand"
  })


  # Lag total demand:
  df_2week_demand["total_demand_lag1"] = df_2week_demand["total_demand"].shift(1 + h)
  # Drop periods without lag information
  df_2week_demand = df_2week_demand.dropna().reset_index(drop=True)


  # Running the regression
  # Dependent variable
  Y = df_2week_demand["total_demand"]
  # Independent variables
  X = df_2week_demand[["total_demand_lag1", "advance_demand"]]
  # Add intercept
  X = sm.add_constant(X)
  model = sm.OLS(Y, X)
  results = model.fit()
  print(f"Forecast horizon: {1 + h} two-week period(s)")
  print(results.summary())

  h += 1


df_2week_demand_copy = df_2week_demand.copy()


# In[15]:


#QUESTION 2

#Assume that the prices of Comfort Mountain, Tough Track, and Ultimate Mountain are $4,200, $1,950, and
#$500, respectively. Develop a logistic regression model to predict the choice probability for each bike.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


y = np.array(df['Product'])
X = np.array(df[['Price_Comfort_Mountain','Price_Tough_Track','Price_Ultimate_Mountain']])

# We fit the model
model_classification = LogisticRegression(random_state=0,max_iter = 1000).fit(X,y)

# F1 Score for model assessment
predictions = model_classification.predict(X)
print("Weighted F1 Score:", f1_score(y, predictions, average='weighted'))

# There are three products in this category that can be seen by classes_
print("Classes:", model_classification.classes_)

print("Intercept:", model_classification.intercept_)

print("Coef:", model_classification.coef_)


# Let's estimate the choice probabilities for the prices given in the question

prices_given = [4200,1950,500]


base = model_classification.predict_proba(np.array(prices_given).reshape(1,3)).ravel()
print("PRICES GIVEN:",prices_given)

print("Choice probabilities for the prices given", base)





# In[16]:


#QUESTION 3:
'''
The bike store manager enters into a contract with a supplier that requires the manager to reserve total
capacity two months in advance. This capacity is shared across all three bike models. The manager determines
the order quantity and price for each model two weeks before the selling period.
Split the dataset into training and test sets. Use the training data to estimate the parameters of the OLS and
logistic regression models. Use the test data to show the optimal capacity, prices, and quantities over the
entire test period
'''


# In[17]:


# Split data into training and test periods

# Create continuous 2-week period numbers
# This avoids problems when the year changes

df_q3 = df.copy()

df_q3["order_date"] = pd.to_datetime(df_q3["order_date"])
df_q3["delivery_date"] = pd.to_datetime(df_q3["delivery_date"])

# Create calendar-year 2-week periods, as in Q1
df_q3["order_period"] = (
    df_q3["order_date"].dt.year * 100
    + (df_q3["order_date"].dt.dayofyear - 1) // 14 + 1
)

df_q3["delivery_period"] = (
    df_q3["delivery_date"].dt.year * 100
    + (df_q3["delivery_date"].dt.dayofyear - 1) // 14 + 1
)

# NEW: continuous two-week period numbers for calculations across years
df_q3["order_period_num"] = (
    df_q3["order_date"].dt.year * 27
    + (df_q3["order_date"].dt.dayofyear - 1) // 14
)

df_q3["delivery_period_num"] = (
    df_q3["delivery_date"].dt.year * 27
    + (df_q3["delivery_date"].dt.dayofyear - 1) // 14
)

# Split by delivery periods, keeping each period together
periods = np.arange(
    df_q3["delivery_period_num"].min(),
    df_q3["delivery_period_num"].max() + 1
)

split = int(len(periods) * 0.8)
test_start = periods[split]
test_periods = periods[split:]


# Create demand data for the 1-period forecast
is_advance = df_q3["order_period_num"] < df_q3["delivery_period_num"]

demand_1 = (
    pd.DataFrame({
        "delivery_period_num": df_q3["delivery_period_num"],
        "quantity": df_q3["quantity"],
        "advance_demand": df_q3["quantity"].where(is_advance, 0)
    })
    .groupby("delivery_period_num")
    .sum()
    .reindex(periods, fill_value=0)
    .rename_axis("delivery_period_num")
    .reset_index()
)

demand_1 = demand_1.rename(columns={"quantity": "total_demand"})

demand_1["total_demand_lag1"] = (
    demand_1["total_demand"].shift(1)
)


# Create demand data for the 4-period forecast
is_advance = (
    df_q3["order_period_num"] < df_q3["delivery_period_num"] - 3
)

demand_4 = (
    pd.DataFrame({
        "delivery_period_num": df_q3["delivery_period_num"],
        "quantity": df_q3["quantity"],
        "advance_demand": df_q3["quantity"].where(is_advance, 0)
    })
    .groupby("delivery_period_num")
    .sum()
    .reindex(periods, fill_value=0)
    .rename_axis("delivery_period_num")
    .reset_index()
)

demand_4 = demand_4.rename(columns={"quantity": "total_demand"})

demand_4["total_demand_lag4"] = (
    demand_4["total_demand"].shift(4)
)


# Estimate the 1-period OLS model using training data
train_1 = demand_1[
    demand_1["delivery_period_num"] < test_start
].dropna()

Y = train_1["total_demand"]

X = train_1[["total_demand_lag1", "advance_demand"]]
X = sm.add_constant(X)

ols_1 = sm.OLS(Y, X).fit()

print("One-period OLS model")
print(ols_1.summary())


# Estimate the 4-period OLS model using training data
train_4 = demand_4[
    demand_4["delivery_period_num"] < test_start
].dropna()

Y = train_4["total_demand"]

X = train_4[["total_demand_lag4", "advance_demand"]]
X = sm.add_constant(X)

ols_4 = sm.OLS(Y, X).fit()

print("Four-period OLS model")
print(ols_4.summary())


# Estimate logistic regression using training transactions
train_transactions = df_q3[
    df_q3["delivery_period_num"] < test_start
].copy()

price_columns = [
    "Price_Comfort_Mountain",
    "Price_Tough_Track",
    "Price_Ultimate_Mountain"
]

y = train_transactions["Product"]

X = train_transactions[price_columns]

model_classification = LogisticRegression(
    random_state=0, max_iter=1000
).fit(X, y)

print("Classes:", model_classification.classes_)
print("Intercept:", model_classification.intercept_)
print("Coefficients:", model_classification.coef_)


# Try 5 candidate prices for each bike
comfort_prices = np.linspace(
    train_transactions["Price_Comfort_Mountain"].min(),
    train_transactions["Price_Comfort_Mountain"].max(), 5
)

tough_prices = np.linspace(
    train_transactions["Price_Tough_Track"].min(),
    train_transactions["Price_Tough_Track"].max(), 5
)

ultimate_prices = np.linspace(
    train_transactions["Price_Ultimate_Mountain"].min(),
    train_transactions["Price_Ultimate_Mountain"].max(), 5
)


# Find which probability belongs to each bike
classes = list(model_classification.classes_)

comfort_index = next(
    i for i, name in enumerate(classes)
    if "comfort" in str(name).lower()
)

tough_index = next(
    i for i, name in enumerate(classes)
    if "tough" in str(name).lower()
)

ultimate_index = next(
    i for i, name in enumerate(classes)
    if "ultimate" in str(name).lower()
)


# Make decisions for every period in the test set
results = []

for period in test_periods:

    # Get demand information for this period
    row_1 = demand_1[
        demand_1["delivery_period_num"] == period
    ]

    row_4 = demand_4[
        demand_4["delivery_period_num"] == period
    ]

    if row_1.empty or row_4.empty:
        continue

    # Forecast demand one period ahead
    X = row_1[["total_demand_lag1", "advance_demand"]]
    X = sm.add_constant(X, has_constant="add")

    demand_forecast = max(
        0, ols_1.predict(X).iloc[0]
    )

    # Forecast capacity four periods ahead
    X = row_4[["total_demand_lag4", "advance_demand"]]
    X = sm.add_constant(X, has_constant="add")

    capacity_forecast = max(
        0, ols_4.predict(X).iloc[0]
    )

    # Reserve enough capacity for the forecast
    capacity = int(np.ceil(capacity_forecast))

    # Expected sales cannot exceed demand or capacity
    expected_sales = min(demand_forecast, capacity)

    # Find prices that maximize expected revenue
    best_revenue = -1

    for comfort in comfort_prices:
        for tough in tough_prices:
            for ultimate in ultimate_prices:

                prices = [comfort, tough, ultimate]

                # Predict choice probabilities


                probabilities = model_classification.predict_proba(
                    pd.DataFrame([prices], columns=price_columns)
                )[0]

                # Put probabilities in the same order as prices
                probabilities = [
                    probabilities[comfort_index],
                    probabilities[tough_index],
                    probabilities[ultimate_index]
                ]

                # Calculate expected quantities
                quantities = expected_sales * np.array(probabilities)

                # Calculate expected revenue
                revenue = sum(
                    np.array(prices) * quantities
                )

                # Save the best combination
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_prices = prices
                    best_quantities = quantities

    # Save the results for this period
    results.append({
        # CHANGED: show calendar-year label rather than continuous number
        "Period": f"{period // 27}-{period % 27 + 1:02d}",
        "Reserved capacity": capacity,
        "Demand forecast": round(demand_forecast, 2),
        "Comfort price": round(best_prices[0], 2),
        "Tough Track price": round(best_prices[1], 2),
        "Ultimate Mountain price": round(best_prices[2], 2),
        "Comfort quantity": round(best_quantities[0], 2),
        "Tough Track quantity": round(best_quantities[1], 2),
        "Ultimate quantity": round(best_quantities[2], 2),
        "Expected revenue": round(best_revenue, 2),
        "Actual demand": int(row_1["total_demand"].iloc[0]),
    })


# Display the results for the test period
results_df = pd.DataFrame(results)

print("Results for the test period:")
print(results_df.to_string(index=False))

