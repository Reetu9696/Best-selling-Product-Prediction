import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('amazon_sales.csv')
'''print(df.head())

df.info()

#dropping useless columns
df.drop(columns=['Order_ID','Customer_ID','Country','Review_Text'],inplace=True)
print(df.head())

df.shape
print(df.isnull().sum())

#checking duplicates
print(df.duplicated().sum())
#descriptive statistics
print(df.describe())

#data visualization
history=['Unit_Price_INR','Total_Sales_INR']
for col in history:
    plt.subplot(1,2,history.index(col)+1)
    sns.histplot(df[col],kde=True,color="blue")
    # plt.title("Distribution of ",col)'''
    
plt.show()
df['Date']=pd.to_datetime(df['Date'])
df['Month']=df["Date"].dt.month
monthly_data=df.groupby(['Month','Product_Category','Product_Name']).agg(Total_Quantity=('Quantity','sum')).reset_index()
print(monthly_data.head())
X_raw=monthly_data[['Month','Product_Category','Product_Name']]
X=pd.get_dummies(X_raw,columns=['Product_Category','Product_Name'],drop_first=True)
y=monthly_data['Total_Quantity']
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X,y)
#generate 2026 prediction matrix
#forecast sales for all 12 months of 2026 for every single item 
all_categories=monthly_data['Product_Category'].unique()
all_products=monthly_data['Product_Name'].unique()
future_records=[]
for month in range(1,13):
    for c in all_categories:
        valid_products=df[df['Product_Category']==c]['Product_Name'].unique()
        for p in valid_products:
            future_records.append({'Month':month,'Product_Category':c,'Product_Name':p})
df_future_raw=pd.DataFrame(future_records)
df_future=pd.get_dummies(df_future_raw,columns=['Product_Category','Product_Name'],drop_first=True)
df_future=df_future.reindex(columns=X.columns,fill_value=0)

# 7. Generate 2026 Forecast
df_future_raw['Predicted_2026_Quantity'] = model.predict(df_future)

# 8. Extract the Final Answers for your Project Objective
# A. Find the absolute top product of 2026
top_product_row = df_future_raw.loc[df_future_raw['Predicted_2026_Quantity'].idxmax()]

# B. Find the breakdown of maximum selling products per specific months
monthly_max = df_future_raw.loc[df_future_raw.groupby('Month')['Predicted_2026_Quantity'].idxmax()]
print("==========================================================")
print("🎯 RESUME PROJECT PREDICTIVE REPORT (FORECAST YEAR: 2026)")
print("==========================================================\n")
print(f"🔥 FUTURE MAX SELLING PRODUCT FOR 2026 = [ {top_product_row['Product_Name']} ]")
print(f"📦 Category: {top_product_row['Product_Category']}")
print(f"📈 Predicted Monthly Peak Volume: {top_product_row['Predicted_2026_Quantity']:.0f} units")
print(f"📅 Peak Selling Month: Month {top_product_row['Month']}\n")

print("📋 2026 MONTH-BY-MONTH INVENTORY DEMAND TARGETS:")
print("----------------------------------------------------------")
for idx, row in monthly_max.iterrows():
    month_name = pd.to_datetime(f"2026-{row['Month']}-01").strftime('%B')
    print(f"📍 {month_name:9}: High-Stock Alert for [ {row['Product_Name']:15} ] ({row['Predicted_2026_Quantity']:.0f} units predicted)")