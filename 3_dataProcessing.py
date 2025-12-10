# 3)Demonstrate data Preprocessing (Data Cleaning, Integration and Transformation)operation on a suitable data.
import pandas as pd
import numpy as np
customers_df = pd.read_csv("customers.csv")
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
print("Coustomer details:\n")
print(customers_df)
print("Order details: \n")
print(orders_df)
print("Product details: \n")
print(products_df)
customers_df['age'].fillna(customers_df['age'].mean(),inplace=True)
customers_df['email'].fillna('N/A', inplace=True)
merged_df = pd.merge(pd.merge(customers_df, orders_df,on='customer_id'),products_df, on='product_id')
merged_df['total_price'] = merged_df['quantity'] * merged_df['price']
merged_df['Feed_back'] = np.where(merged_df['quantity']>1, "Good", "Bad")
print("Cleaned, Integrated, and Transformed Data:")
print(merged_df)
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
ordinal_encoder = OrdinalEncoder()
label_encode = LabelEncoder()
merged_ordinal_encoded = ordinal_encoder.fit_transform(merged_df)
feed_back_encoded = label_encode.fit_transform(merged_df['Feed_back'])
print("Features \n",merged_ordinal_encoded)
print("Target \n",feed_back_encoded)
