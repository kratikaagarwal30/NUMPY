#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("healthcare_dataset.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


#The dataset does not contain any null values and only two columns are numerical


# In[6]:


pip install skimpy


# In[7]:


from skimpy import skim


# In[8]:


skim(df)


# QUESTIONS OF NUMPY

# In[18]:


#1. How can you convert the 'Age' column into a NumPy array?
age_array = df["Age"].values
print(age_array)
print(type(age_array))


# In[19]:


#The above question can also be solved using this function
age_array = df["Age"].to_numpy()
print(age_array)
print(type(age_array))


# In[10]:


#2. Write a NumPy command to calculate the mean age of patients.
df["Age"].mean()


# In[12]:


#3. How can you find the maximum and minimum 'Billing Amount' using NumPy?
a = np.max(df["Billing Amount"])
b = np.min(df["Billing Amount"])
print("The maximum billing amount is :", a)
print("The minimum billing amount is :", b)


# In[20]:


#4. Create a boolean mask that identifies female patients in the dataset.
female_mask = df["Gender"] == "Female"
female_mask


# In[21]:


#5. How can you select the 'Name' and 'Medical Condition' columns and convert them into a NumPy array?
name_medical = df[["Name", "Medical Condition"]].to_numpy()
print(name_medical)
print(type(name_medical))


# In[22]:


#6. Write a command to randomly shuffle the rows of the dataset using NumPy.
#numpy.random.shuffle shuffles the array in place, meaning it changes the order of the array itself. When working with pandas DataFrames, it's often safer and more convenient to shuffle the indices instead and then reindex the DataFrame. 
shuffled_indices = np.arange(len(df))
np.random.shuffle(shuffled_indices)
shuffled_df = df.iloc[shuffled_indices]
shuffled_df


# In[23]:


#7. How do you compute the standard deviation of the 'Billing Amount' for the dataset?
df["Billing Amount"].std


# In[25]:


#8.Create a NumPy array representing the 'Date of Admission' and convert it into a datetime64 type array.
#firstly converting Date of Admission into numpy array
df_array = df["Date of Admission"].to_numpy()
df_array_datetime64 = df_array.astype("datetime64")
df_array_datetime64


# In[26]:


#9. How can you find the unique 'Insurance Provider' values and their counts in the dataset?
df["Insurance Provider"].unique()


# In[37]:


#10. Write a NumPy command to filter patients aged between 25 and 50 years.
filtered_df = df[(df['Age'] >= 25) & (df['Age'] <= 30)]

# If you want a Series of ages
filtered_ages = filtered_df['Age']

# To display the filtered ages
print(filtered_ages)


# In[27]:


#11. How can you replace all 'Test Results' values from 'Normal' to 'OK' using NumPy?
df["Test Results"].replace("Normal", "OK")


# In[28]:


#11. Calculate the age difference between the youngest and oldest patients in the dataset.
age_oldest = df["Age"].max()
age_youngest = df["Age"].min()
difference = age_oldest - age_youngest
difference


# In[45]:


#12. Create a new column 'Days in Hospital' by calculating the difference between 'Discharge Date' and 'Date of Admission'. Use NumPy datetime operations
# Convert the 'Discharge Date' and 'Date of Admission' columns to datetime64[ns] and assign back
df["Discharge Date"] = df["Discharge Date"].astype("datetime64[ns]")
df["Date of Admission"] = df["Date of Admission"].astype("datetime64[ns]")

# Now, calculate 'Days in Hospital' by subtracting 'Date of Admission' from 'Discharge Date'
df["Days in Hospital"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

df


# In[47]:


#13. How can you group patients by 'Gender' and calculate the average 'Billing Amount' for each group?
df.groupby("Gender")["Billing Amount"].mean()


# In[53]:


#14. Using NumPy, how can you identify patients with no 'Medication' prescribed (assuming empty values are represented as 'None' or an empty string)?
df["no_medication"] = df["Medication"] == "None"
df


# In[50]:


#15. Write a NumPy command to sort the dataset based on 'Billing Amount' in descending order.
df["Billing Amount"].sort_values(ascending = False)


# In[54]:


#16. How can you extract the month from the 'Date of Admission' column and create a new array with it?
df["month"] = df["Date of Admission"].dt.month
df


# In[56]:


#17. Implement a method to normalize the 'Billing Amount' column (subtract the mean and divide by the standard deviation).
df["normalized_billing_amount"] = (df["Billing Amount"] - df["Billing Amount"].mean()) / df["Billing Amount"].std()
df


# In[57]:


#18. How can you perform a one-hot encoding of the 'Medical Condition' column using NumPy?
oh = pd.get_dummies(df, columns = ["Medical Condition"])
oh


# In[58]:


#19. Create a pivot table-like structure showing the average 'Billing Amount' for each combination of 'Insurance Provider' and 'Admission Type'.
# Create the pivot table
pivot_table = df.pivot_table(
    values='Billing Amount', 
    index='Insurance Provider', 
    columns='Admission Type', 
    aggfunc=np.mean,
    fill_value=0 # Fill missing values with 0
)

# Display the pivot table
print(pivot_table)


# In[59]:


#20. How can you create a weighted average of 'Billing Amount' where weights are the number of days spent in the hospital?
df["weighted average"] = np.average(df["Billing Amount"], weights = df["Days in Hospital"])
df


# In[63]:


#21. How can you calculate the correlation coefficient between 'Age' and 'Billing Amount'?
df["Age"].corr(df["Billing Amount"])


# In[64]:


import pandas as pd
import matplotlib.pyplot as plt

# Convert 'Date of Admission' to datetime format
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])

# Sort the DataFrame by 'Date of Admission'
df = df.sort_values(by='Date of Admission')

# Plot 'Billing Amount' over 'Date of Admission'
plt.figure(figsize=(10, 6))
plt.plot(df['Date of Admission'], df['Billing Amount'], marker='o', linestyle='-', color='b')
plt.title('Trend of Billing Amount Over Time')
plt.xlabel('Date of Admission')
plt.ylabel('Billing Amount')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

plt.show()


# In[ ]:




