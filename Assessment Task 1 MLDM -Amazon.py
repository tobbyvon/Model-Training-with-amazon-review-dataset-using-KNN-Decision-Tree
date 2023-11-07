#!/usr/bin/env python
# coding: utf-8

# # MLDM TASK 1

# ## Introduction

# The dataset worked with in this project is the Amazon Review.xlsx file, it consist of product reviews from Amazon.com, starting from the year 2008 to 2020, spanning across seven different domains, namely, book (Becoming by Michelle Obama), electronics (Echo Dot 3rd Gen by Amazon), grocery (Sparkling Ice Blue Variety Pack), and personal care (Nautica Voyage By Nautica). These datasets consist of 20000 reviews each.
# The objective is to classify target reviews and to predict if a reviews is positive or negative. This project will employ scikit-learn library for classification and KNN and Decision Tree classification algorithms is applied.

# ## Content

# The cleaning process of the Uncleaned student dataset includes:
# - importing the libraries
# - loading the dataset
# - Checking for number of columns and rows
# - Checking for missing values
# - Feature selection
# - converting categorical features into interger codes
# - Checking for outliers
# - checking for duplicate values
# - Define feature columns and the target column
# - Splitting the dataset into the Training set and the Test set
# - Scaling features
# - Scale the training data 
# - Training the model using KNN
# - Evaluating the performance of model
# - 

# ## IMPORTING LIBRARIES

# In[230]:


# importing libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading my dataset

# In[231]:


# Loading dataset
Amazon_data = pd.read_excel('Amazon Review.xlsx')


# ## Checking for number of columns and rows

# In[232]:


# Checking for number of columns and rows
Amazon_data.shape


# In[233]:


Amazon_data.head()


# # Evaluating my dataset

# In[234]:


#summary of dataset

Amazon_data.describe()


# In[235]:


Amazon_data.info()


# ## Checking for missing values 

# In[236]:


# Checking for missing values
Amazon_data.isnull().sum()


# ## Dealing with missing values

# In order to prevent my model from learning from noice and over fitting, improve my accuracy, and reduce my training time, I will remove the  asin, and Unnamed:6 column as a feature because most and if not all of there values are missing, and also they are not a relevant input to the model.

# In[237]:


Amazon_dataset = Amazon_data.drop( columns=(['asin','Unnamed: 6']))


# In[238]:


Amazon_dataset.head()

## NOTE: variable name change from Amazon_data to Amazon_dataset


# ## Checking for duplicated value

# In[239]:


# Checking for duplicated value
Amazon_dataset.duplicated(['reviews', 'text']).sum()


# I decided to check for duplicate values between the reviews column and the text column because I noticed that they contain similar strings but I soon realised that they are duplicated values. To achieve an accurate model I will also remove the text column as an input feature.

# In[240]:


Amazon_Dataset = Amazon_dataset.drop( columns=(['text']))


# In[241]:


Amazon_Dataset.head()

## NOTE: variable name change from Amazon_dataset to Amazon_Dataset


# ## Convert the categorical features to interger codes 

# In[242]:


# using LabelEncoder to transform categorical values
from sklearn.preprocessing import LabelEncoder

for col in ["reviews", "product name" ]:
    Amazon_Dataset[col] = LabelEncoder().fit_transform(Amazon_Dataset[col])
    
Amazon_Dataset


# In[243]:


# using the replace functions to transform categorical values
Amazon_Dataset["target"] = Amazon_Dataset["target"].replace(['n','p'],[0,1])
Amazon_Dataset.head()


# ## Checking for outliers

# In[244]:


# Checking for outliers using Boxplot
plt.boxplot(Amazon_dataset['ratings'])
plt.title('product ratings')
plt.show


# In[245]:


# statistical information about the rating column. Executing the below code, plot the product ratings on amazons website  
sns.histplot(Amazon_dataset.ratings)
plt.title('product ratings')
plt.show


# In[ ]:





# In[246]:


# Executing the below code, plot the product ratings based on the product name
plt.figure(figsize=(10,5))
plt.title('product ratings based on product name')
sns.histplot(x='ratings', hue='product name', data=Amazon_dataset)
plt.show()


# ### Define feature columns and the target column

# In[247]:


# Determining the class feature and input features
X = Amazon_Dataset[['product name', 'ratings', 'reviews', 'helpful']]
y = Amazon_Dataset['target']


# ## More Imports 

# In[248]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# ### Split the data into training and testing sets

# In[249]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### Scaling features

# In[250]:


# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Training the model using KNN

# In[252]:


# Create and train the KNN model
k = 3  # You can adjust the number of neighbors as needed
model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
model.fit(X_train, y_train)


# ### Make predictions on the test set

# In[253]:


# Predict using the test set
y_pred = model.predict(X_test)
print(y_pred)


# ## Evaluate the model

# In[254]:


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)


# In[258]:


# Getting the Confustion Matrix
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)

# Extract TP, TN, FP, FN from the confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]


print("Confusion Matrix:")
print(cm)


# ### using seaborn heatmap to visualise the confusion matrix

# In[259]:


ax = sns.heatmap(cm, cmap='flare', annot= True, fmt= 'd')
plt.xlabel('predicted class',fontsize=12)
plt.ylabel('True class',fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.show


# In[277]:


# Perform cross-validation for KNN model
from sklearn.model_selection import cross_val_score
knn_scores = cross_val_score(model, X, y, cv=5)  # You can adjust the number of folds (cv)

print("\nCross-Validation Scores for KNN Model:")
print(knn_scores)
print("Mean Accuracy: ", knn_scores.mean())


# # Training the model using Decision Tree algorithm

# In[261]:


from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
decision_tree_model.fit(X_train, y_train)


# ### Make predictions on the test set

# In[262]:


decision_tree_predictions = decision_tree_model.predict(X_test)
print(decision_tree_predictions)


# In[ ]:


Evaluate the model


# In[271]:


def evaluate_model(model_name, predictions):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"** {model_name} **")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
evaluate_model("Decision Tree", decision_tree_predictions)
   


# ### confusion matrix

# In[274]:


# getting the confusion matrix
confusion = metrics.confusion_matrix(y_test, decision_tree_predictions)

# Extract TP, TN, FP, FN from the confusion matrix
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


print("Confusion Matrix:")
print(confusion)


# ## using seaborn heatmap to visualise the confusion matrix

# In[275]:


ax = sns.heatmap(confusion, cmap='flare', annot= True, fmt= 'd')
plt.xlabel('predicted class',fontsize=12)
plt.ylabel('True class',fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.show


# In[278]:


# Perform cross-validation for Decision Tree model
decision_tree_scores = cross_val_score(decision_tree_model, X, y, cv=5)  # You can adjust the number of folds (cv)


# Print cross-validation scores
print("Cross-Validation Scores for Decision Tree Model:")
print(decision_tree_scores)
print("Mean Accuracy: ", decision_tree_scores.mean())


# # Conclusion

# Given that both models have achieved such high levels of accuracy and other performance metrics, it is challenging to distinguish between them based solely on their  results. An advanced evaluation techniques, cross-validation techniques to be precised, was futher used assess the models' robustness and based on the evaluation The Decision Tree model exhibits excellent accuracy but may be overfitting the training data, as evidenced by the perfect accuracy across all folds.
# The KNN model provides good but slightly less consistent accuracy, with a mean accuracy of approximately 0.8134. However, recommend that both algorithms would be appropriate for future deployment.

# In[ ]:




