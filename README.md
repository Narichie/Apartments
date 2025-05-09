# Employing a classification problem to accurately predict whether or not apartments allow pets.

## Introduction
In this machine learning project, I focus on predicting whether apartments permit pets using various models. The dataset used for this analysis is the Apartment for Rent Classified dataset, which can be accessed at https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified. The main objective of this project is to tackle a classification problem, aiming to accurately determine whether or not specific apartments allow pets.
## Description
The project commences with the establishment of objectives, followed by the importation of relevant libraries and the uploading of the apartments dataset. Key libraries utilized include Pandas for data manipulation, Matplotlib for data visualization, and various models from scikit-learn, such as Logistic Regression, Random Forest Classifier, Decision Tree Classifier, and Support Vector Classifier, for the implementation of classification algorithms.
## Exploring and understanding the data
The procedure commences with the retrieval of apartment data from a CSV file titled apartment.csv. Subsequently, an examination of the dataset's head and tail is conducted to assess its structure and content. The data is then integrated into a Pandas DataFrame, and any identified missing values are eliminated to preserve the dataset's integrity and accuracy.
## Look at the Data
A thorough visual inspection of the dataset was performed to better understand its structure and key characteristics. This analysis provided valuable insights into the data's composition and overall quality.
## Show the data visually
To begin with, the numerical variables were visualized using histograms and boxplots, followed by the visualization of categorical variables through count plots. Additionally, a heat map was generated, along with a pair plot, to examine the relationships among the variables.
## Preprocessing the Data
In this analysis, I focus on selecting columns with data types 'object' or 'category,' as these often contain non-numerical information. I also be dropped certain columns that are deemed unnecessary for this analysis.
## List of columns to be dropped
columns_to_drop = ['latitude', 'longitude', 'time', 'category', 'title', 
                   'body', 'amenities', 'currency', 'fee', 'price_display', 
                   'price_type', 'address', 'cityname']

## Dropping the specified columns from the DataFrame
df = df.drop(columns=columns_to_drop, axis=1)
Once the preprocessing was completed, I took another look at both the numerical and categorical variables to visualize them clearly.

## Splitting the Data
X = df.drop('pets_allowed', axis=1)
y = df['pets_allowed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

I identified both numerical and categorical features in the dataset. After that, I implemented imputation and standardization techniques for the numerical features to ensure they were appropriately scaled and complete. Lastly, I applied one-hot encoding to the categorical features to facilitate their inclusion in the analysis.

## Choosing a Model
I will fit the following models:
•	Logistic Regression
•	Decision Tree
•	Random Forest
•	Gradient Boosting (GBM)
•	Support Vector (SVC)

## Training the Models 
I initially trained a Logistic Regression model, which generated a confusion matrix and displayed key metrics: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). Additionally, a comprehensive classification report was provided. This approach was subsequently applied to the remaining models as well.

## Camparing Model Performance
Finally, all five models were compared to evaluate their performance and determine which one was the most effective.

The evaluation of model performance reveals that Random Forest (Model 2) stands out as the most effective model for predicting whether apartments allow pets. Here are the key reasons for this conclusion:

**Highest Accuracy**: Random Forest achieves an accuracy rate of 0.71, which is the highest among all tested models.

**Balanced True Positive and False Positive Rates**: The model maintains a commendable true positive rate of 0.80 while keeping the false positive rate at a reasonable 0.42.

 **Low False Negative Rate**: With a false negative rate of 0.20, Random Forest is less likely to overlook apartments that permit pets compared to other models.

Although some models might exhibit slightly better false positive or true negative rates, Random Forest provides the best overall balance for accurately predicting pet-friendly apartments. 

# Follow me
www.linkedin.com/in/richarda112822




