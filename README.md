# Sentiment-Based Product Recommendation
### Problem Statement
The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm, and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products, and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

### Built With
Python
scikit-learn
xgboost
nltk
pandas
Flask
Bootstrap CDN
Solution Approach

### Dataset and Attribute Description
The dataset and its attributes are available under the dataset\ folder.
Data Cleaning, Visualization, and Text Preprocessing (NLP)
Applied on the dataset.
TF-IDF Vectorizer is used to vectorize the textual data (review_title + review_text). It measures the relative importance of the word with respect to other documents.
Class Imbalance Issue
The dataset suffers from a class imbalance issue, and the SMOTE Oversampling technique is used before applying the model.
Machine Learning Classification Models
### Models applied on the vectorized data and the target column (user_sentiment) include:
Logistic Regression
Naive Bayes
Tree Algorithms: Decision Tree, Random Forest, xgboost
The objective of these ML models is to classify the sentiment as positive (1) or negative (0).
The best model is selected based on various ML classification metrics: Accuracy, Precision, Recall, F1 Score, AUC.
xgboost is selected as the better model based on the evaluation metrics.
### Collaborative Filtering Recommender System
Created based on User-user and Item-item approaches.
RMSE evaluation metric is used for evaluation.
Implementation
SentimentBasedProductRecommendation.ipynb: Jupyter notebook contains the code for Sentiment Classification and Recommender Systems.
Top 20 products are filtered using the better recommender system.
For each of the top 20 products, predicted the user_sentiment for all the reviews and filtered out the top 5 products that have higher positive user sentiment (model.py).
### Deployment
Machine Learning models are saved in pickle files (under the pickle folder).
Flask API (app.py) is used to interface and test the Machine Learning models.
Bootstrap and Flask Jinja templates (templates\index.html) are used for setting up the user interface. No additional custom styles are used.


