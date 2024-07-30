Test the app - https://movie-reviews-sentiment-analysis-k6yk.onrender.com/predict 

(You may have to wait a bit for render to "render" the webpage)

# IMDB Movie Reviews Sentiment Analysis Project
## Introduction: 
Sentiment analysis, a subfield of natural language processing (NLP), aims to determine the sentiment expressed in a piece of text, whether it's positive or negative. In this project, we leverage the IMDB movie reviews dataset from Kaggle to predict the sentiment of movie reviews. The dataset contains 50000 reviews along with their corresponding sentiment labels.

## Objective: 
The objective of this project is to develop a sentiment analysis model that accurately predicts the sentiment of movie reviews. We aim to explore various text preprocessing techniques, text vectorization methods, and machine learning models to achieve this goal.

## Methodology: 
### Data Collection and Preprocessing:
- The IMDB movie reviews dataset from Kaggle is used as the primary data source. Dataset - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- The text data is preprocessed by lowercasing, removing HTML tags, URLs, emails, formatting, special characters, punctuation, stopwords, and performing tokenization.
- Text normalization technique stemming is applied to reduce word variations.
### Feature Engineering and Text Vectorization:
- Various text vectorization methods are explored, including Bag-of-Words (BoW), N-grams, and TF-IDF (Term Frequency-Inverse Document Frequency).
- These methods convert text data into numerical vectors suitable for machine learning models.
### Model Building: 
- Several machine learning algorithms are experimented with, including Logistic Regression, Bernoulli and Multinomial Naive Bayes, Random Forests, and Gradient Boosting models (e.g., XGBoost, LightGBM).
### Model Evaluation: 
- The performance of each model is evaluated using metrics such as accuracy, F1-score, and confusion matrix.
- Cross-validation techniques are used to assess the generalization ability of the models.
### Web Application Development: 
- Once a satisfactory model is trained, a web application is developed using Flask.
- This app is then deployed on an EWS EC2 instance.
- The app allows users to input a movie review text, upon which the trained model predicts the sentiment (positive or negative) of the review.
- The app provides a user-friendly interface for real-time sentiment analysis of movie reviews.
## Results: 
- The model achieves 90% accuracy in predicting the sentiment of movie reviews.
- The web application provides a convenient way for users to analyze the sentiment of movie reviews on the fly.
## Conclusion: 
In conclusion, this project demonstrates the successful application of NLP techniques for sentiment analysis of IMDB movie reviews. By exploring various text preprocessing methods, text vectorization techniques, and machine learning models, we have developed an effective sentiment analysis model. The web application built using Flask provides a user-friendly interface for real-time sentiment analysis, which can be useful for movie enthusiasts, critics, and filmmakers alike.
