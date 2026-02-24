# Wikipedia Text Classifier

## Project Overview
This project focuses on building a machine learning pipeline to classify Wikipedia articles into 15 thematic categories based on their content.

The dataset is unbalanced, with articles per category ranging from 243 to 8,202, and average article lengths from 2,509 to 13,212 words. F1-score is prioritized over accuracy to evaluate model performance effectively.

The project includes:

- Exploratory Data Analysis (EDA) and dataset cleaning  
- Text preprocessing and feature engineering  
- Handling unbalanced classes  
- Implementation and comparison of multiple classification models:
  - Logistic Regression (basic)
  - Logistic Regression with class weights and L2 regularization
  - Naive Bayes
  - Random Forest
- Model evaluation using weighted F1-score  
- Detailed class-level performance analysis  

## Business Objective
The goal is to develop a model capable of accurately predicting the category of a Wikipedia article.  

Key considerations:

- Misclassification of underrepresented categories (e.g., *Politics*) is likely but should be minimized  
- Computational efficiency is essential when processing full-length articles, influencing model choice  

## Dataset
The dataset contains **73,370 Wikipedia articles** after cleaning, with the following columns:

- `_c0`: Article ID  
- `title`: Article title  
- `summary`: Short summary  
- `documents`: Full article content  
- `categoria`: Article category (target variable)  

### Example categories:
`['sports', 'science', 'technology', 'politics', 'pets', 'finance', 'engineering', 'trade', 'research', ...]`  

### Dataset URL:
[Download CSV](https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv)

## Exploratory Data Analysis
Key insights:

- Dataset contains 73,370 articles after removing duplicates and missing values  
- Highly unbalanced distribution across 15 categories  
- Average article lengths vary significantly  
- Word clouds highlight frequent terms per category, revealing content patterns  

Visualizations include:

- Bar plots of article counts per category  
- Average words per article per category  
- Longest and shortest article lengths  
- Word clouds per category  

## Preprocessing
Text preprocessing pipeline:

1. Lowercasing (`documents_lower`, `summary_lower`)  
2. Removing numbers and special characters (`documents_clean`, `summary_clean`)  
3. Tokenization and stopword removal  
4. Feature extraction using **CountVectorizer**  
5. Standardization for Logistic Regression and Random Forest  

## Models Implemented

### Logistic Regression
- Basic model with `multinomial` family  
- Weighted classes + L2 regularization to handle unbalanced data  
- Achieved **F1-score ~0.83 on test set** with minimal overfitting  

### Naive Bayes
- No class weighting  
- F1-score ~0.78 on test set  

### Random Forest
- Weighted classes, 50 trees, max depth=7  
- F1-score ~0.68 on test set  

### Best Model
- Logistic Regression with weighted classes and L2 regularization  
- Can be trained on `documents` or `summary` (summary preferred for computational efficiency)  
- Provides strong generalization and class-level performance  

## Evaluation Metrics
- Weighted F1-score used as primary metric  
- Class-level metrics calculated for all categories  
- Underrepresented categories (e.g., 'Politics') show lower F1, as expected  

## Key Learning Points
- Handling unbalanced datasets in multiclass classification  
- Feature engineering for text-based ML models  
- Comparing model performance beyond accuracy  
- Balancing computational cost vs. model quality  

## Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn, WordCloud  
- PySpark MLlib for scalable text processing and classification  
- Scikit-learn utilities for evaluation 
