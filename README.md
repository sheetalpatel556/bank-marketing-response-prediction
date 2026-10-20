# Bank Marketing Response Prediction

## Project Overview
This machine learning project predicts whether a customer is likely to subscribe to a bank term deposit based on customer and campaign-related information.

The project includes data analysis, preprocessing, model comparison, threshold selection, and a Streamlit web application for interactive predictions.

## Live Demo
[Click here to try the live application](https://sheetalpatel556-bank-marketing-response-prediction-app-gz3lbn.streamlit.app/)

## Dataset
The project uses the Bank Marketing dataset from the UCI Machine Learning Repository.

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Records:** 4,521
- **Features:** 15 input features
- **Target:** Term deposit subscription (`yes` or `no`)

The `duration` feature was excluded to avoid using call duration information that would not be available before a call.

## Machine Learning Workflow
1. Data loading and exploratory data analysis
2. Data cleaning and preprocessing
3. Stratified train-test split
4. Categorical encoding and numerical scaling
5. Training and comparison of multiple classification models
6. Hyperparameter tuning and decision-threshold selection
7. Evaluation using precision, recall, F1-score, accuracy, and ROC-AUC
8. Saving the trained model and deploying the application

## Final Model
The deployed application uses a Gradient Boosting classifier with a decision threshold of **0.16**, selected to improve the balance between precision and recall.

Test-set performance:
- Accuracy: **84.3%**
- Precision: **36.0%**
- Recall: **47.1%**
- F1-score: **40.8%**
- ROC-AUC: **77.1%**

These metrics are from the held-out test set. Predictions are estimates and are not guarantees of customer behavior.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit
- Google Colab
- GitHub

## Run Locally
1. Clone this repository.
2. Install the dependencies:

   `pip install -r requirements.txt`

3. Run the Streamlit app:

   `streamlit run app.py`

## Repository Files
- `app.py` — Streamlit web application
- `bank_marketing_model.pkl` — Saved model and decision threshold
- `requirements.txt` — Python dependencies

## Disclaimer
This project is for educational and portfolio purposes only. It should not be used as the sole basis for real-world banking or customer-targeting decisions.
