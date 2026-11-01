# Credit Card Default Prediction

## a. Problem statement

This machine learning application takes as input client data for credit card holders and predicts if the client will default in the next month's payment.

The input is 30000 records of credit card holders of a bank. For each customer 23 features are specified and then there is prediction if customer will default in the next month payment.

The application splits the given data into train and test data and trains the following models using the training data.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Support Vector Machine

Upon completion of training, the model is saved and is used on the test data to generate the following evaluation metrics:

1. Accuracy
2. AUC score
3. Precision
4. Recall
5. F1 score
6. MCC score

Then we create a web based application which is hosted on Streamlit. The input to this application is the trained model. It provides a web based interface to the user to upload test data, select a model and determine the various evaluation metrics.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, "Default of Credit Card Clients"
- **Link:** <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- **Context:** Credit card clients of a bank in Taiwan, April to September 2005.
- **Size:** 30,000 clients
- **Number of features:** 23 input features
- **Target:** 1 binary target. `default_next_month` - 1 if the client defaulted on the October 2005 payment, 0 otherwise.

### Features (grouped)

1. **Credit limit (`LIMIT_BAL`):** Credit given to the client and family, in NT dollars.
2. **Demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`):**
   - Sex: 1 = male, 2 = female
   - Education: 1 = graduate school, 2 = university, 3 = high school, 4 = others
   - Marital status: 1 = married, 2 = single, 3 = others
   - Age in years
3. **Repayment status (`PAY_1` to `PAY_6`):** Monthly repayment status from September (`PAY_1`) back to April (`PAY_6`).
   - -1 = paid on time
   - 1 to 8 = number of months the payment was delayed
   - -2 = no credit used
   - 0 = revolving credit
4. **Bill amounts (`BILL_AMT1` to `BILL_AMT6`):** Monthly bill statement amount in NT dollars, September back to April.
5. **Payment amounts (`PAY_AMT1` to `PAY_AMT6`):** Amount paid each month in NT dollars, September back to April.

## c. GitHub repository details

- **GitHub link:** <https://github.com/amitsinha80/credit-default-classifier>
- **Live Streamlit app:** <https://credit-default-classifier-amit.streamlit.app/>

## d. Models used along with the comparison table

The output below is taken from running the Streamlit app on the test data (`test_data.csv`, 6,000 rows):

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.811    | 0.724 | 0.723     | 0.236  | 0.356 | 0.337 |
| Decision Tree       | 0.813    | 0.754 | 0.630     | 0.374  | 0.469 | 0.383 |
| KNN                 | 0.808    | 0.734 | 0.629     | 0.325  | 0.428 | 0.353 |
| Naive Bayes         | 0.376    | 0.688 | 0.248     | 0.894  | 0.388 | 0.128 |
| Support Vector Machine | 0.816 | 0.720 | 0.671     | 0.327  | 0.440 | 0.377 |

### Observation on performance of each model on the credit card dataset

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Highest precision (0.723): when it predicts a default it is usually right. But it has the lowest recall (0.236), missing about 3 in 4 actual defaulters. As a linear model it cannot capture the step-like effect of the repayment-status features, and the default 0.5 threshold is too strict for a 22% minority class. Accuracy (0.811) is barely above the 0.779 "always predict no default" baseline. |
| **Decision Tree** | Best AUC (0.754), F1 (0.469) and MCC (0.383), giving the best balance between catching defaulters and avoiding false alarms. Trees naturally learn threshold rules such as "recent payment delayed by 2+ months means high risk", which suit this data. Limiting the depth to 6 prevented overfitting, and training took under a second. |
| **KNN** | Consistently mid-table (AUC 0.734, F1 0.428, MCC 0.353). Distance-based classification weakens with 23 features and heavily skewed amount columns, even after scaling. It is fast to "train" since it only stores the data, but slow at prediction because each test row is compared with all 24,000 training rows. |
| **Naive Bayes** | The outlier: accuracy collapses to 0.376 because it predicts "default" for most clients, giving the highest recall (0.894) but the lowest precision (0.248). Its independence assumption is badly violated (the six bill amounts and the six repayment statuses are strongly correlated), and the Gaussian assumption does not fit the skewed amounts. AUC of 0.688 shows its ranking is still somewhat informative, but its probabilities are poorly calibrated. Lowest MCC (0.128). |
| **Support Vector Machine** | Highest accuracy (0.816) and second-best MCC (0.377), close to the Decision Tree. But it has the lowest AUC (0.720) and a recall of only 0.327, and it was by far the slowest to train (about 50 s versus under 1 s for the others), because kernel SVMs scale poorly with the number of rows. |
| **Overall, Winner for your dataset?** | Decision Tree. It leads on the three metrics that matter most for imbalanced data (AUC, F1, MCC) and is fast and interpretable. The SVM comes close on MCC and accuracy but at far higher training cost; accuracy alone would have misleadingly made the SVM look best. Recall is low for every model except Naive Bayes; class weighting or a lower decision threshold could raise it, at the cost of some precision. |