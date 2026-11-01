# Credit Card Default Prediction

## a. Problem statement

This machine learning application takes as input client data for credit card holders and predicts if the client will default in the next month's payment.

The input is 30000 records of credit card holders of a bank. For each customer 23 features are specified and then there is prediction if customer will default in the next month payment.

The application splits the given data into train and test data. Training script trains and saves the models, and the Streamlit web app only loads them. Following models are trained:

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

| ML Model Name          | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression    | 0.811    | 0.724 | 0.723     | 0.236  | 0.356 | 0.337 |
| Decision Tree          | 0.813    | 0.754 | 0.630     | 0.374  | 0.469 | 0.383 |
| KNN                    | 0.808    | 0.734 | 0.629     | 0.325  | 0.428 | 0.353 |
| Naive Bayes            | 0.376    | 0.688 | 0.248     | 0.894  | 0.388 | 0.128 |
| Support Vector Machine | 0.816    | 0.720 | 0.671     | 0.327  | 0.440 | 0.377 |

### Observation on performance of each model on the credit card dataset

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The Logistic Regression model gives the highest precision of 0.723, which means that when it predicts a client will default, the prediction is usually correct. However, it has the lowest recall of 0.236, which means it fails to identify many actual default cases. Since Logistic Regression is a linear model, it cannot properly capture the sudden changes in default risk associated with different repayment statuses. Also, the default threshold of 0.5 may be too high when only 22% of clients belong to the default class. Its accuracy of 0.811 is only slightly better than the baseline accuracy of 0.779, which can be achieved by simply predicting that no client will default. |
| **Decision Tree** | The decision tree has the best AUC of **0.754**, F1 of **0.469** and MCC of **0.383**. This kind of gives the best balance between catching defaulters and avoiding false alarms. Decision trees naturally learn threshold rules such as "recent payment delayed by 2+ months means high risk", which suit this data. Limiting the depth to 6 prevented overfitting, and training took under a second. |
| **KNN** | The model gives average performance (AUC 0.734, F1 0.428, MCC 0.353). Since there are 23 features and the amount columns are highly skewed, distance-based classification does not work very well even after scaling. Training is fast because the model only stores the training data. However, prediction is slow because each test sample needs to be compared with all 24,000 training samples. |
| **Naive Bayes** | The model performs poorly with an accuracy of only 0.376 because it predicts "default" for most clients. As a result, recall is very high (0.894), but precision is very low (0.248). The model assumes that all features are independent, which is not true in this dataset because the six bill amounts and six repayment statuses are strongly correlated. Also, the amount columns are highly skewed and do not follow a Gaussian distribution. Although the AUC is 0.688, which indicates some ability to distinguish between classes, the predicted probabilities are not very reliable. The model also has the lowest MCC of 0.128. |
| **Support Vector Machine** | The model gives the highest accuracy (0.816) and second-highest MCC (0.377), which is close to the Decision Tree. However, it has the lowest AUC (0.720) and a low recall of 0.327, which means it fails to identify many actual default cases. Training is also very slow, taking around 50 seconds compared to less than 1 second for the other models. This is because kernel-based SVMs become computationally expensive when the number of training samples increases. |
| **Overall, Winner for your dataset?** | Overall, the Decision Tree is the best performing model because it gives the highest AUC, F1 score and MCC, which are important metrics for imbalanced datasets. It is also fast to train and easy to interpret. The SVM performs almost as well in terms of MCC and gives higher accuracy, but its training time is much higher. This also shows why accuracy alone should not be used to select the best model, especially for imbalanced data. Except for Naive Bayes, all models have low recall, which means they fail to identify many actual default cases. Recall can be improved by using class weights or reducing the decision threshold, but this may also reduce precision. |