'''
-   Model Stacking and Prediction fusion
-   -   -   This could be done possibly by bagging or stacking methods already created
-   -   -   could also train an LSTM NN or other model on stacked model predictions for one best prediction
-   -   -   'meta-model'? could use linear regression for this or logistic regression
-   -   -   this I guess is called level-0 models vs level-1 model
'''

'''
NOTE NOTE NOTE NOTE
GPT example metamodel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Example base model predictions (n_samples, n_models)
base_model_preds = np.array([[0.2, 0.6, 0.8],
                              [0.9, 0.7, 0.3],
                              [0.1, 0.2, 0.5]])

# Actual labels
y = np.array([1, 0, 0])

# Split meta-data into train/test for stacking
X_train, X_test, y_train, y_test = train_test_split(base_model_preds, y, test_size=0.2, random_state=42)

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(X_train, y_train)

# Predict using meta-model
final_predictions = meta_model.predict(X_test)
print(final_predictions)

...
# Predict probabilities
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1

# Predict binary outcomes (default threshold = 0.5)
y_pred = log_reg.predict(X_test)
...

END#NOTE END#NOTE END#NOTE'''