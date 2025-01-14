'''
-   Model Stacking and Prediction fusion
-   -   -   This could be done possibly by bagging or stacking methods already created
-   -   -   could also train an LSTM NN or other model on stacked model predictions for one best prediction
-   -   -   'meta-model'? could use linear regression for this or logistic regression
-   -   -   this I guess is called level-0 models vs level-1 model
'''

			#NOTE NOTE NOTE META MODEL IDEAS TO CONSIDER
			#TRY ALL. bi-log-reg / log-reg, naive-bayes, lin-reg, pop-vote, average, NN, LSTM NN, DT, timeseries DT
			#can start with just one, actually yeah just start with one BUT the option for all with notimplemented error for
			#incompleted methods, just to have a working method first! we are very close.

			#here is where different meta models will be created, can consider saving models at a different time,
			#im sure errors will arrise by this point, so try first to get up to and through meta model creation and performance
			#output before considering further, in execution should not take that long

from typing import Literal
import numpy as np

def pred_proba_fusion(
	models#			:	any
	,type_model		:	type
	,X_test			:	np.ndarray
	,fusion_type	:	Literal['Popular','Log_regression','Neural_Net'] = 'Popular'
)   ->    list:
	'''
		This function takes in a list of trained models
  		and an X dataset to predict. It then evaluates all 
		models on the provided test set, fuses them with
  		a prefferred method, and returns a list of predictions.
	'''
	probas = []
	for model in models:
		probas.append(model.predict_proba(X_test))

	proba_fusion = []

	#splitting here based on the method that will be used to combine predictions
	match(fusion_type):
     
     
		case 'Popular':
			for i in range(len(X_test)):
				vote_0, vote_1 = 0, 0
				for m in range(len(probas)):
					if(probas[m][i][0] > probas[m][i][1]):
						vote_0+=1
					else:
						vote_1+=1
				proba_fusion.append(0 if (vote_0 > vote_1) else 1)
    
		case 'Log_regression':
			raise NotImplementedError(f'FATAL: {fusion_type} has not been added to pred_proba_fusion.')
		case 'Neural_Net':
			raise NotImplementedError(f'FATAL: {fusion_type} has not been added to pred_proba_fusion.')

	return proba_fusion

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