### This folder with contain:
-   Functions and Notebooks for a custom rotational ensemble model.
-   A rotational ensemble model consists of building models off of random different sets of features.
-   This is then brought into an ensemble for a fractional improvement on accuracy. 

#### Looks like the amount that can be adjusted for a model like this is pretty vast, will have to construct a plan.

-   Data Preprocessing - - Should end in returning split and reformatted datasets (train/val/ind)
-   -   -   This should be as standard as the previous versions, but broken down into functional code for readability and malleability

-   Data Rotating      - - Should end with an array of 'rotations' of data for n models from the train split
-   -   -   This should have a few options, pick random from all, option to keep some features in all models, ... 
-   -   -   -   ... random pick but category specific, overlapping or model unique features

-   ModelSet Training  - - Training all models on their set split of data
-   -   -   the double rotation model will be built first, an optimal-hyperparameter-set seeking algorithm should be built for this
-   -   -   LSTM NN will have to come later

-   Model Stacking and Prediction fusion
-   -   -   This could be done possibly by bagging or stacking methods already created
-   -   -   could also train an LSTM NN or other model on stacked model predictions for one best prediction
-   -   -   'meta-model'? could use linear regression for this or logistic regression
-   -   -   this I guess is called level-0 models vs level-1 model

-   Model evaluation
-   -   -   Copy over the printout functions of previous versions
-   -   -   consider adding some sort of proba printouts?

#### KEY FOR FOLLOWING INFORMATION:
-   k:  number of model types, or numbers of models in a given model-set.
-   n:  number of rotated-feature-sets, or numbers of rotation on the training set.
-   m:  number of sample-sets, the partitioned sets can be shuffled because they do not interact(but probably shouldnt be).
-   c:  consider this, now, with the following line. (also it serves as a smiley face)
## CURRENT LARGE SCALE PLAN
-   At this indentation I will specify the DIMENSIONS OF PREDICTIONS INVOLVED. (WILL BE IMPORTANT)
-   non-rotation method (no meta-model, no ensemble): 
-   0dim: singular prediction from singular model
- c: in the look-forward, is SMOTE essential? what about limiting to binary classification.
-   -   -   try to incorporate SMOTE to balance classes if finding balancing issues (ONLY IF NON LSTM)
- c: form a working meta-model before building hyperparameter-seeker?
-   -   -   Initiate project by using 1 model type (rotation forest? DT for easy testing?) with hyperparameter-seeker for each model on all rotations.
-   rotational method with one base estimator (meta-model now required):
-   1dim: a set of n rotated-feature-set model predictions
-   -   -   -   Program this so that it can be expanded to multiple (pre-specified) types of models!!!!!!
-   rotational method with k different base estimators (log-reg should still suffice, may now consider a NN):
-   2dim: a set of n rotated-feature-sets CONTAINING a set of k model-types' predictions.
- c: create multiple methods anyways? are all models of equal compatibility!?!?
-   -   -   Use logistic regression as meta-model OR neural network as meta model
-   -   FROM HERE, training is only partitioned along features. For each n rotation of features and its k model-types,
- c: partition with shuffling? section vs sliding window? how to assert that?
-   -   consider implementing sample partitioning. This would create a grid of m sample-sets X n rot-feat-sets X k model-types.
- c: use fully connected? use conv? tree based? THREE DIMENSION OF PREDICTIONS! is there conv3d??
-   -   -   If this were to be done.. think twice about what type of meta-model would be required for understanding the 3D grid of predictions.
-   double rotational method with k different base estimators (NN? LSTM? Conv? What to even do here, this is 3D predictions):
-   3dim: a set of n rotated-feature-sets partitioned into m different sample sets, all CONTAINING a set of k model-type's predictions.