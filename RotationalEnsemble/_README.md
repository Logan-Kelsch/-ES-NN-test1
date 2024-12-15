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