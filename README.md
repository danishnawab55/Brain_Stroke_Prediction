# A lightweight Machine Learning Model to Predict Brain Stroke
This is our Final Year Project which is based on Machine learning Algorithm Max-Voting (Ensemble method)

Abstract: 
The Brain is considered as a powerhouse of the human body and it is body’s most sophisticated organ. Brain stroke is an urgent medical situation in which the blood arteries get ruptured causing damage to the brain cells. The World Health Organization (WHO) states that stroke is the primary cause of both death and dis-ability on a global scale. The chances of recovery of stroke victims increase if we predict stroke based on symptoms and it will also reduce the severity of the stroke. This research aims to develop a lightweight machine-learning model that can correctly predict brain stroke at an early stage. To measure the effectiveness of the model we used a reliable dataset from the Kaggle website for stroke predic-tion. The key contribution of this work is to introduce a model that employs the Max Voting Ensemble method to improve the prediction process. The classifiers used were: Decision Tree, Logistic Regression, Random Forest, K-Nearest Neighbour, Naïve Bayes, Support Vector Machine (SVM), and XGBoost . These classification models were integrated into the final Max Voting model us-ing the sklearnVoting Classifier, and the output is determined by the class receiv-ing the highest number of votes. The performance of this proposed model has been evaluated using Accuracy, Precision, Recall, and f1 score. The outcomes il-lustrate that the suggested Max voting model surpasses the individual classifiers, achieving an accuracy of 94%, and good overall precision, recall, and f1 score compared to the other individual classifiers.

# Proposed Algorithm of Final Prediction Model

Step 1: Exploratory Analysis. - Check data shape (5110 rows, 12 columns) and count of missing values (201 in 'bmi' column) 
data.isnull().sum()
Step 2: Fill Null Values. - Impute missing 'bmi' values with the mean using Sim-pleImpute. And dropped the 'id' column.
	imputer = SimpleImputer(strategy='mean') data['bmi'] = imputer.fit_transform(data[['bmi']])
Step 3: Outlier Removal. - Visualize and identify outliers in the dataset through box plots.

Step 4: Label Encoding. - Convert categorical variables into numerical values using LabelEncoder from sklearn.preprocessing.

Step 5: Splitting the Data for Training and Testing. - Separate the dataset into training and testing sets:
 X ---train_X,test_X 80/20 
 Y ---train_Y,test_Y
Step 6: Normalize. - Standardize the features using StandardScaler from sklearn.preprocessing based on the training data. 
X_train_std=std.fit_transform(X_train) X_test_std=std.transform(X_test)
Step 7: Training. - Train various classifiers and evaluate their performances:
Decision Tree, Logistic Regression, KNN, Random Forest, Naive Bayes, SVM, and XGBoost were trained. Performance metrics including accuracy, precision, recall, and F1-score were computed for each model.

Step 8: Final Model - Max Voting. 
Utilized a Voting Classifier with estimators including Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, and XGBoost.
The final model was trained on the training data: 
final_model.fit(X_train, Y_train)
Predictions were made on the standardized test data: 
pred_final = final_model.predict(X_test_std)
Accuracy, precision, recall, and F1-score were computed for the final model.


