# A lightweight Machine Learning Model to Predict Brain Stroke
This is our Final Year Project which is based on Machine learning Algorithm Max-Voting (Ensemble method)
Seven Classifiers are used and Merged together to one Final Model(Max-Voting Classifier)
1. Decision Tree
2. Logistic Regression
3. Random Forest
4. K-Nearest Neighbours
5. Support Vector Machine
6. Gaussian Naïve Bayes
7. XGBoost

# Final Application Preview:

https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/c8ebe782-36b6-4b8a-bdcc-09509e620d7a

![Screenshot 2024-04-06 223635](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/f1cfdb65-06be-4c4e-9050-a6b729db796c)
![Screenshot 2024-04-06 223754](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/8bef4aaf-2717-4cb9-89bb-ebb2163d706f)
![Screenshot 2024-04-06 223717](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/88db4368-8b57-49ac-990f-cd95bb0af3c2)



# Model Source Code Preview:

https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/93676529-3ba3-4536-bba6-fa84d09d2382


# Abstract:
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


# Comparision of Algorithms

![Screenshot 2024-04-06 215742](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/77aa8169-7348-465c-b248-81fe535a05dc)


# Graphs of Final Results:
1. Accuracy:
   ![Accuracy](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/f3ff47b4-69fa-49a6-9038-d80a0c56847c)

2. Precission:
   ![Screenshot 2024-02-29 135529](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/71d279cd-1e0b-434c-93c6-e4b3ae4f1748)

3. Recall:
   ![Screenshot 2024-02-29 135553](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/7967508f-4006-46b3-9718-72faa7f84721)

4. F1-Score:
   ![F1](https://github.com/danishnawab55/Brain_Stroke_Prediction/assets/88277249/320740f9-a4b7-42da-90d3-25238a601428)



## Contributors 🍉

Thanks goes to these wonderful people ([:hugs:](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/danishnawab55">
                    <img src="https://avatars.githubusercontent.com/u/88277249?v=4" width="100px;" alt="Danish Nawab"/>
                    <br />
                    <sub><b>Danish Nawab</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Sujyoti03">
                    <img src="https://avatars.githubusercontent.com/u/98302735?v=4" width="100px;" alt="Sujyoti Nam Singh"/>
                    <br />
                    <sub><b>Sujyoti Nam Singh</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Yutsav1">
                    <img src="https://avatars.githubusercontent.com/u/94309077?v=4" width="100px;" alt="Yutsav Hari Bhagat"/>
                    <br />
                    <sub><b>Yutsav Hari Bhagat</b></sub>
                </a>
            </td>
        </tr> 



# copyright©️ 2024, All Right
DANISH NAWAB, SUJYOTI NAM SINGH, YUTSAV HARI BHAGAT, NIDHISH RANJAN





