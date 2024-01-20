import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from transformers import BartTokenizer

import util.DataFrameOperations as DataFrameOperations
import Constants
from vectorizers.TFIDFVectorizer import TFIDFVectorizer


def evaluate_model(model, vectorizer, x_train, x_test, y_train, y_test):
    model.fit(vectorizer.get_doc_term_matrix(x_train), y_train)

    # Rate
    score = model.score(vectorizer.get_transformation(x_test), y_test)
    print("Success Rate: " + str(score))

    print("Top 3 Accuracy Score: " + str(
        top_k_accuracy_score(y_test, model.predict_proba(vectorizer.get_transformation(x_test)), k=3)
    ))

    # Classification_Report
    print(str(metrics.classification_report(
        y_test,
        model.predict(vectorizer.get_transformation(x_test)))
    ))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(10, 5))
    predictions = model.predict(vectorizer.get_transformation(x_test))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
    plt.xticks(rotation=90)
    plt.title('Test Data')
    plt.show()


# Tokenize function to be fed into vectorizer
def tokenize(text):
    tokenized_text = bart_tokenizer.tokenize(text)
    return tokenized_text


def get_wrong_predictions(x_test, y_test, predictions):
    """
    Gets the wrong predictions to be evaluated in the future

    :@param x_test: test values
    :@param y_test: actual classifications
    :@param predictions
    """

    df = pd.DataFrame()
    df["x"] = x_test
    df["actual"] = y_test
    df["predictions"] = predictions

    incorrect_raw = df[df["actual"] != df["predictions"]]
    incorrect_filtered = pd.DataFrame(columns=["index", "x", "actual", "predictions"])

    # Check if the misclassification has been handled yet
    with open("sourcefiles/defect_checks.txt") as defect_file:
        pre_checked_row_numbers = [int(i) for i in defect_file]

        # Add if the misclassification does not previously exist
        for index, row in incorrect_raw.iterrows():
            if index in pre_checked_row_numbers:
                incorrect_raw.drop(index=index)

    # Write the misclassifications to the check file
    with open("sourcefiles/defect_checks.txt", "a") as defect_file:
        for index, row in incorrect_raw.iterrows():
            defect_file.write(str(index) + "\n")

    print(incorrect_raw.to_string())


def get_data_for_train_and_test():
    # Read the csv file and convert it into filtered data frame
    df_jira = DataFrameOperations.get_df_from_csv_file(
        Constants.JIRA_DATA_PATH,
        columns=["Summary", "Broader Classification"]
    )

    df_ag = DataFrameOperations.get_df_from_csv_file(
        Constants.REGENERATED_ARTIFICIAL_DATA_PATH,
        columns=["Summary", "Broader Classification"]
    )

    # Filter the data frame
    df_jira_final = DataFrameOperations.filter_csv_data(df_jira, "Summary", "Broader Classification")
    df_ag_final = DataFrameOperations.filter_csv_data(df_ag, "Summary", "Broader Classification")

    # Retrieve train and test values
    jira_x_train, jira_x_test, jira_y_train, jira_y_test = DataFrameOperations.split_dataset_properly(
        df_jira_final, split_size=0.2
    )

    x_train, x_test, y_train, y_test = DataFrameOperations.split_dataset_properly(
        df_ag_final, split_size=0.3, truncate=True
    )

    x_test = pd.concat((x_test, jira_x_test), axis=0)
    y_test = pd.concat((y_test, jira_y_test), axis=0)
    x_train = pd.concat((x_train, jira_x_train), axis=0)
    y_train = pd.concat((y_train, jira_y_train), axis=0)

    return x_train, y_train, x_test, y_test


# Bart Tokenizer - vectorizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_tokenizer_vectorizer = TFIDFVectorizer(tokenizer=tokenize)

# Ridge Classifier
# Score: 0.9049473684210526

"""
clf = RidgeClassifier(
    alpha=0.3,
    solver="lsqr",
    random_state=1,
)
evaluate_model(clf, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)
"""

# K Neighbours Classifier - k=19
# Score: 0.8568421052631578

"""
neigh = KNeighborsClassifier(n_neighbors=19)
evaluate_model(neigh, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)
"""

# Random Forest Classifier
# Score: 0.9505306552056837
# Score with real data: 0.6613226452905812
# Hyperparameters are tuned with random search tree regression
"""
Real Data:
                       precision    recall  f1-score   support

    algorithmic error       0.86      0.46      0.60        13
          build error       0.55      0.75      0.63        32
  communication error       0.50      0.33      0.40         3
data management error       0.50      0.49      0.49        82
     deployment error       1.00      0.33      0.50         3
         design error       1.00      1.00      1.00         2
    integration error       0.46      0.30      0.36        20
        logging error       1.00      1.00      1.00         5
      migration error       1.00      0.71      0.83         7
       patching error       0.65      0.63      0.64        27
    performance error       0.82      0.60      0.69        15
       resource error       0.57      0.80      0.67         5
       security error       0.84      0.83      0.84        59
  specification error       0.50      0.25      0.33         4
         system error       0.55      0.50      0.52        22
        testing error       1.00      1.00      1.00         4
             ui error       0.65      0.81      0.72        74
     validation error       0.69      0.72      0.71       107
     verfigback error       1.00      0.53      0.70        15

             accuracy                           0.66       499
            macro avg       0.74      0.63      0.67       499
         weighted avg       0.67      0.66      0.66       499
"""

x_train, y_train, x_test, y_test = get_data_for_train_and_test()

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    random_state=42,
    min_samples_split=8,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    bootstrap=False,
    class_weight="balanced"
)

evaluate_model(rf_classifier, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)

# MLP Classifier
# Score: 0.9227368421052632

"""
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=150,
    max_iter=1000,
    alpha=0.0005,
    solver="adam",
    activation="relu",
    learning_rate_init=0.0005,
    shuffle=True,
    warm_start=False,
    random_state=1,
    early_stopping=True,
    tol=0.00001
)

evaluate_model(mlp_classifier, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)
"""

# Gradient Boost Classifier
# Score: 0.8880799929830716

"""
gb_clf = GradientBoostingClassifier(
    n_estimators=300,
    subsample=1.0,
    loss="log_loss",
    min_samples_split=5,
    random_state=1,
    max_depth=1
)
evaluate_model(gb_clf, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)
"""

# Ada Boost Classifier
# Possible Config Problem
# Score: 0.26234540829751773

"""
ada_clf = AdaBoostClassifier(
    n_estimators=1000,
    algorithm="SAMME.R",
    random_state=42,
)
evaluate_model(ada_clf, bart_tokenizer_vectorizer, x_train, x_test, y_train, y_test)
"""
