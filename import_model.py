import math

import joblib
from transformers import BartTokenizer

DEFECT_TYPES = ["Build Error",
                "Data Management Error",
                "Logging Error",
                "Migration Error",
                "Patching Error",
                "Performance Error",
                "Security Error",
                "System Error",
                "Testing Error",
                "UI Error",
                "Validation Error",
                "Verfigback Error"
                ]


def deserialize_with_joblib(filename):
    obj = joblib.load(filename)
    print("Obj loaded from file: " + filename)
    return obj


def tokenize(text):
    tokenized_text = bart_tokenizer.tokenize(text)
    return tokenized_text


def get_percentage_rep(percentage):
    box_count = math.ceil(percentage / 2)
    percentage_rep = box_count * "☐"
    return percentage_rep


def evaluate_summary(summary, model, vectorizer):
    predict_proba = model.predict_proba(vectorizer.get_transformation([summary]))
    # Adjust percentages
    predict_proba_adj = list(map(lambda x: round(x * 100, 1), predict_proba[0]))

    prediction_dict = dict()

    for i in range(len(DEFECT_TYPES)):
        prediction_dict[DEFECT_TYPES[i]] = predict_proba_adj[i]

    sorted_prediction_dict = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n=========================================================================================================")
    print("Summary: " + summary + "\n")

    print("Classifications:")
    for i in range(5):
        print("{:<30} {:<10} {:<30} ".format(sorted_prediction_dict[i][0], str(sorted_prediction_dict[i][1]) + "%", get_percentage_rep(sorted_prediction_dict[i][1])))
    print("=========================================================================================================\n")


# Deserialize objects for evaluation
model_from_joblib = deserialize_with_joblib("models/random_forest_model.pkl")
vectorizer_from_joblib = deserialize_with_joblib("models/bart_vectorizer_for_random_forest.pkl")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

vectorizer_from_joblib.__setattr__("tokenizer", tokenize)

summary = input("\nEnter a defect summary: ")

while summary != "q":
    evaluate_summary(
        summary,
        model_from_joblib,
        vectorizer_from_joblib
    )

    summary = input("Enter a defect summary: ")
