# Defect Classification

A defect classification model trained with pre-existing bugs and defects on a telecomunications CRM software with microservice architecture

## Tech

**Classifier:** Random Forest

- Random Forest Classifier was seen to return the highest success rate among many different model over a pre-labelled data stream.

**Vectorizer:** Term Frequency - Inverse Document Frequency

- TF-IDF vectorizer is used to minimize the weights of highest occuring common words on defect summaries, differentiating the defects. 

**Tokenizer:** Bart Tokenizer - Large

## Model Specifications

```
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
```
