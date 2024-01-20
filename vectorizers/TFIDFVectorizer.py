from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    """ Tf-idf vectorizer using scikit-learn tfidf vectorizer """

    def __init__(self, tokenizer=None):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            min_df=0.00005,
            sublinear_tf=True,
            tokenizer=tokenizer,
            token_pattern=None
        )

    def get_doc_term_matrix(self, x):
        """ Returns a doc term matrix to be used in model training """
        return self.vectorizer.fit_transform(x)

    def get_transformation(self, x):
        """ Returns a transformation matrix to be used in model testing """
        return self.vectorizer.transform(x)
