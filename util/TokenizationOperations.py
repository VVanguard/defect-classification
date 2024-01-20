from nltk import word_tokenize
from nltk import defaultdict, FreqDist
import util.TextOperations as TextOperations
import Constants


def get_tokens(text):
    """
    Generates tokens from a given text, while ignoring the stopwords
    :param stop_words: (str[]) Stop words to ignore as tokens
    :param text: (str) text to be tokenized
    :returns tokens: (str[])
    """

    # Generate stop words
    stop_words = Constants.generate_stopwords()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def tokenize_df(df):
    """
    Tokenizes words in each text located in a specific data frame
    :param df: (Pandas.DataFrame) source data frame that includes texts to be tokenized
    :returns tokens: (defaultdict)
    """

    # Create a dict like object for tokens
    tokens = defaultdict(list)

    # Iterate through each text and its corresponding label
    for i in df.index:
        label = df["Category"][i]
        text = df["Summary"][i]

        # Text clean up
        text = TextOperations.clean_text(text)

        # Get tokens that are not stopwords
        text_tokens = get_tokens(text)
        tokens[label].extend(text_tokens)

    # Print frequency distribution of top-20 tokens
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))

    return tokens
