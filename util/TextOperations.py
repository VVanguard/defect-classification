""" Text Operations utility file for various text cleaning and filtering applications """
import string

import Constants


def clean_text(text):
    """
    Cleans a given text by doing filtering operations

    :param text: (str) text to be cleaned

    :returns filtered_text: (str)
    """

    filtered_text = remove_punctuations(text)
    filtered_text = remove_non_ascii(filtered_text)
    filtered_text = remove_stop_words(filtered_text)
    return filtered_text


def remove_punctuations(text):
    """
    Removes punctuations from the text

    :param text: (str)

    :returns filtered_text: (str)
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def remove_non_ascii(text):
    """
    Removes non-ASCII characters from the text

    :param text: (str)

    :returns new string:
    """
    return "".join(c for c in text if ord(c) < 128)


def remove_stop_words(text):
    """
    Removes stopwords from the text

    :param text: (str)

    :returns new string:
    """
    text_tokens = text.split(" ")

    for word in text_tokens:
        if word in list(Constants.generate_stopwords()):
            text_tokens.remove(word)

    return " ".join(text_tokens)
