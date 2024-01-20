from nltk.corpus import stopwords

# Dataset csv documents
COMPLETE_ARTIFICIAL_DATA_PATH = "sourcefiles/CompleteDataset.csv"
JIRA_DATA_PATH = "sourcefiles/MergedJiraDefects.csv"
MERGED_ARTIFICIAL_DATA_PATH = "sourcefiles/ArtificialDefectsBora.csv"
YARKIN_ARTIFICIAL_DATA_PATH = "sourcefiles/CompleteDatasetCleanedYarkin.csv"
REGENERATED_ARTIFICIAL_DATA_PATH = "sourcefiles/RegeneratedArtificialDefects.csv"

# Specific stop words
specific_stop_words = [

]

def generate_stopwords():
    """
    Generates stop words for token filtering

    :returns stop_words:
    """

    stop_words = set(stopwords.words("english"))

    # Add specific stop words
    for word in specific_stop_words:
        stop_words.add(word)

    return stop_words
