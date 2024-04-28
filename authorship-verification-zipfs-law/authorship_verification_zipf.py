import pandas as pd
import re
from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import CountVectorizer


def calculate_weighted_score(text_df: pd.Series, word_probabilities):
    """ Calculate a weighted score based on the word probabilities for each text. """
    scores = []
    for words in text_df.values:
        word_freq = {word: words.count(word) / len(words) for word in words}

        score = 1  # Start assuming the text is fine
        for word, prob in word_probabilities.items():
            if word in word_freq:
                if not are_numbers_close(prob, word_freq[word], tolerance=0.01):
                    score = 0  # Set score to 0 if any crucial word frequency is off
                    break
        scores.append(score)
    return scores

def are_numbers_close(num1, num2, tolerance=0.05):
    """ Check if two numbers are within a certain tolerance. """
    return abs(num1 - num2) < num1 * tolerance

if __name__ == "__main__":

    tira = Client()
    word_probabilities = {
        "the": 6.09, "of": 2.63, "to": 2.44, "a": 2.25, "and": 2.18,
        "in": 2.13, "said": 1.27, "for": 0.92, "that": 0.87, "was": 0.74,
        "on": 0.73, "he": 0.63, "is": 0.62, "with": 0.56, "at": 0.53,
        "by": 0.53, "it": 0.49, "from": 0.48, "as": 0.46, "be": 0.40,
        "were": 0.39, "an": 0.38, "have": 0.38, "his": 0.36, "but": 0.35,
        "has": 0.34, "are": 0.33, "not": 0.32, "who": 0.29, "they": 0.28,
        "its": 0.28, "had": 0.26, "will": 0.26, "would": 0.25, "about": 0.23,
        "i": 0.23, "been": 0.22, "this": 0.22, "their": 0.21, "new": 0.21,
        "or": 0.21, "which": 0.20, "we": 0.20, "more": 0.19, "after": 0.19,
        "us": 0.18, "percent": 0.18, "up": 0.18, "one": 0.18, "people": 0.17
    }


    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    # Preprocess text
    text_validation = text_validation.assign(processed_text=text_validation["text"].str.replace(r'[^\w\s]', '/', regex=True).str.split('/'))


    # classifying the data
    prediction =pd.Series(calculate_weighted_score(text_validation.set_index("id")["processed_text"], word_probabilities))

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # Save the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    text_validation.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
