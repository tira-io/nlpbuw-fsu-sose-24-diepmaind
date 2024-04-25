from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (optional, based on your need)
    text = pd.Series(text).str.replace('[^\w\s]', '', regex=True)
    return text

# mp = [{1:0}{1:2}]
# mp[0][1]

def calculate_weighted_score(row, word_probabilities):
    """ Calculate a weighted score based on the word probabilities. """
    list_words = [row[i]['processed_text'].split() for i in range(len(row)) ]
    list_mp = [1] * len(list_words)
    for i in range(len(list_words)):
        mp = {}
        n = len(list_words[i])
        for word in list_words[i]:
            if word in mp:
                mp[word] += 1
            else:
                mp[word] = mp.get(word,0) + 1
        for key in mp.keys():
            mp[key] /= n
        for key in word_probabilities.keys():
            if not are_numbers_close(word_probabilities[key],mp[key],tolerance=0.05):
                list_mp[i] = 0
    return list_mp

def are_numbers_close(num1, num2, tolerance=0.01):
    return abs(num1 - num2) < tolerance



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
    for i in range(len(text_validation)):
        text_validation[i]['processed_text'] = text_validation[i]["text"].apply(preprocess_text) 

    # Applying str.contains with the pattern to check for any of the phrases, case insensitive
    prediction = calculate_weighted_score(text_validation, word_probabilities)

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
