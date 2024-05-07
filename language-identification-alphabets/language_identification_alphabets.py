from pathlib import Path
from aaransia import get_alphabets_codes
from aaransia import get_alphabets
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory


if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    alphabets = get_alphabets()
    alphabet_codes = get_alphabets_codes()
    mp = {}

    # Iterate through each alphabet
    for alphabet, code in zip(alphabets, alphabet_codes):
        unique_chars = []
        # Check each character in the alphabet
        for char in alphabet:
            # If the character is unique, add it to the list
            if all(char not in other_alphabet for other_alphabet in alphabets if other_alphabet != alphabet):
                unique_chars.append(char)
        mp[code] = unique_chars

    alph_prop = []
    for code in mp.keys():
        alphabet = mp[code]
        counts = pd.Series(0, index=text_validation.index, name=code)
        for char in alphabet:
            counts += (
                text_validation["text"]
                .str.contains(char, regex=False, case=False)
                .astype(int)
            )
        alph_prop.append(counts / len(alphabet))
    alph_prop = pd.concat(alph_prop, axis=1)

    prediction = alph_prop.idxmax(axis=1)

    # converting the prediction to the required format
    prediction.name = "lang"
    prediction = prediction.to_frame()
    prediction["id"] = text_validation["id"]
    prediction = prediction[["id", "lang"]]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
