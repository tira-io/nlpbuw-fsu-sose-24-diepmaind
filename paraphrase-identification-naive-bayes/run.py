from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    sentences = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")


    # Compute the Levenshtein distance
    # df["distance"] = levenshtein_distance(df)
    # df["label"] = (df["distance"] <= 10).astype(int
    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    sentences['combined_sentences'] = sentences['sentence1'] + " " + sentences['sentence2']
    predictions = model.predict(sentences['combined_sentences'])
    sentences["labels"] = predictions
    sentences = sentences[["id", "labels"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
