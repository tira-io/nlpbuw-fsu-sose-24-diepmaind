from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()
    phrases = [
        "advancement in the realm",
        "aims to bridge",
        "aims to democratize",
        "aims to foster innovation and collaboration",
        "becomes increasingly evident",
        "behind the veil",
        "breaking barriers",
        "breakthrough has the potential to revolutionize the way",
        "bringing us",
        "bringing us closer to a future",
        "by combining the capabilities",
        "by harnessing the power",
        "capturing the attention",
        "continue to advance",
        "continue to make significant strides",
        "continue to push the boundaries",
        "continues to progress rapidly",
        "crucial to be mindful",
        "crucially",
        "cutting-edge",
        "drive the next big",
        "encompasses a wide range of real-life scenarios",
        "enhancement further enhances",
        "ensures that even",
        "essential to understand the nuances",
        "excitement",
        "exciting opportunities",
        "exciting possibilities",
        "exciting times lie ahead as we unlock the potential of",
        "excitingly",
        "expanded its capabilities",
        "expect to witness transformative breakthroughs",
        "expect to witness transformative breakthroughs in their capabilities",
        "exploration of various potential answers",
        "explore the fascinating world",
        "exploring new frontiers",
        "exploring this avenue",
        "foster the development",
        "future might see us placing",
        "groundbreaking way",
        "groundbreaking advancement",
        "groundbreaking study",
        "groundbreaking technology",
        "have come a long way in recent years",
        "hold promise",
        "implications are profound",
        "improved efficiency in countless ways",
        "in conclusion",
        "in the fast-paced world",
        "innovative service",
        "intrinsic differences",
        "it discovered an intriguing approach",
        "it remains to be seen",
        "it serves as a stepping stone towards the realization",
        "latest breakthrough signifies",
        "latest offering",
        "let’s delve into the exciting details",
        "main message to take away",
        "make informed decisions",
        "mark a significant step forward",
        "mind-boggling figure",
        "more robust evaluation",
        "navigate the landscape",
        "notably",
        "one step closer",
        "one thing is clear",
        "only time will tell",
        "opens up exciting possibilities",
        "paving the way for enhanced performance",
        "possibilities are endless",
        "potentially revolutionizing the way",
        "push the boundaries",
        "raise fairness concerns",
        "raise intriguing questions",
        "rapid pace of development",
        "rapidly developing",
        "redefine the future",
        "remarkable abilities",
        "remarkable breakthrough",
        "remarkable proficiency",
        "remarkable success",
        "remarkable tool",
        "remarkably",
        "renowned",
        "represent a major milestone",
        "represents a significant milestone in the field",
        "revolutionize the way",
        "revolutionizing the way",
        "risks of drawing unsupported conclusions",
        "seeking trustworthiness",
        "significant step forward",
        "significant strides",
        "the necessity of clear understanding",
        "there is still room for improvement",
        "transformative power",
        "truly exciting",
        "uncover hidden trends",
        "understanding of the capabilities",
        "unleashing the potential",
        "unlocking the power"
    ]

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

    # Constructing a regular expression pattern with 'or' separator
    pattern = '|'.join(phrases)

    # Applying str.contains with the pattern to check for any of the phrases, case insensitive
    prediction = (
        text_validation.set_index("id")['text']
        .str.contains(pattern, case=False)
        .astype(int)
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
