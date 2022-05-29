import logging
import numpy as np
import pandas as pd
import shap
import torch
from style_paraphrase.inference_utils import GPT2Generator
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger("attackdetect")
logger.setLevel(logging.INFO)


def load_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()
    return model


def read_data(file_path):
    import pandas as pd

    data = pd.read_table(file_path).values.tolist()
    return data


def write_data(attack_data, output_file_path):
    import pandas as pd

    pd.DataFrame(
        attack_data,
        columns=[
            "result_type",
            "original_text",
            "perturbed_text",
            "original_output",
            "perturbed_output",
        ],
    ).to_csv(output_file_path, sep="\t", index=False)


def get_predict_label(model, sent, tokenizer):
    inputs = tokenizer(sent, return_tensors="pt", padding=True)
    output = model(
        inputs["input_ids"].cuda()
        if torch.cuda.is_available()
        else inputs["input_ids"].cpu()
    )[
        0
    ].squeeze()  # attention_masks=inputs["attention_mask"].cpu()
    predict = torch.argmax(output).item()
    return predict


def predict_labels(data):
    import numpy as np
    import math

    model = load_model(model_name="textattack/bert-base-uncased-SST-2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize inputs
    embeddings = tokenizer(list(data), return_tensors="pt", padding=True).input_ids
    # Predict
    prediction = (
        model(embeddings.cuda() if torch.cuda.is_available() else embeddings.cpu())
        .logits.detach()
        .cpu()
        .numpy()
    )
    scores = (np.exp(prediction).T / np.exp(prediction).sum(-1)).T
    val = [1 / (1 + math.exp(-x)) for x in scores[:, 1]]
    return val


def transform_attack_data(adv_examples):
    # Filter out skipped and unsuccessful attacks
    adv_examples = adv_examples[adv_examples["result_type"] == "Successful"]

    adv_x_test = np.array(adv_examples["perturbed_text"])
    adv_y_test = np.array(adv_examples["original_output"])

    # adv_x_test = tokenizer(
    #    list(adv_x_test), return_tensors="pt", padding=True
    # ).input_ids

    # adv_y_test = list(adv_y_test)
    return adv_x_test, adv_y_test


def main(params: dict):
    orig_data = read_data(params.orig_file_path)
    victim_model = load_model(model_name=params.model_name)
    tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
    paraphraser = GPT2Generator(params.model_dir, upper_length="same_5")

    mis = 0
    total = 0
    attack_data = []
    paraphraser.modify_p(params.p_val)

    for sent, label in tqdm(orig_data):
        # filter for params.orig_label to only data for the target label (aka directed attack)
        # if (
        #    label != params.orig_label
        #    or get_predict_label(victim_model, sent, tokenizer) != params.orig_label
        # ):
        #    continue
        # flag indicates success of the attack
        flag = False
        logger.info(sent)
        generated_sent = [sent for _ in range(params.iter_epochs)]
        paraphrase_sentences_list = paraphraser.generate_batch(generated_sent)[0]
        for paraphrase_sent in paraphrase_sentences_list:
            logger.info(paraphrase_sent)
            predict = get_predict_label(victim_model, paraphrase_sent, tokenizer)
            if predict != label:
                attack_data.append(
                    ("Successful", sent, paraphrase_sent, label, predict)
                )
                flag = True
                mis += 1
                # can be improved, in the paper the authors indicate they use sentenceBERT
                # to select the best paraphrase, here they only select the first that succeeds
                break
        if not flag:
            attack_data.append(("Failed", sent, sent, label, label))
        logger.info("------------------")

        total += 1
    logger.info(f"Completed attack, success rate: {mis}/{total}")

    # checkpoint the attack data
    write_data(attack_data=attack_data, output_file_path=params.output_file_path)
    attack_data = pd.read_table(params.output_file_path)

    # transform the attack data to be used for SHAP
    adv_x_test, adv_y_test = transform_attack_data(attack_data)
    x_test, y_test = (
        np.array(orig_data)[:, 0].tolist(),
        np.array(orig_data)[:, 1].tolist(),
    )

    # initialise the SHAP explainer
    explainer = shap.Explainer(predict_labels, tokenizer)

    # calculate the SHAP values
    orig_shap_values = explainer(x_test, fixed_context=1)
    adv_shap_values = explainer(adv_x_test, fixed_context=1)

    np.save("/content/orig_shapvals.npy", orig_shap_values)
    np.save("/content/adv_shapvals.npy", adv_shap_values)
