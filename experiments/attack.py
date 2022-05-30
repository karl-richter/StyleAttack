import logging
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.utils import shuffle
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
    """
    attack_data = pd.read_table(params.output_file_path)

    # transform the attack data to be used for SHAP
    orig_x_test, orig_y_test, adv_x_test, adv_y_test = transform_attack_data(
        attack_data
    )

    # initialise the SHAP explainer
    explainer = shap.Explainer(predict_labels, tokenizer)

    # calculate the SHAP values
    orig_shap_values = explainer(orig_x_test, fixed_context=1)
    adv_shap_values = explainer(adv_x_test, fixed_context=1)

    orig_shap_padded = pad_embeddings(orig_shap_values)
    adv_shap_padded = pad_embeddings(adv_shap_values)

    shap_padded = np.concatenate((orig_shap_padded, adv_shap_padded))

    orig_labels = np.zeros(orig_shap_padded.shape[0])
    adv_labels = np.ones(adv_shap_padded.shape[0])

    labels = np.concatenate((orig_labels, adv_labels))

    X, Y = shuffle(shap_padded, labels, random_state=42)

    np.save("/content/drive/MyDrive/NLP-Lab/SHAP/shap_vals.npy", X)
    np.save("/content/drive/MyDrive/NLP-Lab/SHAP/shap_labels.npy", Y)
    """
