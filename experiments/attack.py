import logging
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

    data = pd.read_csv(file_path, sep="\t").values.tolist()
    return data


def write_data(attack_data, output_file_path):
    with open(output_file_path, "w") as f:
        print("p_val", "\t", "orig_sent", "\t", "adv_sent", "\t", "labels", file=f)
        for p_val, orig_sent, adv_sent, label in attack_data:
            print(p_val, "\t", orig_sent, "\t", adv_sent, "\t", label, file=f)


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
    victim_model = load_model(model_name=params.model_name)
    orig_data = read_data(params.orig_file_path)
    tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
    paraphraser = GPT2Generator(params.model_dir, upper_length="same_5")

    mis = 0
    total = 0
    attack_data = []
    paraphraser.modify_p(params.p_val)

    for sent, label in tqdm(orig_data):
        if (
            label != params.orig_label
            or get_predict_label(victim_model, sent, tokenizer) != params.orig_label
        ):
            continue

        flag = False
        logger.info(sent)
        generated_sent = [sent for _ in range(params.iter_epochs)]
        paraphrase_sentences_list = paraphraser.generate_batch(generated_sent)[0]
        for paraphrase_sent in paraphrase_sentences_list:
            logger.info(paraphrase_sent)
            predict = get_predict_label(victim_model, paraphrase_sent, tokenizer)
            if predict != label:
                attack_data.append((1, sent, paraphrase_sent, label))
                flag = True
                mis += 1
                break
        if flag:
            pass
        else:
            attack_data.append((-1, sent, sent, label))
        logger.info("------------------")

        total += 1
    logger.info(f"Completed attack, success rate: {mis}/{total}")
    write_data(attack_data=attack_data, output_file_path=params.output_file_path)
