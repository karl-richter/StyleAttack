import pytest
import logging

logger = logging.getLogger("cddm")


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def test_style_attack():
    params = {
        "model_name": "textattack/bert-base-uncased-SST-2",
        "orig_file_path": "./data/clean/sst-2/tiny.tsv",
        "model_dir": "./data/models/paraphraser_gpt2_large",
        "output_file_path": "./data/record.tsv",
        "p_val": 0.6,
        "iter_epochs": 10,
        "orig_label": 0,
        "bert_type": "bert-base-uncased",
        "output_nums": 2,
    }
    from attack import main

    main(params=dotdict(params))


def test_shap():
    params = {
        "model_name": "textattack/bert-base-uncased-SST-2",
        "attack_file_path": "./data/style_attack.tsv",
        "output_file_path": "./data/shap/",
        "num_sentences": 2,
        "bert_type": "bert-base-uncased",
    }
    from shap_values import main

    main(params=dotdict(params))
