import pytest
import logging

from attack import main

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
        "orig_file_path": "../data/clean/sst-2/tiny.tsv",
        "model_dir": "../data/models/paraphraser_gpt2_large",
        "output_file_path": "../data/record.tsv",
        "p_val": 0.6,
        "iter_epochs": 10,
        "orig_label": 0,
        "bert_type": "bert-base-uncased",
        "output_nums": 2,
    }

    main(params=dotdict(params))
