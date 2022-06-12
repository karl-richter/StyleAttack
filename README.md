# [Fork] StyleAttack

> **_NOTE:_** Original code and data of the EMNLP 2021 paper "**Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer**" [[pdf](https://arxiv.org/abs/2110.07139)]

This repository extends the original attack repo with an adversarial detector trained on SHAP values.  
We aim to benchmark the capabilities of the detector in discriminating between adversarial and original samples.

### Style Attack
Run a StyleAdv attack on a given dataset using a pre-trained victim model from huggingface and a pre-trained GPT-2 based paraphrasing model.
```bash
! CUDA_VISIBLE_DEVICES=0 python StyleAttack/experiments/cli.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path StyleAttack/data/clean/sst-2/dev.tsv --model_dir /content/drive/MyDrive/style_transfer_paraphrase/models/paraphraser_gpt2_large --output_file_path /content/drive/MyDrive/NLP-Lab/StyleAttack/style_attack.tsv
```
### SHAP
Leverage the pre-trained model to derive SHAP-values of the original and pertubed samples of examples where the attack succeeded. Store SHAP-values for training.
```bash
! CUDA_VISIBLE_DEVICES=0 python StyleAttack/experiments/cli_shap.py --attack_file_path /content/drive/MyDrive/NLP-Lab/StyleAttack/style_attack.tsv  --output_file_path /content/drive/MyDrive/NLP-Lab/StyleAttack/ --num_sentences 5
```
### Detector
Train an adversarial detector to descriminate original and pertubed examples.
```bash
! CUDA_VISIBLE_DEVICES=0 python StyleAttack/experiments/cli_detector.py --shap_files_path /content/drive/MyDrive/NLP-Lab/StyleAttack/
```

## Citation

Please cite the original paper:

```
@article{qi2021mind,
  title={Mind the style of text! adversarial and backdoor attacks based on text style transfer},
  author={Qi, Fanchao and Chen, Yangyi and Zhang, Xurui and Li, Mukai and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2110.07139},
  year={2021}
}
```

