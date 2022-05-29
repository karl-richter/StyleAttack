import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)

model.cuda() if torch.cuda.is_available() else model.cpu()
print(model.eval())
