# %%
import torch.nn as nn
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Winter is coming."
encoded_input = tokenizer(text, return_tensors='pt')
output_tokens = model(**encoded_input).last_hidden_state

# %%
