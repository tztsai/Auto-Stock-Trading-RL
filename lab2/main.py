# %%
import sys, os
import data
import logging
import argparse
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=['clf', 'reg'])
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_seq_length', type=int, default=None)
parser.add_argument('--save_path', type=str, default=None)

if IPython.get_ipython():
    sys.argv = [..., 'reg']
args = parser.parse_args()

if args.save_path is None:
    args.save_path = os.path.join('output', 'sbert_' + args.type)

# %% Load data
from torch.utils.data import DataLoader

if args.type == 'clf':
    dataset = data.load_nli()
else:
    dataset = data.load_sts()

train_samples, dev_samples, test_samples = [
    dataset[k] for k in ['train', 'dev', 'test']]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

# %% Build model
import losses
from sbert import SBert
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

model = SBert()

if args.type == 'clf':
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
else:
    train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')

# %% Train the model
warmup_steps = len(train_dataloader) * args.epochs // 10
logging.info("Warmup-steps: {}".format(warmup_steps))
evaluation_steps = len(train_dataloader) // 10

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=args.epochs,
          evaluator=evaluator,
          warmup_steps=warmup_steps,
          evaluation_steps=evaluation_steps,
          output_path=args.save_path)

# %% Evaluate the model
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test')
test_evaluator(model, output_path=args.save_path)
