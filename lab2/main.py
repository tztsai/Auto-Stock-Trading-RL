# %%
import sys, os
import data
import logging
import argparse
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=['sts', 'nli'])
parser.add_argument('-ep', '--epochs', type=int, default=4)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('--max_seq_length', type=int, default=None)
parser.add_argument('--save_path', type=str, default=None)

if IPython.get_ipython():
    sys.argv = [__file__, 'reg']
args = parser.parse_args()

if args.save_path is None:
    args.save_path = os.path.join('output', 'sbert_' + args.type)
    
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# %% Load data
from torch.utils.data import DataLoader

if args.type == 'nli':
    dataset = data.load_nli()
else:
    dataset = data.load_sts()

train_samples, dev_samples, test_samples = [
    dataset[k] for k in ['train', 'dev', 'test']]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

logging.info("Train: %d samples (%d batchs), dev: %d samples, test: %d samples", len(train_samples), len(train_dataloader), len(dev_samples), len(test_samples))

# %% Build model
import losses
from sbert import SBert
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

model = SBert()

if args.type == 'nli':
    train_loss = losses.SoftmaxLoss(model=model)
else:
    train_loss = losses.CosineSimilarityLoss(model=model)

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
