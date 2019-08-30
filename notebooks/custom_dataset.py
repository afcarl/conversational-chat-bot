#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import logging
import random
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from pytorch_transformers import AdamW, WEIGHTS_NAME, CONFIG_NAME


import json
import torch
import torch.nn.functional as F
from itertools import chain
from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer, cached_path, GPT2Config
from pytorch_transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIGPTConfig


# In[2]:


#model = GPT2DoubleHeadsModel(GPT2Config()) #.from_pretrained('gpt2')
#tokenizer = GPT2Tokenizer(vocab_file='../data/ql_dataset_vocab.json', merges_file='../data/ql_dataset_vocab.json')  #.from_pretrained('gpt2')

model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

len(tokenizer)


# In[3]:


# We will use 5 special tokens:
# - <bos> to indicate the start of the sequence
# - <eos> to indicate the end of the sequence
# - <speaker1> to indicate the beginning and the tokens of an utterance from the user
# - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
# - <pad> as a padding token to build batches of sequences
special_tokens = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'additional_special_tokens': ['<speaker1>', '<speaker2>'],
    'pad_token': '<pad>'
}

# We can add these special tokens to the vocabulary and the embeddings of the model:
tokenizer.add_special_tokens(special_tokens)
#model.config.num_special_tokens = len(special_tokens)
model.resize_token_embeddings(len(tokenizer))


# In[4]:




# Let's define our contexts and special tokens
persona = [["i", "like", "playing", "football", "."],
           ["i", "am", "from", "NYC", "."]]
history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def build_inputs(persona, history, reply):
    # Build our sequence by adding delimiters and concatenating
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s
                                for i, s in enumerate(sequence[1:])]
    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))                          # word tokens
    segments = [speaker2 if i % 2 else speaker1             # segment tokens
                for i, s in enumerate(sequence) for _ in s]
    position = list(range(len(words)))                      # position tokens
    return words, segments, position, sequence

words, segments, position, sequence = build_inputs(persona, history, reply)

# >>> print(sequence)  # Our inputs looks like this:
# [['<bos>', 'i', 'like', 'playing', 'football', '.', 'i', 'am', 'from', 'NYC', '.'],
#  ['<speaker1>', 'hello', 'how', 'are', 'you', '?'],
#  ['<speaker2>', 'i', 'am', 'fine', 'thanks', '.'],
#  ['<speaker1>', 'great', 'to', 'hear', '<eos>']]

# Tokenize words and segments embeddings:
words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)


# In[5]:


# Let's add a distractor to our previously defined persona, history and reply
distractor = ["sorry", "to", "hear", "that"]

# Build & tokenize inputs ending with our distractor like we did with the gold reply
words_distractor, segments_distractor, _, _ = build_inputs(persona, history, distractor)
words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)

# Prepare our language modeling targets: keep only the reply segment, -1 on the rest
lm_targets = ([-1] * sum(len(s) for s in sequence[:-1]))              + [-1] + tokenizer.convert_tokens_to_ids(sequence[-1][1:])
lm_distractor = [-1] * len(words_distractor)

# Store the position of the last tokens for the next-sentence prediction loss
last_token = len(words) - 1
last_token_distractor = len(words_distractor) - 1

# Now we can pad reply and distractor inputs and targets to the same length
padding_length = max(len(words), len(words_distractor))
def pad(x, padding):
    return x + [padding] * (padding_length - len(x))

(words, words_distractor,
 segments, segments_distractor) = [pad(x, tokenizer.convert_tokens_to_ids('<pad>'))
                                   for x in (words, words_distractor,
                                             segments, segments_distractor)]

(lm_targets, lm_distractor) = [pad(x, -1) for x in (lm_targets, lm_distractor)]
 
# And gather reply and distractor inputs to build the input tensors:
# words tokens
input_ids = torch.tensor([[words, words_distractor]], dtype=torch.long)
# segment tokens
token_type_ids = torch.tensor([[segments, segments_distractor]], dtype=torch.long)
# Positions tokens can be automatically created by the model as (0, 1, ..., N)
# Last tokens location
mc_token_ids = torch.tensor([[last_token, last_token_distractor]], dtype=torch.long)
# Language modeling labels
lm_labels = torch.tensor([[lm_targets, lm_distractor]], dtype=torch.long)
# Next-sentence prediction labels
mc_labels = torch.tensor([0], dtype=torch.long)  # Gold reply is 1st (index 0)


# In[6]:


# Forward pass
lm_loss, mc_loss, _, _, _ = model(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)

# Total loss as a weighted sum
lm_coef = 2.0
mc_coef = 1.0
total_loss = lm_loss * lm_coef + mc_loss * mc_coef
total_loss


# # Trying to train with a custom dataset

# In[7]:


ql_dataset = {
    'personality': ['My name is John', 'I am a mortgage banker at Quicken Loans',
                   'I want to provide you with a mortgage.'],
    'utterances': [
        {
            'candidates': ['hi ! i have three kids . how many do you have ?',
                           'awesome ! i own 2 dogs , love them',
                           'yes , my favorite is broccoli and tofu in a garlic sauce . yum !',
                           'maybe he can skydive to see a better view',
                           'Hi there. Can I help you with a mortgage?'],
            'history': ['Hi']
        },
        {
            'candidates': ['poetry . roses are red . violet are . . . ?',
                          'my father is a member of the army , served for 10 years now .',
                          'oh i like mexican food , but my favorite food are cheeseburgers',
                          'hey there , are you a mother ?',
                          'Fantastic! Can I get your full name?'],
            'history': ['Hi', 
                        'Hi there. Can I help you with a mortgage?',
                        'Yes that would be great.']
        },
        {
            'candidates': ['awesome ! i own 2 dogs , love them',
                           'yes , my favorite is broccoli and tofu in a garlic sauce . yum !',
                           'maybe he can skydive to see a better view',
                           'i am good , i just got off work and tired , i have two jobs .'
                           'Thank you Zack. Can I get your email address?'],
            'history': ['Hi', 
                        'Hi there. Can I help you with a mortgage?',
                        'Yes that would be great.',
                        'Fantastic! Can I get your full name?',
                        'My name is Zack Jones.']
        },
        {
            'candidates': ['why have you not sent help ? ! the scorpions are stinging my legs ',
                           'that is great i am expecting twins in two months . will these be your first kids ?',
                           'do you live on a farm or ranch ?',
                           'hi how are you doing tonight i am fine .',
                           "i'd love to see her do that .",
                           'That is perfect! I will go ahead and send you an application to get started. Have a great day!'],
            'history': ['Hi', 
                        'Hi there. Can I help you with a mortgage?',
                        'Yes that would be great.',
                        'Fantastic! Can I get your full name?',
                        'My name is Zack Jones.',
                        'My email address is zackjones@gmail.com',
                        'Thank you, good bye!']
        }
    ]
}

ql_dataset = {
    'train': [ql_dataset],
    'valid': [ql_dataset]
}

with open('../data/ql_dataset.json', 'w') as file:
    json.dump(ql_dataset, file)
    file.close()


# In[8]:


# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


# In[9]:


with open('../data/ql_dataset_vocab.json', 'r') as file:
    vocab = json.load(file)
    file.close()


# In[10]:


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


# In[11]:


def average_distributed_scalar(scalar, local_rank=-1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def get_data_loaders(tokenizer, dataset_path='../data/ql_dataset.json', 
                     dataset_cache='../data/ql_dataset.json', num_candidates=2, 
                     personality_permutations=1, max_history=2, distributed=False,
                    train_batch_size=2, valid_batch_size=2):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, dataset_path, dataset_cache)

    #logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates_ = len(dataset[0]["utterances"][0]["candidates"])
        if num_candidates_ > 0 and dataset_name == 'train':
            num_candidates_ = min(num_candidates, num_candidates_)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    #logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    #logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, shuffle=(not distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=valid_batch_size, shuffle=False)

    #logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    #logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


# In[12]:


def get_dataset(tokenizer, dataset_path='../data/ql_dataset.json', 
                 dataset_cache='../data/ql_dataset.json'):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        #logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        #logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        #logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        #logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        #logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        #logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    #logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    #logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# In[13]:


def train(
    distributed=False, local_rank=-1, lr = 6.25e-5, dataset_path='../data/personachat_self_original.json', 
    dataset_cache=cached_path('../data/personachat_self_original.json'),
    model_checkpoint='gpt2', num_candidates=2, max_history=5, train_batch_size=2, valid_batch_size=2,
    gradient_accumulation_steps=8, lm_coef=1.0, mc_coef=1.0, max_norm=1.0, n_epochs=10, 
    personality_permutations=1, eval_before_start=False, device = 'cuda' if torch.cuda.is_available() else 'cpu',
    fp16=''
    ):
    '''
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()
    
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))
    '''
    
    args = None
    
    # Initialize distributed training if needed
    distributed = (local_rank != -1)
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    print(f'{datetime.now()}: Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning')
    
    model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # We will use 5 special tokens:
    # - <bos> to indicate the start of the sequence
    # - <eos> to indicate the end of the sequence
    # - <speaker1> to indicate the beginning and the tokens of an utterance from the user
    # - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
    # - <pad> as a padding token to build batches of sequences
    special_tokens = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'additional_special_tokens': ['<speaker1>', '<speaker2>'],
        'pad_token': '<pad>'
    }

    # We can add these special tokens to the vocabulary and the embeddings of the model:
    tokenizer.add_special_tokens(special_tokens)
    #model.config.num_special_tokens = len(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    #logger.info("Prepare datasets")
    print(f'{datetime.now()}: prepare datasets')
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        
        lm_loss, mc_loss, _, _, _ = model(*batch)
        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / gradient_accumulation_steps
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            #logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            print(f'{datetime.now()}: {tokenizer.decode(input_ids[0, -1, :].tolist())}')
            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, lr), (n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    #metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
    #                "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir='../logs')
        #tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        #tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        #tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        #checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        #trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, 
        #                          {'mymodel': getattr(model, 'module', model)})  
        # "getattr" take care of distributed encapsulation

        #torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        #getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        #tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    #if local_rank in [-1, 0] and n_epochs > 0:
    #    os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
    #    tb_logger.close()


# In[14]:


train(dataset_path='../data/ql_dataset.json', 
    dataset_cache=cached_path('../data/ql_dataset.json'), n_epochs=100, train_batch_size=10, valid_batch_size=10)


# In[22]:


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    
    model.to(args['device'])
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args['max_length']):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args['device']).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args['device']).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args['model']:
            logits = logits[0]
        logits = logits[0, -1, :] / args['temperature']
        logits = top_filtering(logits, top_k=args['top_k'], top_p=args['top_p'])
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args['no_sample'] else torch.multinomial(probs, 1)
        if i < args['min_length'] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        #logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        #logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        #logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    #logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    #logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities


# In[23]:


args = {
    'max_length': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': 'gpt2',
    'top_k': 0,
    'top_p': 0.9,
    'no_sample': True,
    'min_length': 2,
    'temperature': 0.7,
    'max_history': 2
}


# In[24]:


get_dataset_personalities(tokenizer, dataset_path='../data/ql_dataset.json', dataset_cache='../data/ql_dataset.json')


# In[25]:


personalities = get_dataset_personalities(tokenizer, dataset_path='../data/ql_dataset.json', 
                                          dataset_cache='../data/ql_dataset.json')


# In[26]:


personality = random.choice(personalities)


# In[27]:


#model.to(args['device'])
#model.eval()


# In[28]:


history = []
while True:
    raw_text = input(">>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input(">>> ")
    if raw_text == 'q':
        break
        
    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args['max_history']+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    print(out_text)


# In[ ]:




