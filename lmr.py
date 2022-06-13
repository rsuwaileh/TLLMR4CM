# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random
import copy

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import convert_examples_to_features, read_examples_from_file, get_predictions, write_predictions


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def get_labels(lmr_mode):
        # The BILOU labels
    if lmr_mode == "TB": # type-less
        labels = ["B-CONT", "B-CTRY", "B-STAT", "B-CNTY", "B-CITY", "B-DIST", "B-NBHD", "B-ISL", "B-NPOI", "B-HPOI", "B-ST", "B-OTHR", 
                "I-CONT", "I-CTRY", "I-STAT", "I-CNTY", "I-CITY", "I-DIST", "I-NBHD", "I-ISL", "I-NPOI", "I-HPOI", "I-ST", "I-OTHR", 
                "L-CONT", "L-CTRY", "L-STAT", "L-CNTY", "L-CITY", "L-DIST", "L-NBHD", "L-ISL", "L-NPOI", "L-HPOI", "L-ST", "L-OTHR", 
                "U-CONT", "U-CTRY", "U-STAT", "U-CNTY", "U-CITY", "U-DIST", "U-NBHD", "U-ISL", "U-NPOI", "U-HPOI", "U-ST", "U-OTHR", 
                "O"]
    else: #"TL": type-less
        labels = ["B-LOC", "I-LOC", "L-LOC", "U-LOC", "O"]
    return labels

def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])

def evaluate(args, model, tokenizer, labels, pad_token_label_id, prefix=""):
    eval_dataset = load_examples(args, tokenizer, labels, pad_token_label_id)

    args["eval_batch_size"] = args["per_gpu_eval_batch_size"] * max(1, args["n_gpu"])
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args["local_rank"] == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    # multi-gpu evaluate
    if args["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    #print("***** Running evaluation {} *****".format(prefix))
    #print("  Num examples = {}".format(str(len(eval_dataset))))
    #print("  Batch size = {}".format(args["eval_batch_size"]))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args["model_type"] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args["model_type"] in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args["n_gpu"] > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    #print("Eval results {} *****".format(prefix))
    #for key in sorted(results.keys()):
    #    print("  {} = {}".format(key, str(results[key])))

    return results, preds_list


def load_examples(args, tokenizer, labels, pad_token_label_id):
    if args["local_rank"] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    print("Creating features from dataset file at %s", args["gold_path"])
    examples = read_examples_from_file(args["gold_path"], "test")
    features = convert_examples_to_features(
        examples,
        labels,
        args["max_seq_length"],
        tokenizer,
        cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args["model_type"] in ["roberta"]),

        pad_on_left=bool(args["model_type"] in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
    )

    if args["local_rank"] == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def get_locations(gold_path, lmr_mode, model, device):
    #some of the parametres need to be removed
    args = {
        "model_type" : "bert",
        "tokenizer_name": "bert-large-cased",
        "model_name_or_path": "bert-large-cased", 
        "per_gpu_eval_batch_size": 8,
        "max_seq_length": 128, 
        "eval_batch_size": 8,
        "seed": 42,
        "overwrite_cache": True,
        "n_gpu": 0, 
        "no_cuda": True,
        "local_rank": -1
    }
    args["device"] = device
    #args["n_gpu"] = 0 if args["no_cuda"] else torch.cuda.device_count()
    args["gold_path"] = gold_path #text_file
    args["pred_path"] = gold_path.replace(".txt", "_predictions.txt")


    #from TLLMR4CM import set_seed
    set_seed(args)
    labels = get_labels(lmr_mode)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    
    tokenizer = BertTokenizer.from_pretrained(args["tokenizer_name"])
    result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id)
    
    print(result)
    print(predictions)
    
    #tow = [x for x in predictions]
    tow = copy.deepcopy(predictions)
    write_predictions(args["gold_path"], args["pred_path"], tow)
    
    #gk, gl, gt = get_predictions(args["gold_path"], [])
    #print("**************************")
    #print(gk)
    #print(gl)
    #print(gt)

    pk, pl, pt = get_predictions(args["pred_path"], predictions)
    print("**************************")
    print(pk)
    print(pl)
    print(pt)
 
    
    #g = []
    p = []
    for i in range(len(gl)):
        #g.append(["{}:{}\t".format(x, y) for x, y in zip(gl[i], gt[i])])
        p.append(["{}:{}\t".format(x, y) for x, y in zip(pl[i], pt[i])])
    
    return gk, pk, g, p
