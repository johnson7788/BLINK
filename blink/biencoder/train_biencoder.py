# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    """
    # 训练期间的评估函数使用批次内的负样本。
    # 对于大小为B的批次，该批次的标签被用作标签候选者，B由参数eval_batch_size控制
    :param reranker:
    :type reranker:
    :param eval_dataloader:
    :type eval_dataloader:
    :param params:
    :type params:
    :param device:
    :type device:
    :param logger:
    :type logger:
    :return:
    :rtype:
    """
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="评估")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, _, _ = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # 当使用批次内的数据作为负样本的时候，标签的id正好是对角线的值，所以不用外面传入label了
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("评估的准确率是: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info("优化的steps数量 = %d" % num_train_steps)
    logger.info("Warmup的steps数量 = %d", num_warmup_steps)
    return scheduler


def main(params):
    """
    主函数
    :param params:  所有参数
    :type params:
    :return:
    :rtype:
    """
    model_output_path = params["output_path"]  #模型保存路径
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    #初始化模型
    reranker = BiEncoderRanker(params)
    print(f"模型结构是:")
    print(reranker)
    tokenizer = reranker.tokenizer
    model = reranker.model   #双编码器模型

    device = reranker.device
    n_gpu = reranker.n_gpu
    # 梯度累加不能小于1
    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]  #eg: 1

    # 固定随机种子
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    #加载训练数据
    logger.info(f"开始加载训练数据: {params['data_path']}")
    train_samples = utils.read_dataset("train", params["data_path"], debug=params["debug"])
    logger.info(f"共获取到训练样本条数：{len(train_samples)}")
    # 处理数据, train_data:dict,{context_vecs: 提及上下文的向量[样本数，max_context_length]， cand_vecs: 知识图谱中候选实体上下文向量[样本数，max_cand_length], labed_idx: ground_truth实体的id [样本数，1], src: [样本数，1], 表示这条数据的主题是什么
    train_data, train_tensor_data = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)
    # 构造数据集
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    #加载验证集数据
    # TODO: reduce duplicated code here
    valid_samples = utils.read_dataset("valid", params["data_path"], debug=params["debug"])
    logger.info(f"共获取到验证样本条数：{len(valid_samples)}", )
    # 和训练数据一样进行的预处理
    valid_data, valid_tensor_data = data.process_mention_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # 在训练前评估一次
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger,
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()
    params_file = os.path.join(model_output_path, "training_params.txt")
    logger.info(f"训练参数写入到文件中: {params_file}")
    utils.write_to_file(
        params_file, str(params)
    )

    logger.info("开始训练")
    logger.info(
        "device: {} n_gpu: {}, 是否分布式训练: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, candidate_input, _, _ = batch   #获取一个批次数据中的，context_input: 提及的上下文, candidate_input：实体的上下文, labled_idx：标签对应的id, src： 样本的主题
            # _是score的返回值，结果为[batch_size,batch_size], 表示提及的上下文和候选实体之间的相似度
            loss, _ = reranker(context_input, candidate_input)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("在开发集上进行评估")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                model.train()
                logger.info("\n")

        logger.info("*****保存一个epoch的微调模型*****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        # 保存模型
        utils.save_model(model, tokenizer, epoch_output_folder_path, logger)
        logger.info("开始评估这个epoch的微调模型")
        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "训练花费了 {} 分钟\n".format(execution_time),
    )
    logger.info("训练花费了 {} 分钟\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("在第 {} 个epoch时，模型效果最好".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path, logger)
    # 进行最后的评估
    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    # 添加模型的训练参数
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print('参数如下:\n',args)

    params = args.__dict__
    main(params)
