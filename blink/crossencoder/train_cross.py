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

import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None


def modify(context_input, candidate_input, max_seq_length):
    """
    合并context和candidate，并padding, 并返回合并后的tensor，限制最大长度max_context_length
    :param context_input:
    :type context_input:
    :param candidate_input:
    :type candidate_input:
    :param max_seq_length:
    :type max_seq_length:
    :return:
    :rtype:
    """
    new_input = []
    context_input = context_input.tolist()   ## torch.Size([14, 32]), 14代表样本数量，32代表提及的上下文的长度
    candidate_input = candidate_input.tolist()  ##torch.Size([14, 10, 128])， 14代表样本数量，10代表topk个候选实体， 128代表kg中实体的上下的长度

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # 从候选实体中移除 [CLS] token， 组成新的样本
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]  # max_seq_length： 160
            mod_input.append(sample)

        new_input.append(mod_input)
    # 维度: [14,10,159], 14代表样本数量，10代表topk个候选实体， 159代表拼接后双编码和交叉编码器的input_ids后的样本的长度
    return torch.LongTensor(new_input)


def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    """

    :param reranker: 初始化后的模型
    :type reranker:
    :param eval_dataloader:
    :type eval_dataloader:
    :param device: cuda
    :type device:
    :param logger:
    :type logger:
    :param context_length: 128
    :type context_length:
    :param zeshel: 是不是zeshel数据集
    :type zeshel:
    :param silent: 是否是silent，显示进度条与否
    :type silent:
    :return:
    :rtype:
    """
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="评估")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)  # 每条样本都放到device上
        context_input = batch[0]  # 双编码器和交叉编码器的input_ids 拼接后的上下文的input_id,  [batch_size, topk_候选个数,max_seq_length]
        label_input = batch[1]   # 预测的标签， [batch_size]， 应该预测tokp的第几个标签是正确的
        with torch.no_grad():
            # eval_loss： 损失， logits：[batch_size, topk] 预测logits
            eval_loss, logits = reranker(context_input, label_input, context_length)
        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        # 对比预测结果和真实值，计算准确率, label_ids: [1], logits: [batch_size, topk] 预测logits
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)
        # 一共训练了多少个样本，都累加起来
        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()  #eg: 9, 主题的索引
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if zeshel:  #zeroshot模式
        macro = 0.0
        num = 0.0 
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro准确率: %.5f" % (macro / num))
            logger.info("Micro准确率: %.5f" % normalized_eval_accuracy)
    else:
        if logger:
            logger.info("评估的准确率: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
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
    logger.info(" 总共的训练优化step数量 = %d" % num_train_steps)
    logger.info(" Warmup的step数量 = %d", num_warmup_steps)
    return scheduler


def main(params):
    """
    主函数， 训练交叉编码器
    :param params:
    :type params:
    :return:
    :rtype:
    """
    model_output_path = params["output_path"]
    print(f"模型输出路径：{model_output_path}")
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])
    #初始化模型
    reranker = CrossEncoderRanker(params)
    print(f"模型结构是:")
    print(reranker)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # 计算训练批次大小， An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    # 读取数据, 加载topk训练集
    fname = os.path.join(params["data_path"], "train.t7")
    print(f"加载训练数据集：{fname}")
    train_data = torch.load(fname)  # 训练集dict, 包含context_vecs:提及上下文向量[45155，128]， [样本数,max_context_length]  candidate_vecs: [45155,topk候选, max_cand_length], label: 45155, worlds:45155
    context_input = train_data["context_vecs"]  #提及上下文向量
    candidate_input = train_data["candidate_vecs"]  #候选词向量
    label_input = train_data["labels"]  #标签
    if params["debug"]:  # debug模式，加载200条数据
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    print(f"提及上下文的维度是：{context_input.shape}")
    context_input = modify(context_input, candidate_input, max_seq_length) #合并提及和实体的上下向量
    print(f"合并提及和实体的上下向量后的维度是：{context_input.shape}")
    if params["zeshel"]:
        #
        src_input = train_data['worlds'][:len(context_input)]
        train_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        train_tensor_data = TensorDataset(context_input, label_input)
    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )
    # 验证集加载
    fname = os.path.join(params["data_path"], "valid.t7")
    valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"]
    candidate_input = valid_data["candidate_vecs"]
    label_input = valid_data["labels"]
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    # 合并提及和实体的上下向量
    context_input = modify(context_input, candidate_input, max_seq_length)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    print(f"训练之前先评估一次")
    # results = evaluate(reranker,valid_dataloader, device=device,logger=logger,context_length=context_length,zeshel=params["zeshel"],silent=params["silent"])

    number_of_samples_per_dataset = {}

    time_start = time.time()
    print(f'保存训练参数到文件: {os.path.join(model_output_path, "training_params.txt")}')
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("开始训练交叉编码器")
    logger.info(
        "设备: {} n_gpu: {}, 分布式训练: {}".format(device, n_gpu, False)
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

        part = 0
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0] # 提及上下文的向量, shape: (batch_size, topk候选, max_seq_length)
            label_input = batch[1]   #[batch_size]
            loss, _ = reranker(context_input, label_input, context_length)

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
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker,
                    valid_dataloader,
                    device=device,
                    logger=logger,
                    context_length=context_length,
                    zeshel=params["zeshel"],
                    silent=params["silent"],
                )
                logger.info("***** Saving fine - tuned model *****")
                epoch_output_folder_path = os.path.join(
                    model_output_path, "epoch_{}_{}".format(epoch_idx, part)
                )
                part += 1
                utils.save_model(model, tokenizer, epoch_output_folder_path)
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        # reranker.save(epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print('参数如下:\n',args)

    params = args.__dict__
    main(params)
