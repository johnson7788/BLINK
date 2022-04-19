# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import sys

import numpy as np
from tqdm import tqdm
import blink.biencoder.data_process as data
from blink.common.params import ENT_START_TAG, ENT_END_TAG



def prepare_crossencoder_mentions(
    tokenizer,
    samples,
    max_context_length=32,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):

    context_input_list = []  # samples X 128

    for sample in tqdm(samples, desc="遍历样本"):
        context_tokens = data.get_context_representation(
            sample,  # 一条样本
            tokenizer,  # 初始化后的tokenizer
            max_context_length,  #最大上下长度
            mention_key,   #提及对应的样本中字典的key， eg：'mention'
            context_key,   #提及对应的上下文中字典的key， eg：'context'
            ent_start_token,   #实体的起始位置的特殊token， '[unused1]'
            ent_end_token,    #实体结束位置的特殊token， '[unused0]'
        )
        tokens_ids = context_tokens["ids"]   #提及对应的上下文表示的id
        context_input_list.append(tokens_ids)   #收集起来
    # 转换成numpy， context_input_list: [样本数, max_context_length]  eg: [14,32], [样本数，提及的上下文句子的最大长度]
    context_input_list = np.asarray(context_input_list)
    return context_input_list


def prepare_crossencoder_candidates(
    tokenizer, labels, nns, id2title, id2text, max_cand_length=128, topk=100
):
    # 准备数据
    START_TOKEN = tokenizer.cls_token   #'[CLS]'
    END_TOKEN = tokenizer.sep_token    #'[SEP]'

    candidate_input_list = []  # samples X topk=10 X 128
    label_input_list = []  # samples
    idx = 0
    for label, nn in zip(labels, nns):
        candidates = []
        #真实的标签对应着候选的预测结果的第几个
        label_id = -1
        for jdx, candidate_id in enumerate(nn[:topk]):
            # 如果真实标签的id和候选的标签id相同, label: [461053], candidate_id: 461053
            if label == candidate_id:
                label_id = jdx
            # 候选实体的上下表示， title + unused2 + 描述文本
            rep = data.get_candidate_representation(
                id2text[candidate_id],
                tokenizer,
                max_cand_length,  #最大候选的长度
                id2title[candidate_id],
            )
            tokens_ids = rep["ids"]  # 候选实体的input_ids表示

            assert len(tokens_ids) == max_cand_length, f"再次判断，发现候选实体的上下表示的最大长度超过了我们预设的长度"
            candidates.append(tokens_ids)

        label_input_list.append(label_id)
        candidate_input_list.append(candidates)

        idx += 1
        sys.stdout.write("处理双编码器的输入，处理第{}条数据，共有{}条数据\r".format(idx, len(labels)))
        sys.stdout.flush()

    label_input_list = np.asarray(label_input_list)
    candidate_input_list = np.asarray(candidate_input_list)
    # label_input_list： 真实的标签对应着候选的预测结果的第几个， candidate_input_list： 候选实体对应的输入, [topk, max_cand_length]
    return label_input_list, candidate_input_list


def filter_crossencoder_tensor_input(
    context_input_list, label_input_list, candidate_input_list
):
    # 移除label_input_list列表中-1对应的位置的数据， -1表示，候选实体列表中不包含ground_truth真实的答案实体
    context_input_list_filtered = [
        x
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    label_input_list_filtered = [
        z
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    candidate_input_list_filtered = [
        y
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    return (
        context_input_list_filtered,
        label_input_list_filtered,
        candidate_input_list_filtered,
    )


def prepare_crossencoder_data(
    tokenizer, samples, labels, nns, id2title, id2text, keep_all=False
):
    # 准备交叉编码器的数据， tokenizer： 初始化后的tokenizer， samples：原始样本， labels： ground truth 标签的id，nns：双编码器预测的候选实体id列表，id2title: id到标题的映射，id2text: id到text文本内容的映射
    # 提及的句子变成input_ids的格式： context_input_list: [样本数, max_context_length]  eg: [14,32], [样本数，提及的上下文句子的最大长度]
    context_input_list = prepare_crossencoder_mentions(tokenizer, samples)

    # 编码候选实体，即双向编码器的输出, label_input_list： 真实的标签对应着候选的预测结果的第几个， candidate_input_list： 候选实体对应的输入, [topk, max_cand_length]
    label_input_list, candidate_input_list = prepare_crossencoder_candidates(
        tokenizer, labels, nns, id2title, id2text
    )

    if not keep_all:
        # 移除样本，其中ground_truth实体不在候选实体列表中的数据
        (
            context_input_list,
            label_input_list,
            candidate_input_list,
        ) = filter_crossencoder_tensor_input(
            context_input_list, label_input_list, candidate_input_list
        )
    else:
        label_input_list = [0] * len(label_input_list)
    # 转换成tensor格式
    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    candidate_input = torch.LongTensor(candidate_input_list)

    return (
        context_input,   # torch.Size([14, 32]), 14代表样本数量，32代表提及的上下文的长度
        candidate_input,  #torch.Size([14, 10, 128])， 14代表样本数量，10代表topk个候选实体， 128代表kg中实体的上下的长度
        label_input,      #14,  真实的标签对应着候选的预测结果的第几个
    )
