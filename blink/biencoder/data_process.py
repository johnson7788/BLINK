# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,  #'[unused0]'
    ent_end_token=ENT_END_TAG,  #'[unused1]'
):
    """
    #获取上下文表示和对应的id，这即提及的整个句子的表示： ['[CLS]', 'cricket', '-', '[unused0]', 'leicestershire', '[unused1]', 'take', 'over', 'at', 'top', 'after', 'innings', 'victory', '.', 'london', '1996', '-', '08', '-', '30', 'west', 'indian', 'all', '-', 'round', '##er', 'phil', 'simmons', 'took', 'four', 'for', '[SEP]']
    :param sample:
    :type sample:
    :param tokenizer:
    :type tokenizer:
    :param max_seq_length:
    :type max_seq_length:
    :param mention_key:
    :type mention_key:
    :param context_key:
    :type context_key:
    :param ent_start_token:
    :type ent_start_token:
    :param ent_end_token:
    :type ent_end_token:
    :return:
    :rtype:
    """
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:   #提及对应的内容， eg: sample[mention_key]: 'leicestershire'
        mention_tokens = tokenizer.tokenize(sample[mention_key])   #['leicestershire']
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]  #eg: ['[unused0]', 'leicestershire', '[unused1]']
    # context_left: 获取提及的左侧上下文,
    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)   #eg: ['cricket', '-']
    context_right = tokenizer.tokenize(context_right)
    # max_seq_length：32， 最大序列长度， left_quota：13， right_quota：14
    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add
    #context_tokens: 截断和tokenize之后的token： ['cricket', '-', '[unused0]', 'leicestershire', '[unused1]', 'take', 'over', 'at', 'top', 'after', 'innings', 'victory', '.', 'london', '1996', '-', '08', '-', '30', 'west', 'indian', 'all', '-', 'round', '##er', 'phil', 'simmons', 'took', 'four', 'for']
    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    # 加上CLS和SEP
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)  # 转换成id， [101, 4533, 1011, 1, 20034, 2, 2202, 2058, 2012, 2327, 2044, 7202, 3377, 1012, 2414, 2727, 1011, 5511, 1011, 2382, 2225, 2796, 2035, 1011, 2461, 2121, 6316, 13672, 2165, 2176, 2005, 102]
    padding = [0] * (max_seq_length - len(input_ids))   # padding的长度
    input_ids += padding
    assert len(input_ids) == max_seq_length, f"截断后和padding后，token转换成id和的长度和最大序列长度不等"

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc,    #'1622318'
    tokenizer,   # <pytorch_transformers.tokenization_bert.BertTokenizer object at 0x7f4f95c32c40>
    max_seq_length,   #候选实体的上下文的长度eg: 128
    candidate_title=None,  #eg: None
    title_tag=ENT_TITLE_TAG,  #eg:'[unused2]'
):
    cls_token = tokenizer.cls_token  #eg: '[CLS]'
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)  # 候选实体的描述文本进行tokenize， ['leicestershire', 'county', 'cricket', 'club', 'is', 'one', 'of', 'eighteen', 'first', '-', 'class', 'county', 'clubs', 'within', 'the', 'domestic', 'cricket', 'structure', 'of', 'england', 'and', 'wales', '.', 'it', 'represents', 'the', 'historic', 'county', 'of', 'leicestershire', '.', 'it', 'has', 'also', 'been', 'representative', 'of', 'the', 'county', 'of', 'rutland', '.', 'the', 'club', "'", 's', 'limited', 'overs', 'team', 'is', 'called', 'the', 'leicestershire', 'foxes', '.', 'founded', 'in', '1879', ',', 'the', 'club', 'had', 'minor', 'county', 'status', 'until', '1894', 'when', 'it', 'was', 'promoted', 'to', 'first', '-', 'class', 'status', 'pending', 'its', 'entry', 'into', 'the', 'county', 'championship', 'in', '1895', '.', 'since', 'then', ',', 'leicestershire', 'have', 'played', 'in', 'every', 'top', '-', 'level', 'domestic', 'cricket', 'competition'...
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)  #对应的标题也进行['leicestershire', 'county', 'cricket', 'club']
        cand_tokens = title_tokens + [title_tag] + cand_tokens
    # eg: ['[CLS]', '1622', '##31', '##8', '[SEP]'], 最大长度-2，为了加上CLS和SEP
    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]
    #  [101, 28133, 21486, 2620, 102]
    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length, f"实体的上下的长度和要求的最大序列长度不相等，请检查"

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,  # 是否显示进度条
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    """
    处理提及数据
    :param samples: 测试集的每条样本，每条样本包含{'context_left': 'cricket -', 'mention': 'leicestershire', 'context_right': "take over at top after innings victor", 'query_id': '947testa CRICKET:0', 'label_id': 461053, 'Wikipedia_ID': '1622318', 'Wikipedia_URL': 'http://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club', 'Wikipedia_title': 'Leicestershire County Cricket Club', 'label': '1622318'}
    :type samples: list
    :param tokenizer: 初始化后的tokenizer，例如BertTokenizer
    :type tokenizer:
    :param max_context_length: 最大上下文长度 eg: 32
    :type max_context_length: int
    :param max_cand_length:  eg: 128
    :type max_cand_length: int
    :param silent: 显示进度
    :type silent: bool
    :param mention_key:  'mention'， 提及的key，获取每条样本的提及时使用
    :type mention_key:
    :param context_key:  'context'
    :type context_key:
    :param label_key:  samples中label的字典的key， 'label'， 方便获取每条样本的label内容
    :type label_key:
    :param title_key:  'label_title'
    :type title_key:
    :param ent_start_token:  '[unused0]'  包裹提及实体的特殊token
    :type ent_start_token:
    :param ent_end_token:  '[unused1]'
    :type ent_end_token:
    :param title_token:  '[unused2]'
    :type title_token:
    :param debug: 打印日志, 是否是debug模式, debug模式加载数据集200条
    :type debug: bool
    :param logger:  eg:None
    :type logger:None或初始化后的logger
    :return:
    :rtype:
    """
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples, desc="提及数据编码处理")

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )
        # context_tokens: dict，包含tokens和ids, 即token变成id后的结果
        label = sample[label_key]  #获取label的id， eg: '1622318'
        title = sample.get(title_key, None)  #获取样本的标题，可能不存在
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )  #label转换成 tokens 和ids的格式，
        label_idx = int(sample["label_id"])  # label的id, eg: 461053
        # 一条数据
        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }
        # src，表示这条数据的主题是什么
        if "world" in sample:
            src = sample["world"]  #'muppets'
            src = world_to_id[src] #eg: 9
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====打印前5条处理完成的样本: ====")
        for sample in processed_samples[:5]:
            logger.info("上下文tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            if sample.get("src"):
                logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])
    # context_vecs, [batch_size, max_seq_length], 上下文向量, [14,32], 最大候选长度
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    ) # cand_vecs: [batch_size, label_max_seq_length] , [14,128], padding那么多干啥，标签的长度设置太长了,
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    ) # [batch_size, 1] 标签对应的id
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data
