# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys
import os
import pickle
from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None, use_cache=True, minisize=-1):
    """
    只有当不使用faiss索引文件的时候，才加载所有候选实体的encoding
    # use_cache: 使用加载缓存的方式，更快加载
    :minisize, 测试多少条实体，太多实体被killed, -1表示不限制
    """
    # cache_file = "models/models_candidates.cache"
    # if use_cache and os.path.exists(cache_file):
    #     print(f"注意： 使用的{cache_file}缓存加载的实体预处理数据，如果不想使用，可以删掉这个文件")
    #     tuple_data = pickle.load(open(cache_file, "rb"))
    #     return tuple_data
    if faiss_index is None:
        logger.info(f"加载实体encoding, 占用大量内存")
        candidate_encoding = torch.load(entity_encoding)  # 加载entity_encoding： 'models/all_entities_large.t7'， candidate_encoding: torch.Size([5903527, 1024])
        indexer = None
    else:
        if logger:
            logger.info("使用 faiss 索引检索实体")
        candidate_encoding = None
        assert index_path is not None, "错误! 传入的索引文件是空的,请检查!"
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("错误! 不支持的索引 indexer 类型! 仅支持flat或者hnsw。")
        indexer.deserialize_from(index_path)
    logger.info("开始加载实体数据")
    # 一共5903527个实体
    title2id = {}   # 标题到id的映射字典, 5903527
    id2title = {}   #id到标题的映射,5903527
    id2text = {}   # id到文本的映射,5903527
    wikipedia_id2local_id = {}   #wiki原始id到本地定义id的映射,5903527
    local_idx = 0   #初始化本地id的一个索引
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()  # list, 5903527条，所有的实体数据，包含{"text": " xxxxxx ", "idx": "https://en.wikipedia.org/wiki?curid=12", "title": "Anarchism", "entity": "Anarchism"}
        if minisize != -1:
            lines = lines[:minisize]
        for line in tqdm(lines, desc='加载实体数据'):
            entity = json.loads(line)   # 加载一条数据
            # wikipedia_id的id和自己定义的id映射
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())  #eg: 12
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id, f"有重复的Wikipedia 的id"
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    tuple_data = (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )
    # pickle.dump(tuple_data, open(cache_file, "wb"))
    return tuple_data


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        continue
                except:
                    continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("文件行数{}, 获取到{}条测试样本".format(len(lines), len(test_samples)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    """

    """
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("开始加载 biencoder 模型")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)
    # 交叉编码器模型，可以为None或者存在
    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("不使用fast模式，加载crossencoder模型")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("开始加载候选实体集，实体数量较多，加载可能会比较慢")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue,    #'models/entity.jsonl'
        args.entity_encoding,    #'models/all_entities_large.t7'
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,   #初始化后的双编码器模型
        biencoder_params,  #加载的双编码器模型的一些定义的参数
        crossencoder,  #初始化后的交叉模型
        crossencoder_params, #交叉模型对应的参数
        candidate_encoding,  #实体对应的向量torch.Size([5903527, 1024])
        title2id,    #标题对应的id的字典，5903527
        id2title,    #id对应标题的字典，5903527
        id2text,     # id对应的原始文件，5903527
        wikipedia_id2local_id,   # wiki的id映射到我们定义的id的字典，5903527
        faiss_indexer,  #faiss索引
    )


def run(
    args,
    logger,  #日志
    biencoder,  #初始化后的双编码器模型
    biencoder_params,#加载的双编码器模型的一些定义的参数
    crossencoder,  #初始化后的交叉模型
    crossencoder_params,  #交叉模型对应的参数
    candidate_encoding,   #实体对应的向量torch.Size([5903527, 1024])
    title2id,   #标题对应的id的字典，5903527
    id2title,   #id对应标题的字典，5903527
    id2text,    # id对应的原始文件，5903527
    wikipedia_id2local_id,   # wiki的id映射到我们定义的id的字典，5903527Dictionary of the # title corresponding to the id, 5903527
    faiss_indexer=None,  #faiss索引
    test_data=None,   #测试数据
):
    #
    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "错误: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entitied (--test_entities)"
        )
        raise ValueError(msg)
    # wiki的实体id对应的url路径
    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }
    # 停止条件
    stopping_condition = False
    while not stopping_condition:

        samples = None
        # 交互模式
        if args.interactive:
            logger.info("进入交互模式")

            # biencoder_params["eval_batch_size"] = 1

            # 加载 NER 模型
            ner_model = NER.get_model()

            # Interactive
            text = input("请输入文本:")

            # Identify mentions
            samples = _annotate(ner_model, [text])

            _print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("进入测试集测试模式")

            if test_data:
                samples = test_data
            else:
                # Load test mentions
                samples = _get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                )

            stopping_condition = True

        # don't look at labels
        keep_all = (
            args.interactive
            or samples[0]["label"] == "unknown"
            or samples[0]["label_id"] < 0
        )

        # prepare the data for biencoder
        if logger:
            logger.info("准备数据用于biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info("开始运行 biencoder")
        top_k = args.top_k
        labels, nns, scores = _run_biencoder(
            biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
        )

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        else:

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values
                top_k = args.top_k
                x = []
                y = []
                for i in range(1, top_k):
                    temp_y = 0.0
                    for label, top in zip(labels, nns):
                        if label in top[:i]:
                            temp_y += 1
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                # plt.plot(x, y)
                biencoder_accuracy = y[0]
                recall_at = y[-1]
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

            if args.fast:

                predictions = []
                for entity_list in nns:
                    sample_prediction = []
                    for e_id in entity_list:
                        e_title = id2title[e_id]
                        sample_prediction.append(e_title)
                    predictions.append(sample_prediction)

                # use only biencoder
                return (
                    biencoder_accuracy,
                    recall_at,
                    -1,
                    -1,
                    len(samples),
                    predictions,
                    scores,
                )

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all,
        )

        context_input = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        dataloader = _process_crossencoder_dataloader(
            context_input, label_input, crossencoder_params
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = _run_crossencoder(
            crossencoder,
            dataloader,
            logger,
            context_len=biencoder_params["max_context_length"],
        )

        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            _print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, index_list, sample in zip(nns, index_array, samples):
                e_id = entity_list[index_list[-1]]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()
        else:

            scores = []
            predictions = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    e_title = id2title[e_id]
                    sample_prediction.append(e_title)
                    sample_scores.append(scores_list[index])
                predictions.append(sample_prediction)
                scores.append(sample_scores)

            crossencoder_normalized_accuracy = -1
            overall_unormalized_accuracy = -1
            if not keep_all:
                crossencoder_normalized_accuracy = accuracy
                print(
                    "crossencoder normalized accuracy: %.4f"
                    % crossencoder_normalized_accuracy
                )

                if len(samples) > 0:
                    overall_unormalized_accuracy = (
                        crossencoder_normalized_accuracy * len(label_input) / len(samples)
                    )
                print(
                    "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                )
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                len(samples),
                predictions,
                scores,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="交互模式，或者传入--test_mentions 和--test_entities，自动测试"
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="测试提及"
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="测试实体集"
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="双编码器模型路径",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="双编码器模型配置",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="实体目录的路径",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="实体嵌入encoding的路径，维度，5903527, 1024",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="交叉编码模型的路径",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="交叉编码模型的配置",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="双编码器检索到的候选者数量",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="日志输出路径",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="只使用双编码器模式"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="交互模式时是否展示实体url",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="是否使用faiss索引",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="如果使用faiss索引，那么索引文件的路径, pkl文件",
    )

    args = parser.parse_args()
    # 日志文件, eg: output/log.txt
    logger = utils.get_logger(args.output_path)
    # 加载模型， models：tuple，里面包含10个
    models = load_models(args, logger)
    #
    run(args, logger, *models)
