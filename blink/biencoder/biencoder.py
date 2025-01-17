# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        """

        :param params: 约38个参数，{'silent': True, 'debug': False, 'data_parallel': False, 'no_cuda': False, 'top_k': 10, 'seed': 52313, 'zeshel': True, 'max_seq_length': 256, 'max_context_length': 128, 'max_cand_length': 128, 'path_to_model': None, 'bert_model': 'bert-base-uncased', 'pull_from_layer': -1, 'lowercase': True, 'context_key': 'context', 'out_dim': 1, 'add_linear': False, 'data_path': 'data/zeshel/blink_format', 'output_path': 'models/zeshel/biencoder', 'evaluate': False, 'output_eval_file': None, 'train_batch_size': 8, 'max_grad_norm': 1.0, 'learning_rate': 1e-05, 'num_train_epochs': 5, 'print_interval': 10, 'eval_interval': 100, 'save_interval': 1, 'warmup_proportion': 0.1, 'gradient_accumulation_steps': 1, 'type_optimization': 'all_encoder_layers', 'shuffle': False, 'eval_batch_size': 8, 'mode': 'valid', 'save_topk_result': False, 'encode_batch_size': 8, 'cand_pool_path': None, 'cand_encode_path': None}
        :type params:
        """
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])   #提及上下文的模型
        cand_bert = BertModel.from_pretrained(params['bert_model'])   # 知识图谱中实体上下文的模型的模型
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],   #eg: 100
            layer_pulled=params["pull_from_layer"],  #eg: -1
            add_linear=params["add_linear"],  #eg: False
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],  #eg: 100
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,  #[batch_size, max_context_length],
        segment_idx_ctxt,  #[batch_size, max_context_length]
        mask_ctxt,  #[batch_size, max_context_length]
        token_idx_cands,  # eg: None
        segment_idx_cands,  # eg: None
        mask_cands,  # eg: None
    ):  #调用各自的模型进行encoder
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        """

        :param params:
        :type params:
        :param shared:
        :type shared:
        """
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params.get("no_cuda", False) else "cpu"
        )   #cuda
        self.n_gpu = torch.cuda.device_count()  # eg: 1
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )  #'bert_model': 'bert-large-uncased'
        # 初始化双编码器的2个模型，模型1：提及的上下文处理，模型2：知识图谱中实体的上下处理
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:  # 'models/biencoder_wiki_large.bin'
            self.load_model(model_path)  # 加载模型
        #模型放到设备上
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")   # 是否并行, eg: False
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        assert os.path.exists(fname), f"模型文件不存在，请检查{fname}"
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,  # 文本向量， [batch_size, max_context_length], eg: [8,32]
        cand_vecs,   # eg: None
        random_negs=True,  # 随机选择负样本
        cand_encs=None,  # 预先计算的实体嵌入， torch.Size([5903527, 1024]), [实体数量，实体嵌入维度].
    ):
        # 首先编码上下文, 生成bert的输入格式， token_idx, segment_idx, mask
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )  # 经过bert编码后的上下文的向量， [batch_size, embed_size], eg: [8,1024], 或者[8,768]代表base版本

        # 候选实体的encoding如果给定了，那么就不要重新计算了，直接返回上下文encoding和候选encoding之间的矩阵相乘分数
        # 推理的时候使用
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # 训练的时候走这里， 获取候选实体的encoding
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        ) # embedding_cands：[batch_size,embed_size]
        if random_negs:
            # 使用随机负样本进行训练
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # 使用困难负样本进行训练
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    def forward(self, context_input, cand_input, label_input=None):
        """
            # label_input -- 提供的负样本
            # 如果不提供label_input,那么仅使用批次内的数据进行训练
        :param context_input:
        :type context_input:
        :param cand_input:
        :type cand_input:
        :param label_input:
        :type label_input:
        :return:
        :rtype:
        """
        flag = label_input is None  # 判断标签是否为空
        scores = self.score_candidate(context_input, cand_input, flag)  # 结果为[batch_size,batch_size], 表示提及的上下文和候选实体之间的相似度
        bs = scores.size(0)  #获取batch_size  eg: 8
        if label_input is None:
            # 不提供负样本的话，对角线上的位置即为正样本，所以生成一个
            target = torch.LongTensor(torch.arange(bs))  # 创建一个[batch_size]的tensor， 如果label不存在的话, eg: tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')
            target = target.to(self.device)
            # scores: [batch_size,batch_size], target: [batch_size]
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """ token_idx 是一个 2D tensor int.
        return bert模型需要的 token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
