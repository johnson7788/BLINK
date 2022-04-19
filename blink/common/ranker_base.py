# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled   #eg: -1， 默认使用bert最后一层的输出，这里没有使用
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)   #eg: 1024 或者768，large是1024
        # 加载的bert模型
        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        #获取 [CLS] token对应的向量， 或者最后一层的所有向量
        if self.additional_linear is not None:
            embeddings = output_pooler   #[batch_size, embedding_dim], eg: [10,1024]
        else:
            embeddings = output_bert[:, 0, :]

        # #如果有额外线性层，使用额外线性层
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

