# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Utility code for zeshel dataset
import json
import os

import torch

DOC_PATH = "/private/home/ledell/zeshel/data/documents/"

WORLDS = [
    'american_football',
    'doctor_who',
    'fallout',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
]

world_to_id = {src : k for k, src in enumerate(WORLDS)}


def load_entity_dict_zeshel(logger, params):
    entity_dict = {}
    # different worlds in train/valid/test
    if params["mode"] == "train":
        start_idx = 0
        end_idx = 8
    elif params["mode"] == "valid":
        start_idx = 8
        end_idx = 12
    else:
        start_idx = 12
        end_idx = 16
    # 数据的document的路径和data_path在一起
    data_path = params["data_path"]
    document_path = os.path.join(os.path.dirname(data_path), "documents")
    for i, src in enumerate(WORLDS[start_idx:end_idx]):
        fname = src + ".json"
        full_path = os.path.join(document_path, fname)
        assert os.path.exists(full_path), f"数据集的document文件未找到: {full_path}"
        cur_dict = {}
        doc_list = []
        src_id = world_to_id[src]
        with open(full_path, 'rt') as f:
            for line in f:
                line = line.rstrip()
                item = json.loads(line)
                text = item["text"]
                doc_list.append(text[:256])

                if params["debug"]:
                    if len(doc_list) > 200:
                        break

        logger.info("加载world文件 %s 完成" % src)
        entity_dict[src_id] = doc_list
    return entity_dict


class Stats():
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        self.LEN = len(self.rank) 
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        return output_json

