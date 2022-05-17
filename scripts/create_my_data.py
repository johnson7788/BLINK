#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/5/17 10:03 上午
# @File  : create_my_data.py.py
# @Author:
# @Desc  : 创建自定义的数据集，从neo4j获取，并保存为jsonl格式

import json
import os
from py2neo import Graph
import random

def get_data():
    # 连接neo4j
    graph = Graph(host='l0', user='neo4j', password='welcome')
    # 匹配所有是凡士林的产品
    sql = """MATCH (n:Brand {name:"凡士林"})<- [:PRODUCT_BRAND_IS] - (p) RETURN p """
    products_res = graph.run(sql)
    products = products_res.data()
    data = []
    for product in products:
        product_name = product['p']['name']
        product_id = product['p']['cid']
        product_describe = product['p']['shop_describe']
        # 查询每个产品的店铺页面
        sql = """MATCH (p:Product {name:"%s"})-[:PRODUCT_PAGE_IS]->(pp:ProductPage) RETURN pp""" % (product_name)
        product_page_res = graph.run(sql)
        product_pages = product_page_res.data()
        # 产品和每个店铺的页面两两组成样本对,
        # 遍历这个产品的店铺页面
        for page in product_pages:
            page_title = page['pp']['title']
            page_content = page['pp']['content']  # 这个店铺的说明页
            one_data = {
                "context_left": "", # 提及左侧都为空，
                "context_right": page_content, #把页面内容作为提及的右侧文本
                "mention": page_title, #我们把店铺页面的标题作为提及词
                "label": product_describe, # 作为实体的描述
                "label_id": product_id, # 作为实体的id
                "label_title": product_name, # 作为实体
            }
            data.append(one_data)
    print("共收集到%d个样本" % len(data))
    # 保存成jsonl格式，8:1:1分成训练集，测试集，开发集
    base_dir = "data/product"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    train_file = os.path.join(base_dir, "train.jsonl")
    test_file = os.path.join(base_dir, "test.jsonl")
    valid_file = os.path.join(base_dir, "valid.jsonl")
    #打乱数据集
    random.shuffle(data)
    #拆分数据集
    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    valid_data = data[int(len(data) * 0.9):]
    print("训练集，测试集，开发集的样本数量分别是%d，%d，%d" % (len(train_data), len(test_data), len(valid_data)))
    print(f"保存到{train_file}，{test_file}，{valid_file}")
    # 保存数据
    with open(train_file, 'w', encoding='utf-8') as f:
        for one_data in train_data:
            f.write(json.dumps(one_data, ensure_ascii=False) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f:   # 测试集
        for one_data in test_data:
            f.write(json.dumps(one_data, ensure_ascii=False) + '\n')
    with open(valid_file, 'w', encoding='utf-8') as f:  # 验证集
        for one_data in valid_data:
            f.write(json.dumps(one_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    get_data()