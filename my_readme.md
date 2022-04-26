# 目录结构
```
BLINK/
├── blink
│   ├── biencoder
│   │   ├── biencoder.py
│   │   ├── data_process.py
│   │   ├── eval_biencoder.py
│   │   ├── __init__.py
│   │   ├── nn_prediction.py
│   │   ├── train_biencoder.py
│   │   └── zeshel_utils.py
│   ├── build_faiss_index.py
│   ├── candidate_data_fetcher.py
│   ├── candidate_generation.py
│   ├── candidate_ranking
│   │   ├── bert_reranking.py
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── candidate_retrieval
│   │   ├── candidate_generators.py
│   │   ├── data_ingestion.py
│   │   ├── dataset.py
│   │   ├── enrich_data.py
│   │   ├── evaluator.py
│   │   ├── generate_wiki2wikidata_mappings.py
│   │   ├── json_data_generation.py
│   │   ├── link_wikipedia_and_wikidata.py
│   │   ├── perform_and_evaluate_candidate_retrieval_multithreaded.py
│   │   ├── process_intro_sents.py
│   │   ├── process_wikidata.py
│   │   ├── process_wiki_extractor_output_full.py
│   │   ├── process_wiki_extractor_output_links.py
│   │   ├── process_wiki_extractor_output.py
│   │   ├── README.md
│   │   ├── scripts
│   │   │   ├── create_solr_collections.sh
│   │   │   ├── generate_wiki2wikidata_mapping.sh
│   │   │   ├── get_processed_data.sh
│   │   │   ├── ingest_data.sh
│   │   │   ├── ingestion_wrapper.sh
│   │   │   ├── init_collection.sh
│   │   │   ├── link_wikipedia_and_wikidata.sh
│   │   │   ├── process_wikidata_dump.sh
│   │   │   ├── process_wikipedia_dump_links.sh
│   │   │   └── process_wikipedia_dump.sh
│   │   └── utils.py
│   ├── common
│   │   ├── optimizer.py
│   │   ├── params.py
│   │   └── ranker_base.py
│   ├── crossencoder
│   │   ├── crossencoder.py
│   │   ├── data_process.py
│   │   └── train_cross.py
│   ├── indexer
│   │   └── faiss_indexer.py
│   ├── __init__.py
│   ├── main_dense.py
│   ├── main_solr.py
│   ├── ner.py
│   ├── reranker.py
│   ├── run_benchmark.py
│   └── utils.py
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── download_blink_models.sh
├── download_elq_models.sh
├── elq
│   ├── biencoder
│   │   ├── allennlp_span_utils.py
│   │   ├── biencoder.py
│   │   ├── data_process.py
│   │   ├── train_biencoder.py
│   │   └── utils.py
│   ├── build_faiss_index.py
│   ├── candidate_ranking
│   │   └── utils.py
│   ├── common
│   │   ├── params.py
│   │   └── ranker_base.py
│   ├── index
│   │   └── faiss_indexer.py
│   ├── main_dense.py
│   ├── README.md
│   ├── requirements.txt
│   └── vcg_utils
│       └── measures.py
├── elq_slurm_scripts
│   ├── eval_elq.sh
│   ├── get_entity_encodings.sh
│   └── train_elq.sh
├── examples
│   ├── text.txt
│   └── zeshel
│       ├── create_BLINK_zeshel_data.py
│       ├── get_zeshel_data.sh
│       └── README.md
├── img
│   ├── blink_logo_banner.png
│   ├── example_result_dark.png
│   ├── example_result_light.png
│   └── logo.png
├── LICENSE
├── models
│   ├── all_entities_large.t7   #所有的实体，实体对应的嵌入，torch.Size([5903527, 1024])
│   ├── biencoder_wiki_large.bin   #双编码器模型
│   ├── biencoder_wiki_large.json  双编码器模型配置
│   ├── crossencoder_wiki_large.bin  #交叉编码模型的路径
│   ├── crossencoder_wiki_large.json #交叉编码模型的配置
│   ├── elq_large_params.txt
│   ├── elq_webqsp_large.bin
│   ├── elq_wiki_large.bin
│   ├── entity.jsonl   #所有WIKIPEDIA的数据，原始数据， 数据条数：5903527， 实体目录的路径, 实体的原数据， 一条数据  #{"text": " Anarchism is an anti-authoritarian political philosophy that rejects hierarchies deemed unjust and advocates their replacement with self-managed, self-governed societies based on voluntary, cooperative institutions. These institutions are often described as stateless societies, although several authors have defined them more specifically as distinct institutions based on non-hierarchical or free associations. Anarchism's central disagreement with other ideologies is that it holds the state to be undesirable, unnecessary, and harmful.  Anarchism is usually placed on the far-left of the political spectrum, and much of its economics and legal philosophy reflect anti-authoritarian interpretations of communism, collectivism, syndicalism, mutualism, or participatory economics. As anarchism does not offer a fixed body of doctrine from a single particular worldview, many anarchist types and traditions exist and varieties of anarchy diverge widely. Anarchist schools of thought can differ fundamentally, supporting anything from extreme individualism to complete collectivism. Strains of anarchism have often been divided into the categories of social and individualist anarchism, or similar dual classifications. ", "idx": "https://en.wikipedia.org/wiki?curid=12", "title": "Anarchism", "entity": "Anarchism"}
│   ├── entity_token_ids_128.t7
│   └── faiss_hnsw_index.pkl
├── README.md
├── requirements.txt
├── scripts
│   ├── benchmark_ELQ.sh
│   ├── create_BLINK_benchmark_data.py
│   ├── generate_candidates.py
│   ├── get_train_and_benchmark_data.sh
│   ├── merge_candidates.py
│   ├── train_ELQ.sh
│   └── tune_hyperparams_new.py
└── setup.py

20 directories, 105 files
```

# 一条zeshel数据集格式
```console
context_left: 提及左侧的样本内容
context_right: 提及的右侧的样本内容
mention： 提及的文本内容
label： 实体的上下文，即实体的描述，
label_id： 实体的id，即真正的实体，即知识图谱中对应的实体的id
label_title： 实体的标题，即实体的名字
world: 实体的主题，或者实体的类型

{
  "context_left": "goodnight keith moon goodnight keith moon ( isbn 0956011926 ) is a parody of \" goodnight moon \" by bruce worden and clare cross published in 2011 . it is a retelling of the \"",
  "context_right": "\" story with elements changed to fit into the life of the who drummer keith moon . a two - page illustrated spread depicts a room with moon sleeping on a bed , passed out with beer bottles and vomit surrounding him . the room is filled with various bits of paraphernalia including a trashed drumset , a pinball machine , and an animal doll on top of a bookcase . the scene is representative of the party lifestyle of a rock n \u0027 roll star , particularly the out - of - control aspect of moon \u0027 s personality as seen in the public eye . the inclusion of animal alludes to a persistent rumor that the muppet was based on moon despite the lack of evidence",
  "mention": "goodnight moon",
  "label": "Margaret Wise Brown Margaret Wise Brown ( 1910 – 1952 ) was a prolific writer of children \u0027 s books including \" The Noisy Book \" ( 1939 ) , \" The Runaway Bunny \" ( 1942 ) , \" The Little Island \" ( 1946 ) , \" Goodnight Moon \" ( 1947 ) , and many others . In \" Sesame Street \" Episode 2620 , Mr . Handford tries to get Alice Snuffleupagus to fall asleep by reading her \" Big Red Barn \" . Felicia Bond \u0027 s illustrations are shown as he reads the text . At the end of Episode 3785 , Oscar the Grouch reads a worm version of \" Goodnight Moon \" to Slimey after returning from his trip to the Moon .",
  "label_id": 15817,
  "label_title": "Margaret Wise Brown",
  "world": "muppets"
}
```

# gensim需要安装gensim==3.8.3，否则报错AttributeError: 'KeyedVectors' object has no attribute 'key_to_index'
https://github.com/flairNLP/flair/issues/2196

# 示例：
```
Beijing is the capital of China

beijing
id:2301073
title:Beijing
text: Beijing ( , nonstandard ; ), alternatively romanized as Peking ( ), is the capital of the People's Republic of China, the world's third most populous city proper, and most populous capital city. The city, located in northern China, is governed as a munici

china
id:2397
title:China
text: China (; lit. "Middle Kingdom"), officially the People's Republic of China (PRC), is a country in East Asia and the world's most populous country, with a population of around /1e9 round 3 billion. Covering approximately , it is the fourth largest country 
```

# wikipedia实体数量
5903527个，嵌入的维度 torch.Size([5903527, 1024])

# 使用测试集测试文件
python blink/main_dense.py --test_mentions examples/test_mentions.jsonl

# 交互性测试， 加载all_entities_large.t7文件缓慢
python blink/main_dense.py -i

# 快速加载索引, 加载pkl文件缓慢, 64GB内存不够用
python blink/main_dense.py --faiss_index hnsw --index_path models/faiss_hnsw_index.pkl

# Step1: 训练biencoder模型
data/zeshel/blink_format目录下的数据集也是jsonl格式的
python blink/biencoder/train_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 8 --eval_batch_size 8 --bert_model bert-base-uncased --type_optimization all_encoder_layers
耗时3个半小时, 评估的准确率是: 0.98860
```
04/20/2022 21:34:18 - INFO - Blink -   开始评估这个epoch的微调模型
04/20/2022 21:34:49 - INFO - Blink -   评估的准确率是: 0.98860
04/20/2022 21:35:02 - INFO - Blink -   保存模型的checkpoint文件到models/zeshel/biencoder/pytorch_model.bin
04/20/2022 21:35:02 - INFO - Blink -   保存模型的config文件到models/zeshel/biencoder/config.json
04/20/2022 21:35:02 - INFO - Blink -   保存模型的tokenizer文件到目录models/zeshel/biencoder
```
# 双编码器的模型结构
```console
BiEncoderModule(
  (context_encoder): BertEncoder(
    (bert_model): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )
  )
  (cand_encoder): BertEncoder(
    (bert_model): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )
  )
)
```

# Step2: 从Biencoder模型中获得训练和测试数据集的前64预测结果。 top_k =64 ,第三步的训练那么会严重影响batch_size的大小，所以这里设置为32比较好
python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 8 --eval_batch_size 1 --top_k 32 --save_topk_result --bert_model bert-base-uncased --mode train,valid,test --zeshel True
耗时1个半小时
```console
04/20/2022 22:58:42 - INFO - Blink -   World size : 16
100%|█████████████████████████████████████████████████████████████████████████████| 10000/10000 [02:44<00:00, 60.76it/s]
04/20/2022 23:01:26 - INFO - Blink -   In world forgotten_realms
04/20/2022 23:01:26 - INFO - Blink -   Total: 1200 examples. r@1: 0.4992 r@4: 0.7017 r@8: 0.7575 r@16: 0.8150 r@32: 0.8533 r@64: 0.8850
04/20/2022 23:01:26 - INFO - Blink -   In world lego
04/20/2022 23:01:26 - INFO - Blink -   Total: 1199 examples. r@1: 0.4145 r@4: 0.6464 r@8: 0.7264 r@16: 0.7865 r@32: 0.8365 r@64: 0.8724
04/20/2022 23:01:26 - INFO - Blink -   In world star_trek
04/20/2022 23:01:26 - INFO - Blink -   Total: 4227 examples. r@1: 0.3866 r@4: 0.5775 r@8: 0.6428 r@16: 0.6996 r@32: 0.7454 r@64: 0.7878
04/20/2022 23:01:26 - INFO - Blink -   In world yugioh
04/20/2022 23:01:26 - INFO - Blink -   Total: 3374 examples. r@1: 0.2596 r@4: 0.4366 r@8: 0.5113 r@16: 0.5697 r@32: 0.6144 r@64: 0.6580
04/20/2022 23:01:26 - INFO - Blink -   Total: 10000 examples. r@1: 0.3606 r@4: 0.5531 r@8: 0.6222 r@16: 0.6800 r@32: 0.7251 r@64: 0.7658
04/20/2022 23:01:27 - INFO - Blink -   保存topk的测试结果到目录： models/zeshel/top64_candidates
04/20/2022 23:01:27 - INFO - Blink -   保存topk的测试结果到文件： models/zeshel/top64_candidates/test.t7
```

# 保存文件的目录结构
```console
BLINK/models/zeshel$ tree .
.
├── biencoder  # 双编码器训练结果
│   ├── config.json
│   ├── epoch_0
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── epoch_1
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── epoch_2
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── epoch_3
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── epoch_4
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── log.txt
│   ├── pytorch_model.bin
│   ├── training_params.txt
│   ├── training_time.txt
│   └── vocab.txt
├── log.txt
└── top64_candidates  从双编码器获取的前64预测结果
    ├── test.t7
    ├── train.t7
    └── valid.t7
    
# 32个候选实体是，数据集大小
ls -alh models/zeshel/top32_candidates/
total 1.9G
drwxrwxr-x 2   4.0K Apr 21 16:57 .
drwxrwxr-x 6   4.0K Apr 21 16:42 ..
-rw-rw-r-- 1   234M Apr 21 16:57 test.t7
-rw-rw-r-- 1   1.4G Apr 21 16:42 train.t7
-rw-rw-r-- 1   267M Apr 21 16:49 valid.t7

```

# Step3: 训练和评估交叉编码器模型: 可以加--debug，仅使用200条数据 , 训练时间非常长，1个epoch约24小时左右

## 使用top32的样本训练
python blink/crossencoder/train_cross.py --data_path models/zeshel/top32_candidates/ --output_path models/zeshel/crossencoder --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --bert_model bert-base-uncased --type_optimization all_encoder_layers --add_linear --zeshel True

## 使用top64的样本训练
python blink/crossencoder/train_cross.py --data_path models/zeshel/top64_candidates/ --output_path models/zeshel/crossencoder --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --bert_model bert-base-uncased --type_optimization all_encoder_layers --add_linear --zeshel True

## 或者减小序列长度, 18915MiB显存
python blink/crossencoder/train_cross.py --data_path models/zeshel/top32_candidates/ --output_path models/zeshel/crossencoder --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 90 --max_cand_length 90 --max_seq_length 188 --train_batch_size 2 --eval_batch_size 2 --bert_model bert-base-uncased --type_optimization all_encoder_layers --add_linear --zeshel True

训练1个epoch的验证集的准确率日志：
grep 准确率 log.txt
```console
04/21/2022 09:45:55 - INFO - Blink -   Macro准确率: 0.00236
04/21/2022 09:45:55 - INFO - Blink -   Micro准确率: 0.00196
04/21/2022 11:20:03 - INFO - Blink -   Macro准确率: 0.20574
04/21/2022 11:20:03 - INFO - Blink -   Micro准确率: 0.19122
04/21/2022 15:39:30 - INFO - Blink -   Macro准确率: 0.17623
04/21/2022 15:39:30 - INFO - Blink -   Micro准确率: 0.18000
04/22/2022 09:32:14 - INFO - Blink -   Macro准确率: 0.00961
04/22/2022 09:32:14 - INFO - Blink -   Micro准确率: 0.01041
04/22/2022 09:41:22 - INFO - Blink -   Macro准确率: 0.01404
04/22/2022 09:41:22 - INFO - Blink -   Micro准确率: 0.01622
04/22/2022 09:50:32 - INFO - Blink -   Macro准确率: 0.02740
04/22/2022 09:50:32 - INFO - Blink -   Micro准确率: 0.03233
04/22/2022 09:59:43 - INFO - Blink -   Macro准确率: 0.08972
04/22/2022 09:59:43 - INFO - Blink -   Micro准确率: 0.08766
04/22/2022 10:08:53 - INFO - Blink -   Macro准确率: 0.18727
04/22/2022 10:08:53 - INFO - Blink -   Micro准确率: 0.17423
04/22/2022 10:18:04 - INFO - Blink -   Macro准确率: 0.21564
04/22/2022 10:18:04 - INFO - Blink -   Micro准确率: 0.20257
04/22/2022 10:27:14 - INFO - Blink -   Macro准确率: 0.30276
04/22/2022 10:27:14 - INFO - Blink -   Micro准确率: 0.28987
04/22/2022 10:36:25 - INFO - Blink -   Macro准确率: 0.34005
04/22/2022 10:36:25 - INFO - Blink -   Micro准确率: 0.32595
04/22/2022 10:45:36 - INFO - Blink -   Macro准确率: 0.36104
04/22/2022 10:45:36 - INFO - Blink -   Micro准确率: 0.34798
04/22/2022 10:54:46 - INFO - Blink -   Macro准确率: 0.37181
04/22/2022 10:54:46 - INFO - Blink -   Micro准确率: 0.35707
04/22/2022 11:03:57 - INFO - Blink -   Macro准确率: 0.39504
04/22/2022 11:03:57 - INFO - Blink -   Micro准确率: 0.37838
04/22/2022 11:13:08 - INFO - Blink -   Macro准确率: 0.41569
04/22/2022 11:13:08 - INFO - Blink -   Micro准确率: 0.40259
04/22/2022 11:22:19 - INFO - Blink -   Macro准确率: 0.42884
04/22/2022 11:22:19 - INFO - Blink -   Micro准确率: 0.41240
04/22/2022 11:31:29 - INFO - Blink -   Macro准确率: 0.43286
04/22/2022 11:31:29 - INFO - Blink -   Micro准确率: 0.41506
04/22/2022 11:40:40 - INFO - Blink -   Macro准确率: 0.43757
04/22/2022 11:40:40 - INFO - Blink -   Micro准确率: 0.42039
04/22/2022 11:49:51 - INFO - Blink -   Macro准确率: 0.46944
04/22/2022 11:49:51 - INFO - Blink -   Micro准确率: 0.45320
04/22/2022 11:59:02 - INFO - Blink -   Macro准确率: 0.49292
04/22/2022 11:59:02 - INFO - Blink -   Micro准确率: 0.47524
04/22/2022 12:08:13 - INFO - Blink -   Macro准确率: 0.48837
04/22/2022 12:08:13 - INFO - Blink -   Micro准确率: 0.46882
04/22/2022 12:17:24 - INFO - Blink -   Macro准确率: 0.50321
04/22/2022 12:17:24 - INFO - Blink -   Micro准确率: 0.48287
04/22/2022 12:26:35 - INFO - Blink -   Macro准确率: 0.50924
04/22/2022 12:26:35 - INFO - Blink -   Micro准确率: 0.49062
04/22/2022 12:35:46 - INFO - Blink -   Macro准确率: 0.51651
04/22/2022 12:35:46 - INFO - Blink -   Micro准确率: 0.50006
04/22/2022 12:44:57 - INFO - Blink -   Macro准确率: 0.52494
04/22/2022 12:44:57 - INFO - Blink -   Micro准确率: 0.50575
04/22/2022 12:54:08 - INFO - Blink -   Macro准确率: 0.52852
04/22/2022 12:54:08 - INFO - Blink -   Micro准确率: 0.51495
04/22/2022 13:03:19 - INFO - Blink -   Macro准确率: 0.53393
04/22/2022 13:03:19 - INFO - Blink -   Micro准确率: 0.52137
04/22/2022 13:12:29 - INFO - Blink -   Macro准确率: 0.53158
04/22/2022 13:12:29 - INFO - Blink -   Micro准确率: 0.51762
04/22/2022 13:21:40 - INFO - Blink -   Macro准确率: 0.55275
04/22/2022 13:21:40 - INFO - Blink -   Micro准确率: 0.53953
04/22/2022 13:30:51 - INFO - Blink -   Macro准确率: 0.53765
04/22/2022 13:30:51 - INFO - Blink -   Micro准确率: 0.53203
04/22/2022 13:40:03 - INFO - Blink -   Macro准确率: 0.56848
04/22/2022 13:40:03 - INFO - Blink -   Micro准确率: 0.55334
04/22/2022 13:49:15 - INFO - Blink -   Macro准确率: 0.53378
04/22/2022 13:49:15 - INFO - Blink -   Micro准确率: 0.52040
04/22/2022 13:58:29 - INFO - Blink -   Macro准确率: 0.56859
04/22/2022 13:58:29 - INFO - Blink -   Micro准确率: 0.55757
04/22/2022 14:07:42 - INFO - Blink -   Macro准确率: 0.55225
04/22/2022 14:07:42 - INFO - Blink -   Micro准确率: 0.53602
04/22/2022 14:16:55 - INFO - Blink -   Macro准确率: 0.56802
04/22/2022 14:16:55 - INFO - Blink -   Micro准确率: 0.55273
04/22/2022 14:26:08 - INFO - Blink -   Macro准确率: 0.56764
04/22/2022 14:26:08 - INFO - Blink -   Micro准确率: 0.55588
04/22/2022 14:35:22 - INFO - Blink -   Macro准确率: 0.54720
04/22/2022 14:35:22 - INFO - Blink -   Micro准确率: 0.53481
04/22/2022 14:44:36 - INFO - Blink -   Macro准确率: 0.57249
04/22/2022 14:44:36 - INFO - Blink -   Micro准确率: 0.55697
04/22/2022 14:53:50 - INFO - Blink -   Macro准确率: 0.58698
04/22/2022 14:53:50 - INFO - Blink -   Micro准确率: 0.57440
04/22/2022 15:03:06 - INFO - Blink -   Macro准确率: 0.57722
04/22/2022 15:03:06 - INFO - Blink -   Micro准确率: 0.55915
04/22/2022 15:12:21 - INFO - Blink -   Macro准确率: 0.59883
04/22/2022 15:12:21 - INFO - Blink -   Micro准确率: 0.58082
04/22/2022 15:21:36 - INFO - Blink -   Macro准确率: 0.60816
04/22/2022 15:21:36 - INFO - Blink -   Micro准确率: 0.59535
04/22/2022 15:30:51 - INFO - Blink -   Macro准确率: 0.59452
04/22/2022 15:30:51 - INFO - Blink -   Micro准确率: 0.57392
04/22/2022 15:40:06 - INFO - Blink -   Macro准确率: 0.62178
04/22/2022 15:40:06 - INFO - Blink -   Micro准确率: 0.60673
04/22/2022 15:49:22 - INFO - Blink -   Macro准确率: 0.61359
04/22/2022 15:49:22 - INFO - Blink -   Micro准确率: 0.59801
04/22/2022 15:58:38 - INFO - Blink -   Macro准确率: 0.53560
04/22/2022 15:58:38 - INFO - Blink -   Micro准确率: 0.51302
04/22/2022 16:07:53 - INFO - Blink -   Macro准确率: 0.60878
04/22/2022 16:07:53 - INFO - Blink -   Micro准确率: 0.58954
04/22/2022 16:17:09 - INFO - Blink -   Macro准确率: 0.62309
04/22/2022 16:17:09 - INFO - Blink -   Micro准确率: 0.60722
04/22/2022 16:26:25 - INFO - Blink -   Macro准确率: 0.63766
04/22/2022 16:26:25 - INFO - Blink -   Micro准确率: 0.62259
04/22/2022 16:35:41 - INFO - Blink -   Macro准确率: 0.62019
04/22/2022 16:35:41 - INFO - Blink -   Micro准确率: 0.60867
04/22/2022 16:44:56 - INFO - Blink -   Macro准确率: 0.62478
04/22/2022 16:44:56 - INFO - Blink -   Micro准确率: 0.61170
04/22/2022 16:54:11 - INFO - Blink -   Macro准确率: 0.62675
04/22/2022 16:54:11 - INFO - Blink -   Micro准确率: 0.61254
04/22/2022 17:03:26 - INFO - Blink -   Macro准确率: 0.59976
04/22/2022 17:03:26 - INFO - Blink -   Micro准确率: 0.58663
04/22/2022 17:12:41 - INFO - Blink -   Macro准确率: 0.64843
04/22/2022 17:12:41 - INFO - Blink -   Micro准确率: 0.63398
04/22/2022 17:21:56 - INFO - Blink -   Macro准确率: 0.64275
04/22/2022 17:21:56 - INFO - Blink -   Micro准确率: 0.62477
04/22/2022 17:31:10 - INFO - Blink -   Macro准确率: 0.63203
04/22/2022 17:31:10 - INFO - Blink -   Micro准确率: 0.61472
04/22/2022 17:40:25 - INFO - Blink -   Macro准确率: 0.61786
04/22/2022 17:40:25 - INFO - Blink -   Micro准确率: 0.59983
04/22/2022 17:49:39 - INFO - Blink -   Macro准确率: 0.63519
04/22/2022 17:49:39 - INFO - Blink -   Micro准确率: 0.62041
04/22/2022 17:58:53 - INFO - Blink -   Macro准确率: 0.52644
04/22/2022 17:58:53 - INFO - Blink -   Micro准确率: 0.50999
04/22/2022 18:08:07 - INFO - Blink -   Macro准确率: 0.62080
04/22/2022 18:08:07 - INFO - Blink -   Micro准确率: 0.60358
04/22/2022 18:17:20 - INFO - Blink -   Macro准确率: 0.63643
04/22/2022 18:17:20 - INFO - Blink -   Micro准确率: 0.62066
04/22/2022 18:26:34 - INFO - Blink -   Macro准确率: 0.64301
04/22/2022 18:26:34 - INFO - Blink -   Micro准确率: 0.63058
04/22/2022 18:35:53 - INFO - Blink -   Macro准确率: 0.65046
04/22/2022 18:35:53 - INFO - Blink -   Micro准确率: 0.63337
04/22/2022 18:45:14 - INFO - Blink -   Macro准确率: 0.65154
04/22/2022 18:45:14 - INFO - Blink -   Micro准确率: 0.63531
04/22/2022 18:54:36 - INFO - Blink -   Macro准确率: 0.65334
04/22/2022 18:54:36 - INFO - Blink -   Micro准确率: 0.63737
04/22/2022 19:04:00 - INFO - Blink -   Macro准确率: 0.66405
04/22/2022 19:04:00 - INFO - Blink -   Micro准确率: 0.64923
04/22/2022 19:13:24 - INFO - Blink -   Macro准确率: 0.64596
04/22/2022 19:13:24 - INFO - Blink -   Micro准确率: 0.62489
04/22/2022 19:22:48 - INFO - Blink -   Macro准确率: 0.64251
04/22/2022 19:22:48 - INFO - Blink -   Micro准确率: 0.61981
04/22/2022 19:32:13 - INFO - Blink -   Macro准确率: 0.63453
04/22/2022 19:32:13 - INFO - Blink -   Micro准确率: 0.61884
04/22/2022 19:41:37 - INFO - Blink -   Macro准确率: 0.65875
04/22/2022 19:41:37 - INFO - Blink -   Micro准确率: 0.64136
04/22/2022 19:51:00 - INFO - Blink -   Macro准确率: 0.64866
04/22/2022 19:51:00 - INFO - Blink -   Micro准确率: 0.63543
04/22/2022 20:00:24 - INFO - Blink -   Macro准确率: 0.65527
04/22/2022 20:00:24 - INFO - Blink -   Micro准确率: 0.64221
04/22/2022 20:09:48 - INFO - Blink -   Macro准确率: 0.64929
04/22/2022 20:09:48 - INFO - Blink -   Micro准确率: 0.63482
04/22/2022 20:19:13 - INFO - Blink -   Macro准确率: 0.66122
04/22/2022 20:19:13 - INFO - Blink -   Micro准确率: 0.64548
04/22/2022 20:28:39 - INFO - Blink -   Macro准确率: 0.65480
04/22/2022 20:28:39 - INFO - Blink -   Micro准确率: 0.63773
04/22/2022 20:38:01 - INFO - Blink -   Macro准确率: 0.66650
04/22/2022 20:38:01 - INFO - Blink -   Micro准确率: 0.65105
04/22/2022 20:47:22 - INFO - Blink -   Macro准确率: 0.67016
04/22/2022 20:47:22 - INFO - Blink -   Micro准确率: 0.64887
04/22/2022 20:56:40 - INFO - Blink -   Macro准确率: 0.64819
04/22/2022 20:56:40 - INFO - Blink -   Micro准确率: 0.62683
04/22/2022 21:05:58 - INFO - Blink -   Macro准确率: 0.67330
04/22/2022 21:05:58 - INFO - Blink -   Micro准确率: 0.65613
04/22/2022 21:15:15 - INFO - Blink -   Macro准确率: 0.66654
04/22/2022 21:15:15 - INFO - Blink -   Micro准确率: 0.65311
04/22/2022 21:24:31 - INFO - Blink -   Macro准确率: 0.66815
04/22/2022 21:24:31 - INFO - Blink -   Micro准确率: 0.65298
04/22/2022 21:33:48 - INFO - Blink -   Macro准确率: 0.65465
04/22/2022 21:33:48 - INFO - Blink -   Micro准确率: 0.63240
04/22/2022 21:43:04 - INFO - Blink -   Macro准确率: 0.65861
04/22/2022 21:43:04 - INFO - Blink -   Micro准确率: 0.63361
04/22/2022 21:52:20 - INFO - Blink -   Macro准确率: 0.64752
04/22/2022 21:52:20 - INFO - Blink -   Micro准确率: 0.63361
04/22/2022 22:01:36 - INFO - Blink -   Macro准确率: 0.65863
04/22/2022 22:01:36 - INFO - Blink -   Micro准确率: 0.63095
04/22/2022 22:10:51 - INFO - Blink -   Macro准确率: 0.66800
04/22/2022 22:10:51 - INFO - Blink -   Micro准确率: 0.64511
04/22/2022 22:20:07 - INFO - Blink -   Macro准确率: 0.66180
04/22/2022 22:20:07 - INFO - Blink -   Micro准确率: 0.63603
04/22/2022 22:29:23 - INFO - Blink -   Macro准确率: 0.67561
04/22/2022 22:29:23 - INFO - Blink -   Micro准确率: 0.66194
04/22/2022 22:38:38 - INFO - Blink -   Macro准确率: 0.68044
04/22/2022 22:38:38 - INFO - Blink -   Micro准确率: 0.66521
04/22/2022 22:47:54 - INFO - Blink -   Macro准确率: 0.68007
04/22/2022 22:47:54 - INFO - Blink -   Micro准确率: 0.66364
04/22/2022 22:57:09 - INFO - Blink -   Macro准确率: 0.69375
04/22/2022 22:57:09 - INFO - Blink -   Micro准确率: 0.67926
04/22/2022 23:06:25 - INFO - Blink -   Macro准确率: 0.66334
04/22/2022 23:06:25 - INFO - Blink -   Micro准确率: 0.65202
04/22/2022 23:15:40 - INFO - Blink -   Macro准确率: 0.68329
04/22/2022 23:15:40 - INFO - Blink -   Micro准确率: 0.66425
04/22/2022 23:24:56 - INFO - Blink -   Macro准确率: 0.68266
04/22/2022 23:24:56 - INFO - Blink -   Micro准确率: 0.66630
04/22/2022 23:34:11 - INFO - Blink -   Macro准确率: 0.68187
04/22/2022 23:34:11 - INFO - Blink -   Micro准确率: 0.66497
04/22/2022 23:43:27 - INFO - Blink -   Macro准确率: 0.67897
04/22/2022 23:43:27 - INFO - Blink -   Micro准确率: 0.66485
04/22/2022 23:52:42 - INFO - Blink -   Macro准确率: 0.67983
04/22/2022 23:52:42 - INFO - Blink -   Micro准确率: 0.65868
04/23/2022 00:01:57 - INFO - Blink -   Macro准确率: 0.68501
04/23/2022 00:01:57 - INFO - Blink -   Micro准确率: 0.66509
04/23/2022 00:11:12 - INFO - Blink -   Macro准确率: 0.66190
04/23/2022 00:11:12 - INFO - Blink -   Micro准确率: 0.64693
04/23/2022 00:20:28 - INFO - Blink -   Macro准确率: 0.69081
04/23/2022 00:20:28 - INFO - Blink -   Micro准确率: 0.67587
04/23/2022 00:29:43 - INFO - Blink -   Macro准确率: 0.68808
04/23/2022 00:29:43 - INFO - Blink -   Micro准确率: 0.67030
04/23/2022 00:38:58 - INFO - Blink -   Macro准确率: 0.68108
04/23/2022 00:38:58 - INFO - Blink -   Micro准确率: 0.66533
04/23/2022 00:48:13 - INFO - Blink -   Macro准确率: 0.68343
04/23/2022 00:48:13 - INFO - Blink -   Micro准确率: 0.67030
04/23/2022 00:57:28 - INFO - Blink -   Macro准确率: 0.65007
04/23/2022 00:57:28 - INFO - Blink -   Micro准确率: 0.63180
04/23/2022 01:06:43 - INFO - Blink -   Macro准确率: 0.68601
04/23/2022 01:06:43 - INFO - Blink -   Micro准确率: 0.67066
04/23/2022 01:15:58 - INFO - Blink -   Macro准确率: 0.69309
04/23/2022 01:15:58 - INFO - Blink -   Micro准确率: 0.67478
04/23/2022 01:25:13 - INFO - Blink -   Macro准确率: 0.69806
04/23/2022 01:25:13 - INFO - Blink -   Micro准确率: 0.67708
04/23/2022 01:34:28 - INFO - Blink -   Macro准确率: 0.68440
04/23/2022 01:34:28 - INFO - Blink -   Micro准确率: 0.66727
04/23/2022 01:43:43 - INFO - Blink -   Macro准确率: 0.62142
04/23/2022 01:43:43 - INFO - Blink -   Micro准确率: 0.60697
04/23/2022 01:52:57 - INFO - Blink -   Macro准确率: 0.65553
04/23/2022 01:52:57 - INFO - Blink -   Micro准确率: 0.63313
04/23/2022 02:02:12 - INFO - Blink -   Macro准确率: 0.69235
04/23/2022 02:02:12 - INFO - Blink -   Micro准确率: 0.68047
04/23/2022 02:11:27 - INFO - Blink -   Macro准确率: 0.69416
04/23/2022 02:11:27 - INFO - Blink -   Micro准确率: 0.67890
04/23/2022 02:20:42 - INFO - Blink -   Macro准确率: 0.67174
04/23/2022 02:20:42 - INFO - Blink -   Micro准确率: 0.65662
04/23/2022 02:29:56 - INFO - Blink -   Macro准确率: 0.68489
04/23/2022 02:29:56 - INFO - Blink -   Micro准确率: 0.66873
04/23/2022 02:39:11 - INFO - Blink -   Macro准确率: 0.69170
04/23/2022 02:39:11 - INFO - Blink -   Micro准确率: 0.67744
04/23/2022 02:48:25 - INFO - Blink -   Macro准确率: 0.69993
04/23/2022 02:48:25 - INFO - Blink -   Micro准确率: 0.67902
04/23/2022 02:57:40 - INFO - Blink -   Macro准确率: 0.69676
04/23/2022 02:57:40 - INFO - Blink -   Micro准确率: 0.67720
04/23/2022 03:06:54 - INFO - Blink -   Macro准确率: 0.69854
04/23/2022 03:06:54 - INFO - Blink -   Micro准确率: 0.68144
04/23/2022 03:16:09 - INFO - Blink -   Macro准确率: 0.68790
04/23/2022 03:16:09 - INFO - Blink -   Micro准确率: 0.67393
04/23/2022 03:25:24 - INFO - Blink -   Macro准确率: 0.70851
04/23/2022 03:25:24 - INFO - Blink -   Micro准确率: 0.69137
04/23/2022 03:34:38 - INFO - Blink -   Macro准确率: 0.69917
04/23/2022 03:34:38 - INFO - Blink -   Micro准确率: 0.68204
04/23/2022 03:43:53 - INFO - Blink -   Macro准确率: 0.71135
04/23/2022 03:43:53 - INFO - Blink -   Micro准确率: 0.69088
04/23/2022 03:53:07 - INFO - Blink -   Macro准确率: 0.69903
04/23/2022 03:53:07 - INFO - Blink -   Micro准确率: 0.68338
04/23/2022 04:02:22 - INFO - Blink -   Macro准确率: 0.71413
04/23/2022 04:02:22 - INFO - Blink -   Micro准确率: 0.69367
04/23/2022 04:11:36 - INFO - Blink -   Macro准确率: 0.70863
04/23/2022 04:11:36 - INFO - Blink -   Micro准确率: 0.68834
04/23/2022 04:20:51 - INFO - Blink -   Macro准确率: 0.70558
04/23/2022 04:20:51 - INFO - Blink -   Micro准确率: 0.68495
04/23/2022 04:30:05 - INFO - Blink -   Macro准确率: 0.70057
04/23/2022 04:30:05 - INFO - Blink -   Micro准确率: 0.68168
04/23/2022 04:39:20 - INFO - Blink -   Macro准确率: 0.70820
04/23/2022 04:39:20 - INFO - Blink -   Micro准确率: 0.68907
04/23/2022 04:48:34 - INFO - Blink -   Macro准确率: 0.70364
04/23/2022 04:48:34 - INFO - Blink -   Micro准确率: 0.68786
04/23/2022 04:57:48 - INFO - Blink -   Macro准确率: 0.70746
04/23/2022 04:57:48 - INFO - Blink -   Micro准确率: 0.68677
04/23/2022 05:07:02 - INFO - Blink -   Macro准确率: 0.70608
04/23/2022 05:07:02 - INFO - Blink -   Micro准确率: 0.68870
04/23/2022 05:16:16 - INFO - Blink -   Macro准确率: 0.71399
04/23/2022 05:16:16 - INFO - Blink -   Micro准确率: 0.69318
04/23/2022 05:25:30 - INFO - Blink -   Macro准确率: 0.71788
04/23/2022 05:25:30 - INFO - Blink -   Micro准确率: 0.69609
04/23/2022 05:34:45 - INFO - Blink -   Macro准确率: 0.70837
04/23/2022 05:34:45 - INFO - Blink -   Micro准确率: 0.68628
04/23/2022 05:43:58 - INFO - Blink -   Macro准确率: 0.70908
04/23/2022 05:43:58 - INFO - Blink -   Micro准确率: 0.68822
04/23/2022 05:53:12 - INFO - Blink -   Macro准确率: 0.69829
04/23/2022 05:53:12 - INFO - Blink -   Micro准确率: 0.67841
04/23/2022 06:02:26 - INFO - Blink -   Macro准确率: 0.70604
04/23/2022 06:02:26 - INFO - Blink -   Micro准确率: 0.68447
04/23/2022 06:11:40 - INFO - Blink -   Macro准确率: 0.70641
04/23/2022 06:11:40 - INFO - Blink -   Micro准确率: 0.68931
04/23/2022 06:20:54 - INFO - Blink -   Macro准确率: 0.71739
04/23/2022 06:20:54 - INFO - Blink -   Micro准确率: 0.69476
04/23/2022 06:30:07 - INFO - Blink -   Macro准确率: 0.71136
04/23/2022 06:30:07 - INFO - Blink -   Micro准确率: 0.68301
04/23/2022 06:39:21 - INFO - Blink -   Macro准确率: 0.71423
04/23/2022 06:39:21 - INFO - Blink -   Micro准确率: 0.69028
04/23/2022 06:48:35 - INFO - Blink -   Macro准确率: 0.71151
04/23/2022 06:48:35 - INFO - Blink -   Micro准确率: 0.69161
04/23/2022 06:57:49 - INFO - Blink -   Macro准确率: 0.71378
04/23/2022 06:57:49 - INFO - Blink -   Micro准确率: 0.69439
04/23/2022 07:07:02 - INFO - Blink -   Macro准确率: 0.71521
04/23/2022 07:07:02 - INFO - Blink -   Micro准确率: 0.69415
04/23/2022 07:16:16 - INFO - Blink -   Macro准确率: 0.70282
04/23/2022 07:16:16 - INFO - Blink -   Micro准确率: 0.68410
04/23/2022 07:25:30 - INFO - Blink -   Macro准确率: 0.71468
04/23/2022 07:25:30 - INFO - Blink -   Micro准确率: 0.69439
04/23/2022 07:34:44 - INFO - Blink -   Macro准确率: 0.71765
04/23/2022 07:34:44 - INFO - Blink -   Micro准确率: 0.69633
04/23/2022 07:43:58 - INFO - Blink -   Macro准确率: 0.71770
04/23/2022 07:43:58 - INFO - Blink -   Micro准确率: 0.70081
04/23/2022 07:53:12 - INFO - Blink -   Macro准确率: 0.71439
04/23/2022 07:53:12 - INFO - Blink -   Micro准确率: 0.69197
04/23/2022 08:02:26 - INFO - Blink -   Macro准确率: 0.71171
04/23/2022 08:02:26 - INFO - Blink -   Micro准确率: 0.69076
04/23/2022 08:11:40 - INFO - Blink -   Macro准确率: 0.71647
04/23/2022 08:11:40 - INFO - Blink -   Micro准确率: 0.69621
04/23/2022 08:20:55 - INFO - Blink -   Macro准确率: 0.72098
04/23/2022 08:20:55 - INFO - Blink -   Micro准确率: 0.69863
04/23/2022 08:30:09 - INFO - Blink -   Macro准确率: 0.72898
04/23/2022 08:30:09 - INFO - Blink -   Micro准确率: 0.71050
04/23/2022 08:39:24 - INFO - Blink -   Macro准确率: 0.72719
04/23/2022 08:39:24 - INFO - Blink -   Micro准确率: 0.70759
04/23/2022 08:48:38 - INFO - Blink -   Macro准确率: 0.72986
04/23/2022 08:48:38 - INFO - Blink -   Micro准确率: 0.70493
04/23/2022 08:57:53 - INFO - Blink -   Macro准确率: 0.72768
04/23/2022 08:57:53 - INFO - Blink -   Micro准确率: 0.70505
04/23/2022 09:07:07 - INFO - Blink -   Macro准确率: 0.72988
04/23/2022 09:07:07 - INFO - Blink -   Micro准确率: 0.70880
04/23/2022 09:16:22 - INFO - Blink -   Macro准确率: 0.72977
04/23/2022 09:16:22 - INFO - Blink -   Micro准确率: 0.70711
04/23/2022 09:25:37 - INFO - Blink -   Macro准确率: 0.71860
04/23/2022 09:25:37 - INFO - Blink -   Micro准确率: 0.69900
04/23/2022 09:34:52 - INFO - Blink -   Macro准确率: 0.71622
04/23/2022 09:34:52 - INFO - Blink -   Micro准确率: 0.70178
04/23/2022 09:44:07 - INFO - Blink -   Macro准确率: 0.72820
04/23/2022 09:44:07 - INFO - Blink -   Micro准确率: 0.70892
04/23/2022 09:53:22 - INFO - Blink -   Macro准确率: 0.71741
04/23/2022 09:53:22 - INFO - Blink -   Micro准确率: 0.70105
04/23/2022 10:02:37 - INFO - Blink -   Macro准确率: 0.72691
04/23/2022 10:02:37 - INFO - Blink -   Micro准确率: 0.70771
04/23/2022 10:11:53 - INFO - Blink -   Macro准确率: 0.71869
04/23/2022 10:11:53 - INFO - Blink -   Micro准确率: 0.70045
04/23/2022 10:21:09 - INFO - Blink -   Macro准确率: 0.72352
04/23/2022 10:21:09 - INFO - Blink -   Micro准确率: 0.70456
04/23/2022 10:30:25 - INFO - Blink -   Macro准确率: 0.71980
04/23/2022 10:30:25 - INFO - Blink -   Micro准确率: 0.70602
04/23/2022 10:39:41 - INFO - Blink -   Macro准确率: 0.72884
04/23/2022 10:39:41 - INFO - Blink -   Micro准确率: 0.71147
04/23/2022 10:48:58 - INFO - Blink -   Macro准确率: 0.71962
04/23/2022 10:48:58 - INFO - Blink -   Micro准确率: 0.70565
04/23/2022 10:58:15 - INFO - Blink -   Macro准确率: 0.70883
04/23/2022 10:58:15 - INFO - Blink -   Micro准确率: 0.69730
04/23/2022 11:07:32 - INFO - Blink -   Macro准确率: 0.71895
04/23/2022 11:07:32 - INFO - Blink -   Micro准确率: 0.70130
04/23/2022 11:16:49 - INFO - Blink -   Macro准确率: 0.71174
04/23/2022 11:16:49 - INFO - Blink -   Micro准确率: 0.69694
04/23/2022 11:26:07 - INFO - Blink -   Macro准确率: 0.72170
04/23/2022 11:26:07 - INFO - Blink -   Micro准确率: 0.70130
04/23/2022 11:35:24 - INFO - Blink -   Macro准确率: 0.71880
04/23/2022 11:35:24 - INFO - Blink -   Micro准确率: 0.70323
04/23/2022 11:44:42 - INFO - Blink -   Macro准确率: 0.73378
04/23/2022 11:44:42 - INFO - Blink -   Micro准确率: 0.71510
04/23/2022 11:54:00 - INFO - Blink -   Macro准确率: 0.73605
04/23/2022 11:54:00 - INFO - Blink -   Micro准确率: 0.71340
04/23/2022 12:03:18 - INFO - Blink -   Macro准确率: 0.72089
04/23/2022 12:03:18 - INFO - Blink -   Micro准确率: 0.70033
04/23/2022 12:12:36 - INFO - Blink -   Macro准确率: 0.72899
04/23/2022 12:12:36 - INFO - Blink -   Micro准确率: 0.71086
04/23/2022 12:21:54 - INFO - Blink -   Macro准确率: 0.72711
04/23/2022 12:21:54 - INFO - Blink -   Micro准确率: 0.71074
04/23/2022 12:31:12 - INFO - Blink -   Macro准确率: 0.73209
04/23/2022 12:31:12 - INFO - Blink -   Micro准确率: 0.71159
04/23/2022 12:40:31 - INFO - Blink -   Macro准确率: 0.72334
04/23/2022 12:40:31 - INFO - Blink -   Micro准确率: 0.70662
04/23/2022 12:49:50 - INFO - Blink -   Macro准确率: 0.73747
04/23/2022 12:49:50 - INFO - Blink -   Micro准确率: 0.71377
04/23/2022 12:59:09 - INFO - Blink -   Macro准确率: 0.73255
04/23/2022 12:59:09 - INFO - Blink -   Micro准确率: 0.71207
04/23/2022 13:08:29 - INFO - Blink -   Macro准确率: 0.73392
04/23/2022 13:08:29 - INFO - Blink -   Micro准确率: 0.71243
04/23/2022 13:17:48 - INFO - Blink -   Macro准确率: 0.73820
04/23/2022 13:17:48 - INFO - Blink -   Micro准确率: 0.71461
04/23/2022 13:27:08 - INFO - Blink -   Macro准确率: 0.73628
04/23/2022 13:27:08 - INFO - Blink -   Micro准确率: 0.71038
04/23/2022 13:36:29 - INFO - Blink -   Macro准确率: 0.72635
04/23/2022 13:36:29 - INFO - Blink -   Micro准确率: 0.70033
04/23/2022 13:45:50 - INFO - Blink -   Macro准确率: 0.72200
04/23/2022 13:45:50 - INFO - Blink -   Micro准确率: 0.69585
04/23/2022 13:55:11 - INFO - Blink -   Macro准确率: 0.73600
04/23/2022 13:55:11 - INFO - Blink -   Micro准确率: 0.71268
04/23/2022 14:04:32 - INFO - Blink -   Macro准确率: 0.72467
04/23/2022 14:04:32 - INFO - Blink -   Micro准确率: 0.70226
04/23/2022 14:13:57 - INFO - Blink -   Macro准确率: 0.72456
04/23/2022 14:13:57 - INFO - Blink -   Micro准确率: 0.70408
04/23/2022 14:23:26 - INFO - Blink -   Macro准确率: 0.72413
04/23/2022 14:23:26 - INFO - Blink -   Micro准确率: 0.69996
04/23/2022 14:32:57 - INFO - Blink -   Macro准确率: 0.73356
04/23/2022 14:32:57 - INFO - Blink -   Micro准确率: 0.71219
04/23/2022 14:42:30 - INFO - Blink -   Macro准确率: 0.71730
04/23/2022 14:42:30 - INFO - Blink -   Micro准确率: 0.69500
04/23/2022 14:52:03 - INFO - Blink -   Macro准确率: 0.72993
04/23/2022 14:52:03 - INFO - Blink -   Micro准确率: 0.70396
04/23/2022 15:01:37 - INFO - Blink -   Macro准确率: 0.72358
04/23/2022 15:01:37 - INFO - Blink -   Micro准确率: 0.70214
04/23/2022 15:11:11 - INFO - Blink -   Macro准确率: 0.73603
04/23/2022 15:11:11 - INFO - Blink -   Micro准确率: 0.71498
04/23/2022 15:20:45 - INFO - Blink -   Macro准确率: 0.73344
04/23/2022 15:20:45 - INFO - Blink -   Micro准确率: 0.70844
04/23/2022 15:30:20 - INFO - Blink -   Macro准确率: 0.72910
04/23/2022 15:30:20 - INFO - Blink -   Micro准确率: 0.70771
04/23/2022 15:39:55 - INFO - Blink -   Macro准确率: 0.72672
04/23/2022 15:39:55 - INFO - Blink -   Micro准确率: 0.70662
04/23/2022 15:49:31 - INFO - Blink -   Macro准确率: 0.72696
04/23/2022 15:49:31 - INFO - Blink -   Micro准确率: 0.70469
04/23/2022 15:59:07 - INFO - Blink -   Macro准确率: 0.71498
04/23/2022 15:59:07 - INFO - Blink -   Micro准确率: 0.69851
04/23/2022 16:08:43 - INFO - Blink -   Macro准确率: 0.72533
04/23/2022 16:08:43 - INFO - Blink -   Micro准确率: 0.70372
04/23/2022 16:18:18 - INFO - Blink -   Macro准确率: 0.72642
04/23/2022 16:18:18 - INFO - Blink -   Micro准确率: 0.70650
04/23/2022 16:27:54 - INFO - Blink -   Macro准确率: 0.72955
04/23/2022 16:27:54 - INFO - Blink -   Micro准确率: 0.70977
04/23/2022 16:37:29 - INFO - Blink -   Macro准确率: 0.72678
04/23/2022 16:37:29 - INFO - Blink -   Micro准确率: 0.70929
04/23/2022 16:47:05 - INFO - Blink -   Macro准确率: 0.72875
04/23/2022 16:47:05 - INFO - Blink -   Micro准确率: 0.70941
04/23/2022 16:56:42 - INFO - Blink -   Macro准确率: 0.73603
04/23/2022 16:56:42 - INFO - Blink -   Micro准确率: 0.71486
04/23/2022 17:06:18 - INFO - Blink -   Macro准确率: 0.72776
04/23/2022 17:06:18 - INFO - Blink -   Micro准确率: 0.70844
04/23/2022 17:15:55 - INFO - Blink -   Macro准确率: 0.72776
04/23/2022 17:15:55 - INFO - Blink -   Micro准确率: 0.70311
04/23/2022 17:25:31 - INFO - Blink -   Macro准确率: 0.72838
04/23/2022 17:25:31 - INFO - Blink -   Micro准确率: 0.71038
04/23/2022 17:35:07 - INFO - Blink -   Macro准确率: 0.72447
04/23/2022 17:35:07 - INFO - Blink -   Micro准确率: 0.70856
04/23/2022 17:44:43 - INFO - Blink -   Macro准确率: 0.72705
04/23/2022 17:44:43 - INFO - Blink -   Micro准确率: 0.71001
04/23/2022 17:54:18 - INFO - Blink -   Macro准确率: 0.72922
04/23/2022 17:54:18 - INFO - Blink -   Micro准确率: 0.71183
04/23/2022 18:03:54 - INFO - Blink -   Macro准确率: 0.73259
04/23/2022 18:03:54 - INFO - Blink -   Micro准确率: 0.71280
04/23/2022 18:13:28 - INFO - Blink -   Macro准确率: 0.73058
04/23/2022 18:13:28 - INFO - Blink -   Micro准确率: 0.71062
04/23/2022 18:23:03 - INFO - Blink -   Macro准确率: 0.73341
04/23/2022 18:23:03 - INFO - Blink -   Micro准确率: 0.71183
04/23/2022 18:32:38 - INFO - Blink -   Macro准确率: 0.73746
04/23/2022 18:32:38 - INFO - Blink -   Micro准确率: 0.71449
04/23/2022 18:42:12 - INFO - Blink -   Macro准确率: 0.72853
04/23/2022 18:42:12 - INFO - Blink -   Micro准确率: 0.70929
04/23/2022 18:51:46 - INFO - Blink -   Macro准确率: 0.73685
04/23/2022 18:51:46 - INFO - Blink -   Micro准确率: 0.71570
04/23/2022 19:01:20 - INFO - Blink -   Macro准确率: 0.72726
04/23/2022 19:01:20 - INFO - Blink -   Micro准确率: 0.70674
04/23/2022 19:10:51 - INFO - Blink -   Macro准确率: 0.73698
04/23/2022 19:10:51 - INFO - Blink -   Micro准确率: 0.71437
04/23/2022 19:19:58 - INFO - Blink -   Macro准确率: 0.73760
04/23/2022 19:19:58 - INFO - Blink -   Micro准确率: 0.71510
04/23/2022 19:29:22 - INFO - Blink -   Macro准确率: 0.73657
04/23/2022 19:29:22 - INFO - Blink -   Micro准确率: 0.71534
04/23/2022 19:38:47 - INFO - Blink -   Macro准确率: 0.73306
04/23/2022 19:38:47 - INFO - Blink -   Micro准确率: 0.71050
04/23/2022 19:48:10 - INFO - Blink -   Macro准确率: 0.71589
04/23/2022 19:48:10 - INFO - Blink -   Micro准确率: 0.70154
04/23/2022 19:57:34 - INFO - Blink -   Macro准确率: 0.73059
04/23/2022 19:57:34 - INFO - Blink -   Micro准确率: 0.71110
04/23/2022 20:06:56 - INFO - Blink -   Macro准确率: 0.73040
04/23/2022 20:06:56 - INFO - Blink -   Micro准确率: 0.71110
04/23/2022 20:16:19 - INFO - Blink -   Macro准确率: 0.72913
04/23/2022 20:16:19 - INFO - Blink -   Micro准确率: 0.70795
04/23/2022 20:25:41 - INFO - Blink -   Macro准确率: 0.72087
04/23/2022 20:25:41 - INFO - Blink -   Micro准确率: 0.70735
04/23/2022 20:35:02 - INFO - Blink -   Macro准确率: 0.72818
04/23/2022 20:35:02 - INFO - Blink -   Micro准确率: 0.71135
04/23/2022 20:44:24 - INFO - Blink -   Macro准确率: 0.72294
04/23/2022 20:44:24 - INFO - Blink -   Micro准确率: 0.70662
04/23/2022 20:53:46 - INFO - Blink -   Macro准确率: 0.73401
04/23/2022 20:53:46 - INFO - Blink -   Micro准确率: 0.71655
04/23/2022 21:03:08 - INFO - Blink -   Macro准确率: 0.73413
04/23/2022 21:03:08 - INFO - Blink -   Micro准确率: 0.71401
04/23/2022 21:12:29 - INFO - Blink -   Macro准确率: 0.73169
04/23/2022 21:12:29 - INFO - Blink -   Micro准确率: 0.71026
04/23/2022 21:21:51 - INFO - Blink -   Macro准确率: 0.73172
04/23/2022 21:21:51 - INFO - Blink -   Micro准确率: 0.70977
04/23/2022 21:31:12 - INFO - Blink -   Macro准确率: 0.72753
04/23/2022 21:31:12 - INFO - Blink -   Micro准确率: 0.70856
04/23/2022 21:40:33 - INFO - Blink -   Macro准确率: 0.72061
04/23/2022 21:40:33 - INFO - Blink -   Micro准确率: 0.70093
04/23/2022 21:49:54 - INFO - Blink -   Macro准确率: 0.72221
04/23/2022 21:49:54 - INFO - Blink -   Micro准确率: 0.70420
04/23/2022 21:59:15 - INFO - Blink -   Macro准确率: 0.72306
04/23/2022 21:59:15 - INFO - Blink -   Micro准确率: 0.70553
04/23/2022 22:08:36 - INFO - Blink -   Macro准确率: 0.73339
04/23/2022 22:08:36 - INFO - Blink -   Micro准确率: 0.71159
04/23/2022 22:17:57 - INFO - Blink -   Macro准确率: 0.73201
04/23/2022 22:17:57 - INFO - Blink -   Micro准确率: 0.70941
04/23/2022 22:27:18 - INFO - Blink -   Macro准确率: 0.73308
04/23/2022 22:27:18 - INFO - Blink -   Micro准确率: 0.71171
04/23/2022 22:36:39 - INFO - Blink -   Macro准确率: 0.73586
04/23/2022 22:36:39 - INFO - Blink -   Micro准确率: 0.71219
04/23/2022 22:46:00 - INFO - Blink -   Macro准确率: 0.72767
04/23/2022 22:46:00 - INFO - Blink -   Micro准确率: 0.70771
04/23/2022 22:55:20 - INFO - Blink -   Macro准确率: 0.72731
04/23/2022 22:55:20 - INFO - Blink -   Micro准确率: 0.70832
04/23/2022 23:04:41 - INFO - Blink -   Macro准确率: 0.73147
04/23/2022 23:04:41 - INFO - Blink -   Micro准确率: 0.71026
04/23/2022 23:14:01 - INFO - Blink -   Macro准确率: 0.71458
04/23/2022 23:14:01 - INFO - Blink -   Micro准确率: 0.69730
04/23/2022 23:23:22 - INFO - Blink -   Macro准确率: 0.71430
04/23/2022 23:23:22 - INFO - Blink -   Micro准确率: 0.69815
04/23/2022 23:32:42 - INFO - Blink -   Macro准确率: 0.72737
04/23/2022 23:32:42 - INFO - Blink -   Micro准确率: 0.70832
04/23/2022 23:42:02 - INFO - Blink -   Macro准确率: 0.72813
04/23/2022 23:42:02 - INFO - Blink -   Micro准确率: 0.71086
04/23/2022 23:51:22 - INFO - Blink -   Macro准确率: 0.73247
04/23/2022 23:51:22 - INFO - Blink -   Micro准确率: 0.71365
04/24/2022 00:00:42 - INFO - Blink -   Macro准确率: 0.73154
04/24/2022 00:00:42 - INFO - Blink -   Micro准确率: 0.71171
04/24/2022 00:10:03 - INFO - Blink -   Macro准确率: 0.72928
04/24/2022 00:10:03 - INFO - Blink -   Micro准确率: 0.70832
04/24/2022 00:19:23 - INFO - Blink -   Macro准确率: 0.72658
04/24/2022 00:19:23 - INFO - Blink -   Micro准确率: 0.70142
04/24/2022 00:28:43 - INFO - Blink -   Macro准确率: 0.73493
04/24/2022 00:28:43 - INFO - Blink -   Micro准确率: 0.71365
04/24/2022 00:38:03 - INFO - Blink -   Macro准确率: 0.73189
04/24/2022 00:38:03 - INFO - Blink -   Micro准确率: 0.70771
04/24/2022 00:47:24 - INFO - Blink -   Macro准确率: 0.73647
04/24/2022 00:47:24 - INFO - Blink -   Micro准确率: 0.71292
04/24/2022 00:56:44 - INFO - Blink -   Macro准确率: 0.72669
04/24/2022 00:56:44 - INFO - Blink -   Micro准确率: 0.70662
04/24/2022 01:06:04 - INFO - Blink -   Macro准确率: 0.73464
04/24/2022 01:06:04 - INFO - Blink -   Micro准确率: 0.71110
04/24/2022 01:15:24 - INFO - Blink -   Macro准确率: 0.73394
04/24/2022 01:15:24 - INFO - Blink -   Micro准确率: 0.71110
04/24/2022 01:24:43 - INFO - Blink -   Macro准确率: 0.72676
04/24/2022 01:24:43 - INFO - Blink -   Micro准确率: 0.70130
04/24/2022 01:34:02 - INFO - Blink -   Macro准确率: 0.73429
04/24/2022 01:34:02 - INFO - Blink -   Micro准确率: 0.71328
04/24/2022 01:43:22 - INFO - Blink -   Macro准确率: 0.73008
04/24/2022 01:43:22 - INFO - Blink -   Micro准确率: 0.70880
04/24/2022 01:52:41 - INFO - Blink -   Macro准确率: 0.73293
04/24/2022 01:52:41 - INFO - Blink -   Micro准确率: 0.71110
04/24/2022 02:02:01 - INFO - Blink -   Macro准确率: 0.73476
04/24/2022 02:02:01 - INFO - Blink -   Micro准确率: 0.71268
04/24/2022 02:11:20 - INFO - Blink -   Macro准确率: 0.73537
04/24/2022 02:11:20 - INFO - Blink -   Micro准确率: 0.71243
04/24/2022 02:20:39 - INFO - Blink -   Macro准确率: 0.72916
04/24/2022 02:20:39 - INFO - Blink -   Micro准确率: 0.71013
04/24/2022 02:29:58 - INFO - Blink -   Macro准确率: 0.72500
04/24/2022 02:29:58 - INFO - Blink -   Micro准确率: 0.70687
04/24/2022 02:39:17 - INFO - Blink -   Macro准确率: 0.72958
04/24/2022 02:39:17 - INFO - Blink -   Micro准确率: 0.71050
04/24/2022 02:48:36 - INFO - Blink -   Macro准确率: 0.72985
04/24/2022 02:48:36 - INFO - Blink -   Micro准确率: 0.70844
04/24/2022 02:57:55 - INFO - Blink -   Macro准确率: 0.70813
04/24/2022 02:57:55 - INFO - Blink -   Micro准确率: 0.68919
04/24/2022 03:07:14 - INFO - Blink -   Macro准确率: 0.72323
04/24/2022 03:07:14 - INFO - Blink -   Micro准确率: 0.70033
04/24/2022 03:16:33 - INFO - Blink -   Macro准确率: 0.72774
04/24/2022 03:16:33 - INFO - Blink -   Micro准确率: 0.70638
04/24/2022 03:25:52 - INFO - Blink -   Macro准确率: 0.73208
04/24/2022 03:25:52 - INFO - Blink -   Micro准确率: 0.71147
04/24/2022 03:35:11 - INFO - Blink -   Macro准确率: 0.74068
04/24/2022 03:35:11 - INFO - Blink -   Micro准确率: 0.71946
04/24/2022 03:44:30 - INFO - Blink -   Macro准确率: 0.72901
04/24/2022 03:44:30 - INFO - Blink -   Micro准确率: 0.70747
04/24/2022 03:53:49 - INFO - Blink -   Macro准确率: 0.74119
04/24/2022 03:53:49 - INFO - Blink -   Micro准确率: 0.71958
04/24/2022 04:03:08 - INFO - Blink -   Macro准确率: 0.73670
04/24/2022 04:03:08 - INFO - Blink -   Micro准确率: 0.71474
04/24/2022 04:12:26 - INFO - Blink -   Macro准确率: 0.73846
04/24/2022 04:12:26 - INFO - Blink -   Micro准确率: 0.71595
04/24/2022 04:21:45 - INFO - Blink -   Macro准确率: 0.73040
04/24/2022 04:21:45 - INFO - Blink -   Micro准确率: 0.70360
04/24/2022 04:31:03 - INFO - Blink -   Macro准确率: 0.74491
04/24/2022 04:31:03 - INFO - Blink -   Micro准确率: 0.72248
04/24/2022 04:40:22 - INFO - Blink -   Macro准确率: 0.73795
04/24/2022 04:40:22 - INFO - Blink -   Micro准确率: 0.71389
04/24/2022 04:49:40 - INFO - Blink -   Macro准确率: 0.73283
04/24/2022 04:49:40 - INFO - Blink -   Micro准确率: 0.70977
04/24/2022 04:58:59 - INFO - Blink -   Macro准确率: 0.74165
04/24/2022 04:58:59 - INFO - Blink -   Micro准确率: 0.71631
04/24/2022 05:08:17 - INFO - Blink -   Macro准确率: 0.74067
04/24/2022 05:08:17 - INFO - Blink -   Micro准确率: 0.71752
04/24/2022 05:17:35 - INFO - Blink -   Macro准确率: 0.74089
04/24/2022 05:17:35 - INFO - Blink -   Micro准确率: 0.71679
04/24/2022 05:26:53 - INFO - Blink -   Macro准确率: 0.74344
04/24/2022 05:26:53 - INFO - Blink -   Micro准确率: 0.72091
04/24/2022 05:36:11 - INFO - Blink -   Macro准确率: 0.74394
04/24/2022 05:36:11 - INFO - Blink -   Micro准确率: 0.72285
04/24/2022 05:45:29 - INFO - Blink -   Macro准确率: 0.73590
04/24/2022 05:45:29 - INFO - Blink -   Micro准确率: 0.71558
04/24/2022 05:54:47 - INFO - Blink -   Macro准确率: 0.73003
04/24/2022 05:54:47 - INFO - Blink -   Micro准确率: 0.70868
04/24/2022 06:04:05 - INFO - Blink -   Macro准确率: 0.74088
04/24/2022 06:04:05 - INFO - Blink -   Micro准确率: 0.71752
04/24/2022 06:13:23 - INFO - Blink -   Macro准确率: 0.73851
04/24/2022 06:13:23 - INFO - Blink -   Micro准确率: 0.71328
04/24/2022 06:22:41 - INFO - Blink -   Macro准确率: 0.74385
04/24/2022 06:22:41 - INFO - Blink -   Micro准确率: 0.71861
04/24/2022 06:31:59 - INFO - Blink -   Macro准确率: 0.73852
04/24/2022 06:31:59 - INFO - Blink -   Micro准确率: 0.71243
04/24/2022 06:41:17 - INFO - Blink -   Macro准确率: 0.73859
04/24/2022 06:41:17 - INFO - Blink -   Micro准确率: 0.71583
04/24/2022 06:50:35 - INFO - Blink -   Macro准确率: 0.72743
04/24/2022 06:50:35 - INFO - Blink -   Micro准确率: 0.70275
04/24/2022 06:59:53 - INFO - Blink -   Macro准确率: 0.72127
04/24/2022 06:59:53 - INFO - Blink -   Micro准确率: 0.70335
04/24/2022 07:09:11 - INFO - Blink -   Macro准确率: 0.73297
04/24/2022 07:09:11 - INFO - Blink -   Micro准确率: 0.71183
04/24/2022 07:18:29 - INFO - Blink -   Macro准确率: 0.72911
04/24/2022 07:18:29 - INFO - Blink -   Micro准确率: 0.70771
04/24/2022 07:27:47 - INFO - Blink -   Macro准确率: 0.73303
04/24/2022 07:27:47 - INFO - Blink -   Micro准确率: 0.71098
04/24/2022 07:37:05 - INFO - Blink -   Macro准确率: 0.73547
04/24/2022 07:37:05 - INFO - Blink -   Micro准确率: 0.71522
04/24/2022 07:46:23 - INFO - Blink -   Macro准确率: 0.73397
04/24/2022 07:46:23 - INFO - Blink -   Micro准确率: 0.71389
04/24/2022 07:55:41 - INFO - Blink -   Macro准确率: 0.73591
04/24/2022 07:55:41 - INFO - Blink -   Micro准确率: 0.71352
04/24/2022 08:04:59 - INFO - Blink -   Macro准确率: 0.74343
04/24/2022 08:04:59 - INFO - Blink -   Micro准确率: 0.72006
04/24/2022 08:14:18 - INFO - Blink -   Macro准确率: 0.74228
04/24/2022 08:14:18 - INFO - Blink -   Micro准确率: 0.71897
04/24/2022 08:23:36 - INFO - Blink -   Macro准确率: 0.73562
04/24/2022 08:23:36 - INFO - Blink -   Micro准确率: 0.71365
04/24/2022 08:32:55 - INFO - Blink -   Macro准确率: 0.73284
04/24/2022 08:32:55 - INFO - Blink -   Micro准确率: 0.71195
04/24/2022 08:42:14 - INFO - Blink -   Macro准确率: 0.73706
04/24/2022 08:42:14 - INFO - Blink -   Micro准确率: 0.71595
04/24/2022 08:51:32 - INFO - Blink -   Macro准确率: 0.74899
04/24/2022 08:51:32 - INFO - Blink -   Micro准确率: 0.72781
04/24/2022 09:00:49 - INFO - Blink -   Macro准确率: 0.73870
04/24/2022 09:00:49 - INFO - Blink -   Micro准确率: 0.71522
04/24/2022 09:10:05 - INFO - Blink -   Macro准确率: 0.74662
04/24/2022 09:10:05 - INFO - Blink -   Micro准确率: 0.72587
04/24/2022 09:19:22 - INFO - Blink -   Macro准确率: 0.74236
04/24/2022 09:19:22 - INFO - Blink -   Micro准确率: 0.72321
04/24/2022 09:28:37 - INFO - Blink -   Macro准确率: 0.74120
04/24/2022 09:28:37 - INFO - Blink -   Micro准确率: 0.71909
04/24/2022 09:37:52 - INFO - Blink -   Macro准确率: 0.74672
04/24/2022 09:37:52 - INFO - Blink -   Micro准确率: 0.72503
04/24/2022 09:47:08 - INFO - Blink -   Macro准确率: 0.74679
04/24/2022 09:47:08 - INFO - Blink -   Micro准确率: 0.72454
```

```console
CrossEncoderRanker(
  (model): CrossEncoderModule(
    (encoder): BertEncoder(
      (bert_model): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (additional_linear): Linear(in_features=768, out_features=1, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
)
```

# 测试训练后的zeshel数据，不太适用于使用main_dense.py进行推理
