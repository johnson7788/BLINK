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
│   ├── entity.jsonl   #数据条数：5903527， 实体目录的路径, 实体的原数据， 一条数据  #{"text": " Anarchism is an anti-authoritarian political philosophy that rejects hierarchies deemed unjust and advocates their replacement with self-managed, self-governed societies based on voluntary, cooperative institutions. These institutions are often described as stateless societies, although several authors have defined them more specifically as distinct institutions based on non-hierarchical or free associations. Anarchism's central disagreement with other ideologies is that it holds the state to be undesirable, unnecessary, and harmful.  Anarchism is usually placed on the far-left of the political spectrum, and much of its economics and legal philosophy reflect anti-authoritarian interpretations of communism, collectivism, syndicalism, mutualism, or participatory economics. As anarchism does not offer a fixed body of doctrine from a single particular worldview, many anarchist types and traditions exist and varieties of anarchy diverge widely. Anarchist schools of thought can differ fundamentally, supporting anything from extreme individualism to complete collectivism. Strains of anarchism have often been divided into the categories of social and individualist anarchism, or similar dual classifications. ", "idx": "https://en.wikipedia.org/wiki?curid=12", "title": "Anarchism", "entity": "Anarchism"}
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

# 一条数据集格式
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

# 实体数量
5903527个，嵌入的维度 torch.Size([5903527, 1024])

# 使用测试集测试文件
python blink/main_dense.py --test_mentions examples/test_mentions.jsonl

# 交互性测试， 加载all_entities_large.t7文件缓慢
python blink/main_dense.py -i

# 快速加载索引, 加载pkl文件缓慢, 64GB内存不够用
python blink/main_dense.py --faiss_index hnsw --index_path models/faiss_hnsw_index.pkl

# Step1: 训练biencoder模型
python blink/biencoder/train_biencoder.py --data_path data/zeshel/blink_format --output_path models/zeshel/biencoder --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 8 --eval_batch_size 8 --bert_model bert-base-uncased --type_optimization all_encoder_layers
耗时3个半小时
```
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

# Step2: 从Biencoder模型中获得训练和测试数据集的前64预测结果。
python blink/biencoder/eval_biencoder.py --path_to_model models/zeshel/biencoder/pytorch_model.bin --data_path data/zeshel/blink_format --output_path models/zeshel --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result --bert_model bert-base-uncased --mode train,valid,test --zeshel True
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

```

# Step3: 训练和评估交叉编码器模型:
python blink/crossencoder/train_cross.py --data_path models/zeshel/top64_candidates/ --output_path models/zeshel/crossencoder --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 --train_batch_size 2 --eval_batch_size 2 --bert_model bert-base-uncased --type_optimization all_encoder_layers --add_linear --zeshel True