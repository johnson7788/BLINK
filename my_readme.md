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

# 交互性测试
python blink/main_dense.py -i