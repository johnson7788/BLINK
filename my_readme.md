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
│   ├── all_entities_large.t7
│   ├── biencoder_wiki_large.bin   #双编码器模型
│   ├── biencoder_wiki_large.json  双编码器模型配置
│   ├── crossencoder_wiki_large.bin
│   ├── crossencoder_wiki_large.json
│   ├── elq_large_params.txt
│   ├── elq_webqsp_large.bin
│   ├── elq_wiki_large.bin
│   ├── entity.jsonl   #实体目录的路径
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