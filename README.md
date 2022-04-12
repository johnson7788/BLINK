![BLINK logo](./img/blink_logo_banner.png)
--------------------------------------------------------------------------------
BLINK是一个实体链接的python库，它使用维基百科作为目标知识库。

将实体链接到维基百科的过程也被称为 [Wikification](https://en.wikipedia.org/wiki/Wikification).


### news
- (September 2020) added [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) - 端到端的实体链接问题
- (3 July 2020) added [FAISS](https://github.com/facebookresearch/faiss) support in BLINK - 高效的精确/近似检索
- FAISS 论文：https://arxiv.org/abs/1702.08734

## BLINK 架构
BLINK架构将在下面的文件中描述：

```bibtex
@inproceedings{wu2019zero,
 title={Zero-shot Entity Linking with Dense Entity Retrieval},
 author={Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, Luke Zettlemoyer},
 booktitle={EMNLP},
 year={2020}
}
```

[https://arxiv.org/pdf/1911.03814.pdf](https://arxiv.org/pdf/1911.03814.pdf)

简而言之，BLINK使用两阶段方法进行实体链接，基于微调的BERT架构。
在第一阶段，BLINK在一个由双编码器定义的密集空间中进行检索，该编码器独立地嵌入了提及背景和实体描述。
然后用交叉编码器对每个候选者进行更仔细的检查，交叉编码器将提及和实体文本拼接起来。BLINK在多个数据集上取得了最先进的结果。

## ELQ 架构
ELQ对问题进行端到端的实体链接。ELQ的架构将在下面的文件中描述。

```bibtex
@inproceedings{li2020efficient,
 title={Efficient One-Pass End-to-End Entity Linking for Questions},
 author={Li, Belinda Z. and Min, Sewon and Iyer, Srinivasan and Mehdad, Yashar and Yih, Wen-tau},
 booktitle={EMNLP},
 year={2020}
}
```

[https://arxiv.org/pdf/2010.02413.pdf](https://arxiv.org/pdf/2010.02413.pdf)

关于如何运行ELQ的更多细节, refer to the [ELQ README](https://github.com/facebookresearch/BLINK/tree/master/elq).



## 使用 BLINK

### 1. 创建conda环境和安装要求

(可选）使用一个单独的conda环境可能是个好主意。它可以通过运行以下程序创建
```
conda create -n blink37 -y python=3.7 && conda activate blink37
pip install -r requirements.txt
```

### 2. 下载BLINK模型

可以用以下脚本下载BLINK预训练的模型。
```console
chmod +x download_blink_models.sh
./download_blink_models.sh
```
我们还在BLINK中提供了一个[FAISS](https://github.com/facebookresearch/faiss)索引器，它可以为双编码器模型进行有效的精确/近似检索。

- [flat index](http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl)
- [hnsw (approximate search) index](http://dl.fbaipublicfiles.com/BLINK/faiss_hnsw_index.pkl)

要自己建立和保存FAISS（精确搜索）索引，请运行
`python blink/build_faiss_index.py --output_path models/faiss_flat_index.pkl`


### 3. 以互动方式使用BLINK
探索BLINK链接能力的一个快速方法是通过`main_dense`互动脚本。
BLINK使用[Flair](https://github.com/flairNLP/flair)进行命名实体识别（NER），从输入文本中获得实体提及，然后运行实体链接。

```console
python blink/main_dense.py -i
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
快速模式：在快速模式下，模型只使用双编码器，速度快得多（精度略有下降，详见 "基准测试BLINK "部分）。

```console
python blink/main_dense.py -i --fast
```
要用保存的FAISS索引运行BLINK，请运行。
```console
python blink/main_dense.py --faiss_index flat --index_path models/faiss_flat_index.pkl
```
or 
```console
python blink/main_dense.py --faiss_index hnsw --index_path models/faiss_hnsw_index.pkl
```


示例: 
```console
Bert and Ernie are two Muppets who appear together in numerous skits on the popular children's television show of the United States, Sesame Street.
```
Output:
<img align="middle" src="img/example_result_light.png" height="480">

注意：通过``--show_url``参数将显示每个实体的维基百科网址。
显示的ID号与从``./download_models.sh``下载的``entity.jsonl``文件中的实体顺序一致（从0开始）。
``entity.jsonl``文件包含每行一个实体的信息（包括维基百科网址、标题、文本等）。

### 4. 在你的代码中使用BLINK 

```console
pip install -e git+git@github.com:facebookresearch/BLINK#egg=BLINK
```

```python
import blink.main_dense as main_dense
import argparse

models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Shakespeare's account of the Roman general".lower(),
                    "mention": "Julius Caesar".lower(),
                    "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
                }
                ]

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

```

## Benchmarking BLINK

我们提供了一些脚本，以便根据流行的实体链接数据集对BLINK进行基准测试。
请注意，我们的脚本是在完整的维基百科环境下评估BLINK的，也就是说，BLINK的实体库包含所有的维基百科页面。

要对BLINK进行基准测试，请运行以下命令。

```console
./scripts/get_train_and_benchmark_data.sh
python scripts/create_BLINK_benchmark_data.py
python blink/run_benchmark.py
```

下表总结了BLINK对所考虑的数据集的性能。

| dataset | biencoder accuracy (fast mode) | biencoder recall@10 | biencoder recall@30 | biencoder recall@100 | crossencoder normalized accuracy | overall unnormalized accuracy | support |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- |  ------------- |
| AIDA-YAGO2 testa | 0.8145 | 0.9425 | 0.9639 | 0.9826 | 0.8700 | 0.8212 | 4766 |
| AIDA-YAGO2 testb | 0.7951 | 0.9238 | 0.9487 | 0.9663 | 0.8669 | 0.8027 | 4446 |
| ACE 2004 | 0.8443 | 0.9795| 0.9836 | 0.9836 | 0.8870 | 0.8689 | 244 |
| aquaint | 0.8662 | 0.9618| 0.9765| 0.9897 | 0.8889 | 0.8588 | 680 |
| clueweb - WNED-CWEB (CWEB) | 0.6747 | 0.8223 | 0.8609 | 0.8868 | 0.826 | 0.6825 | 10491 |
| msnbc | 0.8428 | 0.9303 | 0.9546 | 0.9676| 0.9031 | 0.8509 | 617 |
| wikipedia - WNED-WIKI (WIKI) | 0.7976 | 0.9347 | 0.9546 | 0.9776| 0.8609 | 0.8067 | 6383 |
| TAC-KBP 2010<sup>1</sup> | 0.8898 | 0.9549 | 0.9706 | 0.9843 | 0.9517 | 0.9087 | 1019 |

<sup>1</sup> Licensed dataset available [here](https://catalog.ldc.upenn.edu/LDC2018T16).


## BLINK知识库
BLINK知识库（实体库）是基于2019/08/01维基百科的导出，其原始格式可从以下网站下载  [http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2)

## 用solr作为IR系统的BLINK
BLINK的第一个版本使用一个 [Apache Solr](https://lucene.apache.org/solr) 基于信息检索的系统与基于BERT的交叉编码器相结合。
这个基于IR的版本现在已经废弃了，因为它的性能已经被目前的BLINK架构所超越。
如果你对旧版本感兴趣，请参考 [this README](blink/candidate_retrieval/README.md).

## Troubleshooting

If the module cannot be found, preface the python command with `PYTHONPATH=.`

## License
BLINK is MIT licensed. See the [LICENSE](LICENSE) file for details.
