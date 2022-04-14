## 训练 BLINK on Zero-shot Entity Linking dataset

下载数据集:
    下载地址： https://drive.google.com/uc?export=download&id=1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO
    ./examples/zeshel/get_zeshel_data.sh
 
把数据集转换成BLINK格式:

    python examples/zeshel/create_BLINK_zeshel_data.py

训练双编码器模型。注意：以下命令需要在8个拥有32G内存的GPU上运行。减少train_batch_size和eval_batch_size以减少GPU/内存资源。

    python blink/biencoder/train_biencoder.py \
      --data_path data/zeshel/blink_format \
      --output_path models/zeshel/biencoder \  
      --learning_rate 1e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 \
      --train_batch_size 128 --eval_batch_size 64 --bert_model bert-large-uncased \
      --type_optimization all_encoder_layers --data_parallel

从Biencoder模型中获得对训练、有效和测试数据集的前64预测结果。

    python blink/biencoder/eval_biencoder.py \
      --path_to_model models/zeshel/biencoder/pytorch_model.bin \
      --data_path data/zeshel/blink_format \
      --output_path models/zeshel \
      --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result \
      --bert_model bert-large-uncased --mode train,valid,test \
      --zeshel True --data_parallel

训练和评估交叉编码器模型:

    python blink/crossencoder/train_cross.py \
      --data_path  models/zeshel/top64_candidates/ \
      --output_path models/zeshel/crossencoder \
      --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 \
      --train_batch_size 2 --eval_batch_size 2 --bert_model bert-large-uncased \
      --type_optimization all_encoder_layers --add_linear --data_parallel \
      --zeshel True
