# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using BLINK.
import argparse
import importlib
import os
import sys
import datetime


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


class BlinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args: bool ,
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_blink_args=True, add_model_args=False, 
        description='BLINK parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )  # 目录： '/media/wac/backup/john/johnson/BLINK'
        os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        添加通用的BLINK命令，给所有的脚本
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", default=False, help="是否打印进度条, 默认打印进度条"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="是否在调试模式下运行，只有200个样本。",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="是否分布候选者生成过程。",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="是否在有条件时不使用CUDA",
        )
        parser.add_argument("--top_k", default=10, type=int, help="候选者的数量")
        parser.add_argument(
            "--seed", type=int, default=52313, help="用于初始化的随机种子"
        )
        parser.add_argument(
            "--zeshel",
            default=True,
            type=bool,
            help="数据集是否来自zeroshot。",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("模型的参数")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="WordPiece tokenization后的最大总输入序列长度。长于此数的序列将被截断，短于此数的序列将被填充。",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="WordPiece tokenization后的上下文的最大总输入序列长度。长于此数的序列将被截断，短于此数的序列将被填充。",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="候选实体的上下文的最大序列长度， 长于此数的序列将被截断，短于此数的序列将被填充。",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="要加载的模型的完整路径, 即继续训练时的模型",
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="使用哪个模型，列表中选择的Bert预训练模型: bert-base-uncased,bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="使用bert的哪一层的参数,默认使用bert最后一层的输出，没有使用, 不用管了",
        )
        parser.add_argument(
            "--lowercase",
            action="store_false",
            help="是否对输入的文本进行小写。对于无大小写的模型为真，对于有大小写的模型为假。",
        )
        parser.add_argument("--context_key", default="context", type=str, help="数据中的一条数据字典，上下文内容对应的key是")
        parser.add_argument(
            "--out_dim", type=int, default=1, help="双编码器的输出尺寸",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="是否在BERT的基础上增加一个额外的线性投影。",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="训练数据的路径",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="输出目录，生成的输出文件（模型等）将被导出。",
        )


    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("模型的训练参数")
        parser.add_argument(
            "--evaluate", action="store_true", help="训练完成后，是否运行模型评估"
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="写入评估结果的txt文件。",
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="训练的总批次大小。"
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="Adam的初始学习率。",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="训练epoch的数量.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=10, 
            help="打印损失的间隔",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=100,
            help="训练期间的评估间隔",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="保存模型的间隔"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="进行线性学习率预热的训练比例为，例如 0.1 = 10% of training. ",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="梯度累积，在执行反向/更新传递之前，要累积的更新步的数量。",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="在BERT中优化哪种类型的层",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, 
            help="是否对训练数据进行打乱, 打乱是比较好的",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("模型评估参数")
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="用于评估的总批次规模.",
        )
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="评估模式，选择哪个数据集进行评估，Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="是否保存预测结果。",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="编码的批次大小"
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="缓存的候选者pool的路径（候选者的IDtoken）。",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="缓存的候选编码的路径",
        )
