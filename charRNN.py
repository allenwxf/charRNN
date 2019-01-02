# -*-utf8-*-

# 对模型进行调整
#
# 上面提到最重要的模型配置选项是 rnn_size 和 rnn_layers：它们决定网络的复杂度。通常来说，你在教程中看到的网络是由 128 个神经元或 256 个
# 神经元组成的网络。然而，textgenrnn 的架构略有不同，因为它有一个包含了前面所有模型层的注意力层。因此，除非你拥有特别大量的文本（>10MB），
# 让模型更深比让模型更宽要好一些（例如，4x128 比 1x512 的模型要好）。rnn_bidirectional 控制循环神经网络是否是双向的，也就是说，
# 它同时向前和向后处理一个字符（如果文本遵循特定的规则，如莎士比亚的字符标题，这种方法会十分有效）。max_length 决定用于预测下一个字符的网络的
# 最大字符数，当网络需要学习更长的序列时应该增大它，而当网络需要学习更短的序列时则应该减小它。
#
# 在训练过程中也有很多有用的配置选项。num_epochs 决定了完整遍历数据的次数，如果你想对模型进行更多次的训练，你可以调整这个参数。
# batch_size 决定了在一个计算步中训练的模型序列的数量，深度学习模型的批处理大小一般是 32 或 128，但是当你拥有一个 GPU 的时候，
# 你可以通过使用给定的 1024 作为缺省批处理大小来获得训练速度的提升。train_size 决定待训练字符样本的比例，将它设置为< 1.0 可以同时加快
# 每个 epoch 的训练速度，同时防止模型通过一字不差地学习并复制原文来「作弊」（你可以将「validation」设置为 True，在每一个 epoch 后利用
# 未使用的数据运行模型，来看看模型是否过拟合）。

from textgenrnn import textgenrnn
from datetime import datetime
import jieba

model_cfg = {
    'word_level': True,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 64,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 8,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': False,   # consider text both forwards and backward, can give a training boost
    'max_length': 5,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 500,   # maximum number of words to model; the rest will be ignored (word-level model only)
}

train_cfg = {
    'line_delimited': False,   # set to True if each text has its own line in the source file
    'num_epochs': 100,   # set higher to train the model for longer
    'gen_epochs': 10,   # generates sample text from model after given number of epochs
    'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

srcfile_name = "dataset/jayLyrics.txt"
file_name = "dataset/jayLyricsCorpus.txt"

jieba_dict_file = "dataset/jieba-dict/dict.txt"

jieba.load_userdict(jieba_dict_file)

# 原始文本分词
corpusfile = open(file_name, "w", encoding="utf-8")
with open(srcfile_name, "r", encoding="utf-8") as srcfile:
    linenum = 0
    for line in srcfile.readlines():
        linenum += 1
        # if linenum > 20:
        #     break
        line = line.strip()
        # print("raw: ", line)
        cutline = jieba.cut(line, cut_all=False, HMM=True)
        cutline = " ".join(cutline)
        print("cut: ", cutline)
        corpusfile.write(cutline + "\n")
srcfile.close()
corpusfile.close()

model_name = 'reviewGen'   # change to set file name of resulting trained models/texts

textgen = textgenrnn(name=model_name)

train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

train_function(
    file_path=file_name,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=512,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level'])


# this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
# changing the temperature schedule can result in wildly different output!
temperature = [1.0, 0.5, 0.2, 0.1]
prefix = None  # if you want each generated text to start with a given seed text

if train_cfg['line_delimited']:
    n = 1000
    max_gen_length = 300 if model_cfg['word_level'] else 600
else:
    n = 1
    max_gen_length = 2000 if model_cfg['word_level'] else 10000

timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)

textgen.generate_to_file(gen_file,
                         temperature=temperature,
                         prefix=prefix,
                         n=n,
                         max_gen_length=max_gen_length)