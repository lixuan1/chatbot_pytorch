import torch
import os

# 设备定义，优先使用cuda，否则使用cpu
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("data", "save")
# 预定义的token
PAD_token = 0  # 表示padding
SOS_token = 1  # 句子的开始
EOS_token = 2  # 句子的结束

MAX_LENGTH = 10  # Maximum sentence length to consider

# 配置模型
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 从哪个checkpoint恢复，如果是None，那么从头开始训练。
loadFilename = None
checkpoint_iter = 4000

# 配置训练的超参数和优化器
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# 保存checkpoint目录
directory = os.path.join(model_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
# 定义预测时使用的模型路径
model_checkpoint = os.path.join(directory, '{}_{}.tar'.format(n_iteration, 'checkpoint'))