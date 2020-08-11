import torch
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  #为GPU设置随机种子
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.deterministic = True

# 首先，我们要创建两个Field 对象：这两个对象包含了我们打算如何预处理文本数据的信息。
# spaCy:英语分词器,类似于NLTK库，如果没有传递tokenize参数，则默认只是在空格上拆分字符串。
# torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
TEXT = data.Field(tokenize='spacy')
#LabelField是Field类的一个特殊子集，专门用于处理标签。 
LABEL = data.LabelField(dtype=torch.float)

# 加载IMDB电影评论数据集
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


print(1)
print(vars(train_data.examples[0]))

