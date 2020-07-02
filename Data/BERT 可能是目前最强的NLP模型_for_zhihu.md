## BERT 可能是目前最强的NLP模型

### 1.  BERT简介

### 2. transformer

### 3. BERT架构与训练

### 4. BERT model的使用

### 5. BERT demo with pytorch

### 6. 总结





### 1. BERT简介

#### 1.1 简介

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的encoder，于2018年由Google[^1]提出，获得了计算语言学协会(NAACL)北美分会2019年年度会议最佳长篇论文奖[^2]，在11个NLP任务上的表现刷新了记录，是目前在NLP领域最火热，也可能是最好用的模型。 

#### 1.2 BERT特点

关于BERT的特点有非常多，但是我这里想说的是BERT的核心特点。

##### 1.2.1提取特征

众所周知，在今天的CV领域，大家在处理具体的分类任务时，可以用一些训练好的model（如ResNet或更潮的EfficientNet）来充分提取图片的特征（feature），然后利用这些模型提取到的feature接上一个分类器（如一千维到两维的线性分类器），只训练后面分类器的部分，就可以很快完成任务。

在NLP领域，我们也希望有这样的model，来充分提取语言的特征，使其便于拼接一些其他网络结构来完成各种语言文本的分类预测等问题。类似的提取特征的模型还有ELMO，而BERT可能是目前最好用的此类模型。并且一些工作已经证明，在视频序列等cv领域，也有很好的表现。

##### 1.2.2 无监督学习

NLP领域与CV领域的一个不同是，语料丰富，但是标注的语料很少，标注困难，因此如果可以无监督的进行训练，会使得模型更能充分训练。而BERT就是可以利用无需标注的语料来训练。这也是它的性能能如此优秀的原因。

##### 1.2.3. 速度和长序列

RNN因为依赖与上层的输出，使得训练速度慢，同时语料过长，则不能进行很好的训练。而BERT可以并行计算，这使得他的速度大大提高，并且可以处理长序列的问题。

#### 1.3 BERT 用途

常规的NLP任务，如文本分类，文本聚类，语言翻译，问答系统，文本特征提取等，BERT都可以胜任，而且BERT还具有一些新的功能，比如google利用BERT来自动地通过新闻资讯生成一个词条的维基百科。

接下来，就让我们开启BERT学习之旅吧！！！

### 2. Transformer

#### 2.1 简介

想了解BERT，我们首先要熟悉Transformer，也就是那篇鼎鼎有名的文章Attention Is All You Need[^3]。这里需要注意的时，在这篇论文之前已经Attention（注意力机制）已经广泛应用于NLP领域，主要是与RNN结合来处理问题。而这篇论文正如题目所说(你只需要Attention)，完全抛弃了RNN，只使用Attention机制，使得NLP的处理可以并行计算，大大加快了模型的训练和计算速度，并且表现非常好，成为NLP领域的新思路。

#### 2.2 self-attention 架构

![image-20200701225247744](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701225247744.png)

上图（图源自李宏毅老师youtube关于transformer的讲解课件，非常建议去看[^4]）给出了架构的第一层。

我们首先需要确定的是我们需要的时找到序列之间的关系信息，使其具有RNN的特点，同时保证序列的运算可以并行化，这是目的，架构是为了这个目的而服务的。

首先transformer通过三个矩阵$ <img src="https://www.zhihu.com/equation?tex=W^q,W^k,W^v" alt="W^q,W^k,W^v" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=将" alt="将" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=分成" alt="分成" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=q^i,k^i,v^i" alt="q^i,k^i,v^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=三个vector，分别对应query，key，value，其中query用于查找其他" alt="三个vector，分别对应query，key，value，其中query用于查找其他" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1"> $的key，用于使他们彼此产生关系，而value用来表示他们各自含有的信息。

![image-20200701230254634](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701230254634.png)

在得到$ <img src="https://www.zhihu.com/equation?tex=q^i,k^i,v^i" alt="q^i,k^i,v^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=之后，我们将" alt="之后，我们将" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=q^i" alt="q^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=与其他的key点乘，得到" alt="与其他的key点乘，得到" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=\hat{\alpha}_{i,j}" alt="\hat{\alpha}_{i,j}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=随后计算" alt="随后计算" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=b^i=\sum_j\hat{\alpha}_{i,j}*v^j" alt="b^i=\sum_j\hat{\alpha}_{i,j}*v^j" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=,这样我们就可以得到第一层的" alt=",这样我们就可以得到第一层的" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=对应的" alt="对应的" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=b^i" alt="b^i" class="ee_img tr_noresize" eeimg="1"> $ 。

上述过程是可以被平行化的，整体的公式如下：

$ <img src="https://www.zhihu.com/equation?tex=Q = W^qX, K = W^kX, V = W^vX" alt="Q = W^qX, K = W^kX, V = W^vX" class="ee_img tr_noresize" eeimg="1"> $，得到query,key,value，对应图一

 $ <img src="https://www.zhihu.com/equation?tex=B = Q^TK*V/\sqrt{d}" alt="B = Q^TK*V/\sqrt{d}" class="ee_img tr_noresize" eeimg="1"> $,完成self-Attention，得到输出，对应图二

可以看到对应输入，我们就是做了一系列的矩阵乘法，这些乘法是可以并行的，而且是矩阵乘法，我们可以很轻易的用GPU加速。

#### 2.3 Multi-Head Attention

其实理解了2.2的结构，就理解了transformer的架构，但在实做上，我们需要使用更复杂的结构来保证模型可以充分获取特征，而multi-head就是这样的例子。

![image-20200701232412748](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701232412748.png)

如上图，所谓multi-head,其实就是将$ <img src="https://www.zhihu.com/equation?tex=q^i,k^i,v^i" alt="q^i,k^i,v^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=分成多个部分，得到多个" alt="分成多个部分，得到多个" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=b^i" alt="b^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=，最后将" alt="，最后将" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=b^{i,j}" alt="b^{i,j}" class="ee_img tr_noresize" eeimg="1"> $根据所需维数拼接起来。

![image-20200701232728784](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701232728784.png)

有什么用？不同的head可能看到不同的信息，关注于不同的区域。

#### 2.4 positional encoding

如果仔细观察可以发现，上述的的模型虽然关注了序列彼此的信息，但是忽略了位置信息，即第一个字和最后一个字对第二个字的相对位置丢失了，（举个例子，BERT击败了ELMO和ELMO击败了BERT是一样的）之就要用到，位置编码（positional encoding）。

![image-20200701233312172](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701233312172.png)

对每一个$ <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=，加入" alt="，加入" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=e^i" alt="e^i" class="ee_img tr_noresize" eeimg="1"> $（代表位置信息）,进行位置编码。

当然还有另一种做法，就是拼接，就是把代表位置的信息拼接在$ <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1"> $后面。

#### 2.5 transformer 架构

![image-20200701234009433](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200701234009433.png)

上图就是transformer的架构，这是一个seq2seq(序列到序列)的架构，可以用于机器翻译，问答系统等问题。他包括encoder部分和decoder的部分。

其中encoder部分有$ <img src="https://www.zhihu.com/equation?tex=N_x" alt="N_x" class="ee_img tr_noresize" eeimg="1"> $个layer，每个layer由两部分，一部分是我们说的Multi-head Attention，另一部分是一个全连接的网络结构，需要注意的是这里使用到了残差网络，把第一部分的输出和全连接的输出一同输出，保证网络结构可以加深。

在decoder部分，整体结构与之前一样，但是将输出的前面部分和encoder的结果进行了组合，来生成接下来的部分。

#### 2.6 transformer总结

transformer利用self-attention机制，很好的解决了RNN不能并行计算的问题，使得序列模型在保持精度的同时，在速度上有了非常大的提升，同时，由它衍生出的BERT，可能成为NLP领域未来最重要的一种方法。



### 3. BERT架构与训练

#### 3.1 BERT架构

所谓BERT，其实可以理解为就是transformer的encoder层，是一个seq2sqe的model。这个并没有什么太多要说的，所以说BERT的核心不在于他的架构，而在于它无监督的训练方式。

公布的BERT有两个版本，一个Nx为12，即12层，另一个为24层。两者相差并不是太大，后者略好，但是参数较多，model非常大。

![image-20200702000421451](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702000421451.png)



#### 3.2 BERT的训练

##### 3.2.1 Masked LM

![image-20200702000659385](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702000659385.png)

所谓masked LM，其实就是一个盖字猜字的做法，paper里指出，通过盖住15%的部分，output外接分类器来以这个猜测的词来训练。具体训练细节可以看论文。

##### 3.2.2 Next Sentence Prediction (NSP)

![image-20200702001027617](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702001027617.png)

所谓NSP，就是检测两个句子是不是可以拼接在一起。

这里要介绍一下token，除了每个词有自己的token外，BERT有一些单独的Token,比如[CLS]，它代表这个句子的分类，对应的output往往接一个分类器，比如[SEP]是表示两个句子的拼接符，比如[MASK]表示这个词被遮挡。

关于BERT的训练，可能是一件非常耗时的事情，根据google的介绍，可能需要非常长的时间，花费巨大。但是对于下游任务来说，是非常快速便捷的。





#### 3.3 BERT fine-tunning

对于不同的NLP下游任务，只需要在BERT基础上fine-tunning即可，即拼接一些结构。

![image-20200702142858389](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702142858389.png)

上图[^5]展示了四种下游任务，分别单一句子分类，成对句子分类，句子词性标注，问答任务。

特地指出这里的问答任务，是指的给定文献和问题，要求答案在文本之中，然后machine来选择哪个词汇是答案。

上述任务的结构图里已经非常清楚了，就是添加一些线性分类器加上softmax即可。（其实在当前框架下，这些任务也已经被写好了，下面会展示）

### 4. BERT model的使用

#### 4.1 python transformer lib[^6]

transformer lib汇聚了当前最广泛的NLP架构（BERT，GPT-2等），提供了32+种预训练模型，支持100+种语言。他提供的模型架构和与训练模型，使得使用者大大减少了机器学习的难度和时间。并且当前提供支持pytorch和TensorFlow2.0的模型。

#### 4.2 使用教程

transformer的完整教程之后应该会单独出一篇文章，这里我只是间接一下BERT的使用。下述代码有参考leemeng[^7]

##### 4.2.1 引入Token

```python
from transformers import BertTokenizer


PRETRAINED_MODEL_NAME = "bert-base-uncased"  # 预训练模型名字


# 取得此预训练的模型使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 定义文本，转为token对应的id
text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

# 到这里我们ids就可以作为我们模型的input
```

##### 4.2.2 引入模型

![image-20200702150323884](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702150323884.png)

transformer提供了几种模型供我们直接使用，包括猜字模型，句子预测，句子分类，问答系统等模型。

```python
# 引入分类模型
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
model
```

如上我们就可以得到模型的架构，和一个与训练好的模型。之后的训练就与正常的model一样了，当然你也可以自己修改模型架构，来实现你想实现的功能。

接下来我们会用一个具体的例子来详细描述如何用BERT解决一个NLP的问题。

### 5. BERT demo with pytorch

我使用的数据集是在台大交换时的NLP任务数据集，包含13240条推文，任务是对它们进行情感分析。下面代码已经公布在我的个人kaggle上了，单击[链接](https://www.kaggle.com/jiachengyou/bertfortweet)就可以查看了。

![image-20200702151747400](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702151747400.png)

#### 5.1 引入需要的库和data

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
path = '/kaggle/input/ml2019fall-hw5'
trainx_path = os.path.join(path, 'train_x.csv')
trainy_path = os.path.join(path, 'train_y.csv')
testx_path = os.path.join(path, 'test_x.csv')
trainx = pd.read_csv(trainx_path)
trainy = pd.read_csv(trainy_path)
testx = pd.read_csv(testx_path)
trainx['comment'].iloc[0]
trainy.head()
```

#### 5.2 分批导入DataSet进行处理

```python
class MyDataset(Dataset):
    def __init__(self, mode, dataX, dataY, tokenizer):
        self.mode = mode
        self.tokenizer = tokenizer
        self.dataX = dataX
        self.dataY = dataY
        self.len = len(dataX)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            text = self.dataX['comment'].iloc[index]
            ### 处理emoji，去掉@
            text = emoji.demojize(text)
            text = text.replace('@user', ' ')
#             print(text)
            item = self.dataY['label'].iloc[index]
            tokens = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return (ids, item)
        elif self.mode == 'test':
            text = self.dataX['comment'].iloc[index]
            item = None
            tokens = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return (torch.tensor(ids), item)
    def __len__(self):
        return self.len
# 前10000笔为traindata，后面三千多笔为validationData
trainData = MyDataset('train', trainx[:10000], trainy[:10000], tokenizer)
valData = MyDataset('train', trainx[10000:], trainy[10000:], tokenizer)
# testData
testData = MyDataset('test', testx, None, tokenizer)
```

#### 5.3 DataLoader

```python
# 选择第一个值为标准进行padding
def create_mini_batch(samples):
    tokens_tensors = [torch.tensor(s[0]) for s in samples]
    labels_tensors = [torch.tensor(s[1]) for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)    
    return tokens_tensors,torch.tensor(labels_tensors)
def create_mini_batch2(samples):
    tokens_tensors = [torch.tensor(s[0]) for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)    
    return tokens_tensors

#设置batch—size
BATCH_SIZE = 32
trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

# valDataLoader = DataLoader(trainData[10000:], batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

testDataLoader = DataLoader(testData, batch_size=256, collate_fn=create_mini_batch2)
```

#### 5.4 引入模型 设置评估函数

```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
# 因为原来data的分布不均，这里使用f1分数而不是acc作为评价标准。
# 关于f1，可以自己去搜索，代码如下
def f1(model, dataloader):
    predictions = None
    correct = 0
    total = 0
    res = np.zeros((2,2))
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            

            tokens_tensors = data[0]
            outputs = model(input_ids=tokens_tensors)
            
            logits = outputs[0]
            _, preds = torch.max(logits.data, 1)
            

            labels = torch.tensor(data[1])
            for (label, pred) in zip(labels,preds):
                res[label][pred] += 1
    
    p = res[0][0] / (res[0][0] + res[0][1])
    r = res[0][0] / (res[0][0] + res[1][0])
    
    f1 = 2*p*r / (p+r)
    
    return f1
```

#### 5.5 train

```python
# 设置deviceGPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
# 打印初始f1
f1_score = f1(model, valDataLoader)
print("f1_score:", f1_score)

# train
model.train()

# 使用 Adam Optim 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


EPOCHS = 6  # epoch
for epoch in range(EPOCHS):
    
    running_loss = 0.0
    for data in trainDataLoader:
        
        tokens_tensors,labels = [t.to(device) for t in data]

        # 梯度调零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, labels=labels)

        loss = outputs[0]
        
        # backward
        loss.sum().backward()
        optimizer.step()


        # +batch loss
        running_loss += loss.sum().item()   
    torch.save(model.state_dict(), './model{0}.pkl'.format(epoch))

    f1_score = f1(model, valDataLoader)
    print('[epoch %d] loss: %.3f, f1: %.3f' %
          (epoch + 1, running_loss, f1_score))
```

![image-20200702152800818](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702152800818.png)



可以看到，最终在validationDataset上有0.8的f1分数，我把上述的模型提交到原来kaggle的比赛privateScore为0.86976，而之前的第一也只是0.84883.

![image-20200702153045358](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702153045358.png)

之前的榜单第一是0.84，我当时用的是LSTM，跑了一周，最后第九，唉如果早点知道的话，就能干翻第一了，哈哈哈哈。

![image-20200702153100686](D:\program\Typora\img\BERT 可能是目前最强的NLP模型\image-20200702153100686.png)



### 6. 总结

本篇文章主要介绍了BERT的简介，Transformer的架构以及BERT的具体使用，通过一个实际例子展示了BERT在NLP方面的强大，如果大家有不明白的地方，欢迎私信我。（这里是人工学习，机器智能专栏，与你一起开启AI之旅）



[^1]: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[^2]: [wiki:BERT](https://en.wikipedia.org/wiki/BERT_(language_model))
[^3]: [Attention Is All You Need ](https://arxiv.org/abs/1706.03762)
[^4]:  [hung-yi-Lee Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA&t=1859s)

[^5]: [hung-yi-Lee ELMO,BERT,GPT](https://www.youtube.com/watch?v=UYPa347-DdE)
[^6]:[transformers document](https://huggingface.co/transformers/)
[^7 ]: [BERT代码教程](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)







