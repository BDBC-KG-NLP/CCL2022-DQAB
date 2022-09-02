## RocketQA

[TOC]

### Task1 & Task2（使用rocketqa官网流程进行）

#### 1. 构建para文件

1. 使用`content.xlsx`构造task1需要检索的文本库，这里假设取名为`task1.para`，格式为**标题、文本、ID**，用**'\t'**来分隔

   ```python
   # 根据自己对数据文件的理解构造task1.para用于检索
   import pandas as pd
   
   with open(r'./data/task1.para', 'w', encoding='utf-8') as fw:
       
       # content.xlsx的简单遍历方式
       df = pd.read_excel('./data/content.xlsx')
       for line in df.values:
           title = str(line[0]).strip()
           text = ''.join(str(line[1]).split('\n'))
           label_content = eval(line[2], {'null': ''})
           content_key = str(line[3]).strip()
           
   		fw.write(title + '\t' + text + '\t' + content_key + '\n')
   ```

2. 使用`content_detail.xlsx`(官网的名称应该是“`所有段落列表.xlsx`”)构造task2需要检索的文本库，这里假设取名为`task2.para`，格式为**标题、文本、ID、detail**，用**'\t'**来分隔

   ```python
   import pandas as pd
   
   df = pd.read_excel('../data/content_detail.xlsx')
   with open('task2.para', 'w', encoding='utf-8') as f:
       for line in df.values:
           f.write(str(line[2]).strip() + '\t' + str(line[3]).strip() + '\t' + str(line[0]).strip() + '\t' + str(line[1]).strip() + '\n')
   ```

#### 2. 构造训练文件

1. 训练文件格式如下表：

   |      | dual.tsv           | cross.tsv         |
   | ---- | ------------------ | ----------------- |
   | 1    | question           | question          |
   | 2    | positive_title/aim | title/aim         |
   | 3    | positive_para      | para              |
   | 4    | negative_title/aim | positive/negative |
   | 5    | negative_para      |                   |
   | 6    | positive/negative  |                   |

   dual模型的训练文件格式为**问题、正例标题、正例文本、负例标题、负例文本、'0'**，用**'\t'**来分隔

   cross模型的训练文件格式为**问题、标题、文本、'1'/'0'**，用**'\t'**来分隔

2. 使用train.txt中的详细信息构造对应的训练文件

   - 对于dual_encoder的训练文件构造，采用elasticsearch或者bm25的方式，针对每个问题，从文档库中检索出与该问题答案文本最为相似（但不是正确答案）的Top20/Top50的文本进行负例构造
   - 对于cross_encoder的训练文件构造，采用dual_encoder对训练集问题生成一次结果，将结果文件中的非正确答案作为负例构造

#### 3. 训练模型

与RocketQA官方文档相同，参考https://github.com/PaddlePaddle/RocketQA

```python
import rocketqa

def train_dual_encoder(base_model, train_set):
    dual_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=0, batch_size=64)
    dual_encoder.train(train_set, 2, 'task2_de', save_steps=5000, learning_rate=1e-5,
                       log_folder='task2_dual_log')

def train_cross_encoder(base_model, train_set):
    cross_encoder = rocketqa.load_model(model=base_model, use_cuda=True, device_id=3, batch_size=64)
    cross_encoder.train(train_set, 10, 'task1_cross', save_steps=3000, learning_rate=3e-5, log_folder='task1_cross_log')

if __name__ == '__main__':
    train_dual_encoder('zh_dureader_de_v2', '../data/task2_dual.tsv')
    # train_cross_encoder('zh_dureader_ce_v2', '../data/task2_cross.tsv')

```

#### 4. 调用模型，生成结果文件

task2大体流程如下，详细代码请参考`test.py`

```python
# 首先根据para文件构建好para_list, title_list, id_list, detail_list
para_list, title_list, id_list, detail_list = [], [], [], []
for line in open(tp_file, encoding='utf-8'):
    t, p, id, detail = line.split('\t')
    detail = eval(detail)
    para_list.append(p)
    title_list.append(t)
    id_list.append(id)
    detail_list.append(detail)

    
# 使用encoder返回scores，使用heapq获取top5的脚标top5index
scores = dual_encoder.matching(query=query_list, \
                                           para=para_list[:len(query_list)], \
                                           title=title_list[:len(query_list)])
scores = list(scores)
top5 = heapq.nlargest(5, scores)
top5index = heapq.nlargest(5, range(len(scores)), scores.__getitem__)

# 使用top5index构成结果文件
answer = [{'content-key': id_list[idx], 'detail': detail_list[idx]} for idx in top5index]
fw.write(json.dumps({
    'question': query,
    'answer': answer
}, ensure_ascii=False) + '\n')
```



### Task3 任务说明文档

#### 1、思路介绍

子任务三，为细粒度文本级答案抽取任务，要求从候选答案段落中找到一个或者多个连续的片段作为答案，此任务要求模型具有细粒度的文本理解和信息抽取能力。

在前序任务中，已经得到与用户提问`Q`相关的候选答案段落集合`P'`（即子任务二的输出），进一步在段落中抽取连续片段作为细粒度答案，答案可以为词（Word）、短语（Phrase）或句子（Sentence）各种形式，这是一个片段抽取的阅读理解任务。所以采用阅读理解模型强大的语义信息理解能力实现细粒度答案抽取目标。

#### 2、阅读理解模型训练

- 通用领域训练的阅读理解模型构建。
  - 通过收集和爬取大量网络上开源的片段抽取式问答数据集和实体识别数据集，基于RoBERTa中文语言模型训练机器阅读理解问答模型，增强模型语义建模和信息抽取能力
  - 参考[BDBC-KG-NLP/Chinese-Pretrain-MRC-Model (github.com)](https://github.com/BDBC-KG-NLP/Chinese-Pretrain-MRC-Model)
- 任务领域数据微调
  - 任务给出的训练集和开发集数据经过整理后，即为问题`question`和答案`answer`对。进一步构造段落数据，寻找包含`answer`所在上下文，以完整句子为界获取段落文本`content`，构成`（quetion, content, answer）`三元组，即为训练数据
  - 负例构造。针对问题question随机选择其他段落文本构造负例，通过负样本训练能提高模型泛化能力
  - 在通用阅读理解模型基础上，通过上述数据进行微调，实现Task3答案文本抽取模型。

#### 3、方法流程

- 针对每一条待测数据在任务2中找到其所有相关段落，以用户提问作为阅读理解模型问题`question`，任务2答案段落作为阅读理解模型`content`，通过阅读理解模型抽取答案文本（区间）
- 由于该阅读理解模型存在段落长度512的限制，对于超长的文本，采用滑动窗口方法分段处理
- 对于多个答案，以篇章`key`作为划分依据，同一篇章的不同段落答案记为同一组答案，转化为比赛提交所需要的数据格式。

具体代码见附件
