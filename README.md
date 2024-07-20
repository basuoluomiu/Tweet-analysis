# 基于Bert预训练模型的社交媒体情感分析系统

此项目旨在建立一个系统，能够自动识别人们在 Twitter 上发表的推文表达的情绪状态（例如愤怒、开心、恐惧等），对于此任务，项目中选取 Bert 实现，使用来自 Hugging Face 系统的三个核心库：Datasets、Tokenizers、Transformers。

## 数据集

在我所选用的数据集中，有6类不同的情感标签，所以可以将其视为6分类问题。

在数据处理环节，我定义了一个函数讲labels的数值转换为字符串值。

## 标记化

Bert 无法接收原始 python 字符串作为输入，我们需要将字符串分解为称为标记的子组，并将其编码为数字向量。

### 1.字符标记化，使用 python 内置的 list 类

编写了token2idx函数，将词汇表中的每个字符映射为一个唯一的整数，组成词汇字典后就可以重构文本，再使用 Pytorch 内置功能将其转换为独热编码。

```python
text = 'Tokenisation of text is a core task of NLP.'
tokenised_text = list(text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenised_text)))}

print(f'Length of vocabulary: {len(token2idx)}')
print(token2idx)
```

字符标记化的缺点：字符级标记化忽略文本中的任何结构，将整个字符串视为字符流，这有助于处理拼写错误和后置词，但主要缺点是需要从数据中学习语言结构

### 2.单词标记化

我们可以将文本分割成单词，并将每个单词映射为一个整数，而不是将文本分割成字符。最简单的标记化形式是利用 python 内置的字符串类分割方法，与字符标记化不同的是，如果我们有去音、变位、拼写错误，词汇字典的大小就会迅速增长词汇量越大，问题就越大，因为这需要模型有过量的参数（效率很低）。通常会选择语料库中最常见的 10 万个单词不在词汇表中的词被归类为未知词，并映射到共享的 UNK 标记上。然而，在标记化过程中可能会丢失一些重要信息，因为模型没有与 UNK 相关的词的信息

### 3.子词标记化

子词标记化结合了字符和词标记化的优点，主要特点是：它是通过混合使用统计规则和算法，从预训练语料库中学习出来的。有几种常用于 NLP 的子词标记化算法，WordPiece，它被 BERT 和 DistilBERT 标记化器使用；AutoTokenizer 类允许我们快速加载与预训练模型相关的标记符号器； transformers.BertTokenizer 中手动加载标记符号器。

我们使用从transformer中导入分词器来完成。

## 模型训练

### 1.加载预训练模型

我们将使用 model_ckpt "bert-base-uncased" 加载相同的 BERT 模型。不过，这次我们将加载 AutoModelForSequenceClassification（我们在提取嵌入特征时使用了 AutoModel）。AutoModelForSequenceClassification 模型在预训练模型输出上有一个分类头，我们只需指定模型需要预测的标签数 num_labels=6。

### 2.定义性能指标

我们将监控 F1 分数和准确性，该函数需要在 Trainer 类中传递。

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

### 3.训练参数

接下来需要定义模型训练参数，这可以通过 TrainingArguments 来完成：



| 参数       | 值   |
| ---------- | ---- |
| 迭代次数   | 3    |
| 学习率     | 2e-5 |
| batch_size | 64   |
| 权重衰减   | 0.01 |

### 4.模型训练

![image](https://github.com/user-attachments/assets/3ded7fcc-b93c-420f-8420-f605e398f484)


## 模型分析

我们应该对模型的预测进行更深入的研究，根据模型损失对验证进行排序，写一个函数，返回模型损失和预测标签 forward_pass_with_label

```python
def forward_pass_with_label(batch):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = torch.nn.functional.cross_entropy(output.logits,batch["label"].to(device),reduction="none")
        
    return {"loss":loss.cpu().numpy(),"predicted_label":pred_label.cpu().numpy()}

emotions_encoded.set_format("torch",columns=["input_ids", "attention_mask", "label"])
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label,batched=True,batch_size=16)
```

## 使用已训练模型

我们利用 AutoModelForSequenceClassification 训练了该模型，该模型在基本 DistilBERT 模型中添加了一个分类头，当我们需要对新的非种子数据进行模型预测时，我们可以利用 pipeline 方法，假设有一句话：I finally snagged tickets to Jay's concert!

```python
from transformers import pipeline

classifier = pipeline("text-classification",model="bert-base-uncased-finetuned-emotion")

new_data = 'I finally snagged tickets to Jay's concert！'
```
out:
```python
[{'label': 'LABEL_1', 'score': 0.933964729309082}]
```

## 集成到web

使用Flask轻量级web框架编写程序

在主文件app.py中引入flask、torch、从transformer中引入pipeline方法

```python
model_name = "bert-base-uncased-finetuned-emotion"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pipeline("text-classification", model=model_name, device=device)
```

定义一个index路由显示主页面，一个predict路由进行推理预测。

```python
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    predictions = predict_sentiment(text)
    predictions = Matching(predictions)
    return jsonify(predictions)
```

在前端页面设置一个仪表盘，在用户预测完成后显示预测值与置信度
![image](https://github.com/user-attachments/assets/5f3e2469-9679-44c8-a561-6c9c286e3002)


## 本地部署

1.按照requirements.txt安装依赖

2.确保bert-base-uncased-finetuned-emotion预训练模型各参数文件完整。

3.运行app.py
