# textClassifier

# IMDB电影评论二分类

## 训练数据  
[labeledTrainData.tsv](https://github.com/charlesliucn/kaggle-in-python/blob/master/kaggle_competitions/IMDB/labeledTrainData.tsv)

## 词向量文件  
[glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)  
gswyhq@gswyhq-PC:~/data/glove.6B$ unzip ../glove.6B.zip  
Archive:  ../glove.6B.zip  
  inflating: glove.6B.50d.txt          
  inflating: glove.6B.100d.txt       
  inflating: glove.6B.200d.txt       
  inflating: glove.6B.300d.txt 
  
## 运行环境及依赖
```
Deepin GNU/Linux 15.9 \n \l
pip3 install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com  
```

## 训练及预测
* 多层注意力模型   
训练：`python3 textClassifierHATT.py train`    
预测：`python3 textClassifierHATT.py`   


* 卷积神经网络  
训练：`python3 textClassifierConv.py train`  
预测：`python3 textClassifierConv.py`  

* 循环神经网络  
1. Bi-LSTM  
2. Attention GRU  
训练：`python3 textClassifierRNN.py train`  
预测：`python3 textClassifierRNN.py`


This repo is forked from [https://github.com/richliao/textClassifier](https://github.com/richliao/textClassifier)
