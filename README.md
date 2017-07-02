# HAN-text-classification-tf

这是"Hierarchical Attention Network for Document Classification"论文使用tensorflow仿真实现的代码，具体的论文介绍和代码讲解可以参考我的博客内容：
论文笔记： http://blog.csdn.net/liuchonge/article/details/73610734
代码讲解： http://blog.csdn.net/liuchonge/article/details/74092014
数据集下载链接为： 
https://github.com/rekiksab/Yelp/tree/master/yelp_challenge/yelp_phoenix_academic_dataset
为了执行程序，首先你要下载上面的数据集里面的`yelp_academic_dataset_review.json`文件，然后分别执行：

    python DataUtils.py
    python train.py
    
上面一句是先进行数据集的预处理操作，将`yelp_academic_dataset_review.json`文件转化为模型所需要的输入格式，并保存在`yelp_data`文件中。第二句是执行模型训练部分的代码，接下来就需要默默的等待就好了。可以到`0.0.0.0:6006`的本地网址查看tensorboard可视化界面，（在此之前需要执行tensorboard命令，开启该网址）。

模型架构：
## Non-Attention ##
“Document Modeling with Gated Recurrent Neural Network 
for Sentiment Classification”使用两个神经网络分别建模句子和文档，采用一种**自下向上的基于向量的文本表示模型**。模型架构如下图所示：
![这里写图片描述](http://img.blog.csdn.net/20170622194722200?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Y2hvbmdl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## With-Attention  HAN模型 ##
"Hierarchical Attention Network for Document Classification"的模型架构，其实主要的思想和上面的差不多，也是分层构建只不过加上了两个Attention层，用于分别对句子和文档中的单词、句子的重要性进行建模。而且引入Attention机制除了提高模型的精确度之外还可以进行单词、句子重要性的分析和可视化，让我们对文本分类的内部有一定了解。模型主要可以分为下面四个部分，如下图所示：

 1.  a word sequence encoder,
 2.  a word-level attention layer,
 3.  a sentence encoder
 4.  a sentence-level attention layer.

![这里写图片描述](http://img.blog.csdn.net/20170622204502957?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Y2hvbmdl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)、
## 最终的训练结果 ##
![这里写图片描述](http://img.blog.csdn.net/20170702144952787?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Y2hvbmdl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20170702172840958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1Y2hvbmdl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
