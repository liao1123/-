## 项目参考github网址
[跨模态检索明文项目：主要获取预训练模型、数据集等](https://github.com/OFA-Sys/Chinese-CLIP)  
[可搜索加密方案主体](https://github.com/SSE-CMR/SSECMR)  
[前端展示页面demo，but 不运行了？](https://www.modelscope.cn/studios/iic/chinese_clip_applications/summary)  
[CLIP模型获取的实值向量需要转化成二进制hash码，参考这篇文章，其实只用到了多层感知机训练一个hash空间](https://github.com/XinyuXia97/UCMFH)
## 项目py文件解释
1. dataset_wash.py  
    对获取到的数据集进行清洗，本文件主要采用的是Flickr数据集（Chinese-CLIP给出的数据集，先使用的是MUGE，但是这个数据集不是语义的，有打label嫌疑）  
    > Flickr数据集：图像31783幅，文本158915条
    
    原始下载的数据文件
    把原始数据文件train test valid原始数据集合并一起（实际上是为了数据更加丰富，当然重新训练模型不可取）  
    这里对原始的id先进行排序操作（原本的是打乱的不好进行读取） 然后重新设置id让他们连续  
    不使用jsonl格式进行保存，读取太慢，虽然可以可视化，但是使用torch吧，读取快且省空间  
    但是jsonl可以可视化
2. extract_feature.py  
    从网站上下载与训练好的CLIPmodel，然后推理数据集得到图片文本特征保存文件夹  
    主要参数包括：模型、数据集
3. make_img_text_pair.py  
    从清洗好的文本数据集里面获取图像和文本的对应关系，获取组合对应的图片文本对，方便训练hash函数
4. train_MLP_feature2hash.py  
    利用上一步获取的文本图像对，分别训练文本侧和图像侧的多层感知机，应用对比损失函数来降低他们之间的差距，保存训练好的模型。
5. extract_hash.py
    读取之前保存的图片文本特征，对图片、文本对应的特征输入已经训练好的多层感知机去得到对应的二进制hash码   
    主要参数包括：模型、数据集、hash码长度
6. IMI_enIMI.py  
    按照可搜索加密方案要求生成倒排多索引然后对其进行加密  
7. total_show.py  
    没有实现实时获取内存，获取GPU的内存变化，只实现到了CPU的实时变化图  
    主体三个函数是文搜图、图搜文、图搜图，主要功能是传入检索函数数量、模型、图片或文本参数，返回运行结果和时间消耗
8. total_retrial.py  
    主要就是三种模态的检索过程，导入CLIP模型实时推理实值特征、导入训练好的多层感知机将实值特征变成二进制hash码、将hash进行可搜索加密处理进行查找。
9. hyper_parameter.py  
    放一些地址或者整个项目的超参，其实可以用args代替
10. 123.py  
    就是平常用来简单脚本练习的文件


