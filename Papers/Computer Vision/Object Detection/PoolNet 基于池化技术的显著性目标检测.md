# CVPR 2019 | PoolNet：基于池化技术的显著性目标检测

2019/05/27 16:47

**作者:** 文永亮 哈尔滨工业大学（深圳）

**研究方向:** 目标检测、GAN.


![!\[img\](https://image.jiqizhixin.com/uploads/editor/0da47236-692c-48e8-b9b6-0454180237e0/640.png)][1]


**研究动机:** 这是一篇发表于 CVPR 2019 的关于显著性目标检测的 paper，在 U 型结构的特征网络中，高层富含语义特征捕获的位置信息在自底向上的传播过程中可能会逐渐被稀释，另外卷积神经网络的感受野大小与深度是不成正比的。

目前很多流行方法都是引入 Attention（注意力机制），但是本文是基于 U 型结构的特征网络研究池化对显著性检测的改进，具体步骤是引入了两个模块GGM (Global Guidance Module，全局引导模块) 和 FAM (Feature Aggregation Module，特征整合模块)，进而锐化显著物体细节，并且检测速度能够达到 30FPS。因为这两个模块都是基于池化做的改进所以作者称其为PoolNet，并且放出了源码：

https://github.com/backseason/PoolNet

模型架构

![image_1dd2ou06t1fm91srd1odv1grn2tbm.png-358.7kB][2]

## 两个模块

### GGM（全局引导模块）

我们知道高层语义特征对挖掘显著对象的详细位置是很有帮助的，但是中低层的语义特征也可以提供必要的细节。因为在 top-down 的过程中，高层语义信息被稀释，而且实际上的感受野也是小于理论感受野，所以对于全局信息的捕捉十分的缺乏，导致显著物体被背景吞噬。

因此作者提出了 GGM 模块，GGM 其实是 PPM（Pyramid Pooling module，金字塔池化模块）的改进并且加上了一系列的 GGFs（Global Guiding Flows，全局引导流），这样做的好处是，在特征图上的每层都能关注到显著物体，另外不同的是，GGM 是一个独立的模块，而 PPM 是在 U 型架构中，在基础网络（backbone）中参与引导全局信息的过程。 

其实这部分论文说得并不是很清晰，没有说 GGM 的详细结构，我们可以知道 PPM \[7] 的结构如下：

![!\[img\](https://image.jiqizhixin.com/uploads/editor/9b8e4e5b-7048-452f-9280-13b184e44068/640.png)][3]

该 PPM 模块融合了 4 种不同金字塔尺度的特征，第一行红色是最粗糙的特征–全局池化生成单个 bin 输出，后面三行是不同尺度的池化特征。为了保证全局特征的权重，如果金字塔共有 N 个级别，则在每个级别后使用 1×1 的卷积将对于级别通道降为原本的 1/N。再通过双线性插值获得未池化前的大小，最终 concat 到一起。 

如果明白了这个的话，其实 GGM 就是在 PPM 的结构上的改进，PPM 是对每个特征图都进行了金字塔池化，所以作者说是嵌入在 U 型结构中的，但是他加入了 global guiding flows（GGFs），即 Fig1 中绿色箭头，引入了对每级特征的不同程度的上采样映射（文中称之为 identity mapping），所以可以是个独立的模块。

简单地说，作者想要 FPN 在 top-down 的路径上不被稀释语义特征，所以在每次横向连接的时候都加入高层的语义信息，这样做也是一个十分直接主观的想法。 

FAM（特征整合模块）

特征整合模块也是使用了池化技巧的模块，如下图，先把 GGM 得到的高层语义与该级特征分别上采样之后横向连接一番得到 FAM 的输入 b，之后采取的操作是先把 b 用 {2,4,8} 的三种下采样得到蓝绿红特征图然后 avg pool（平均池化）再上采样回原来尺寸，最后蓝绿红紫（紫色是 FAM 的输入 b）四个分支像素相加得到整合后的特征图。

![!\[img\](https://image.jiqizhixin.com/uploads/editor/1389eb63-9b89-45bf-9cb6-e97ee107abc4/640.png)][4]

### FAM 有以下两个优点： 

1. 帮助模型降低上采样（upsample）导致的混叠效应（aliasing）；

2. 从不同的多角度的尺度上纵观显著物体的空间位置，放大整个网络的感受野。 

第二点很容易理解，从不同角度看，不同的放缩尺度看待特征，能够放大网络的感受野。对于第一点降低混叠效应的理解，用明珊师姐说的话，混叠效应就相当于引入杂质，GGFs 从基础网络最后得到的特征图经过金字塔池化之后需要最高是 8 倍上采样才能与前面的特征图融合，这样高倍数的采样确实容易引入杂质。

作者就是因为这样才会提出 FAM，进行特征整合，先把特征用不同倍数的下采样，池化之后，再用不同倍数的上采样，最后叠加在一起。因为单个高倍数上采样容易导致失真，所以补救措施就是高倍数上采样之后，再下采样，再池化上采样平均下来可以弥补错误。

![!\[img\](https://image.jiqizhixin.com/uploads/editor/f102ad62-39c1-41c0-934c-b1c55945c754/640.png)][5]

上图就是为了说明 FAM 的优点的，经过高倍上采样之后的图像（b）和（d）容易引入许多杂质，致使边缘不清晰，但是经过 FAM 模块之后的特征图就能降低混叠效应。

## 实验结果

论文在常用的 6 种数据集上做了实验，有 ECSSD \[8], PASCALS \[9], DUT-OMRON \[10], HKU-IS [11], SOD [12] 和 DUTS [13], 使用二值交叉熵做显著性检测，平衡二值交叉熵（balanced binary cross entropy）[14] 作为边缘检测（edge detection）。

以下是文章方法跟目前 state-of-the-arts 的方法的对比效果，绿框是 GT，红框是本文效果。可以看到无论在速度还是精度上都有很大的优势。

![!\[img\](https://image.jiqizhixin.com/uploads/editor/8b1f0599-a28c-4af0-9de4-30a2b712409f/640.png)][6]

![!\[img\](https://image.jiqizhixin.com/uploads/editor/b7d2f2e7-ecea-448a-8851-4a6a802fc4d6/640.png)][7]

![!\[img\](https://image.jiqizhixin.com/uploads/editor/8404eda0-6049-4a5e-a8be-6fa463d68225/640.png)][8]

论文还针对三个改进的技术 PPM、GGFs 和 FAMs 的不同组合做了实验，(a) 是原图，(b) 是 Ground truth，(c) 是 FPN 的结果，(d) 是 FPN+FAMs，(e) 是 FPN+PPM，(f) 是 FPN+GGM，(g) 是 FPN+GGM+FAMs。

![!\[img\](https://image.jiqizhixin.com/uploads/editor/c3fc1a1f-a488-467e-b172-619f941b26f2/640.png)][9]

![!\[img\](https://image.jiqizhixin.com/uploads/editor/71bbc6f4-2e86-4a82-8767-1f1fc88a2a1d/640.png)][10]

## 总结

该 paper 提出了两种基于池化技术的模块 GGM（全局引导模块）和 FAM（特征整合模块），改进 FPN 在显著性检测的应用，而且这两个模块也能应用在其他金字塔模型中，具有普遍性，但是 FAM 的整合过程我认为有点像是用平均中和了上采样带来的混叠效应，但是不够优雅，先下采样池化再上采样带来的损失可能代价太大。


## 参考文献


\[1]. Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In CVPR, 2017. 1, 3. 

\[2]. Tiantian Wang, Ali Borji, Lihe Zhang, Pingping Zhang, and Huchuan Lu. A stagewise refinement model for detecting salient objects in images. In ICCV, pages 4019–4028, 2017. 1, 3, 6, 7, 8.

\[3].Nian Liu and Junwei Han. Dhsnet: Deep hierarchical saliency network for salient object detection. In CVPR, 2016.1, 2, 3, 7, 8. 

\[4]. Qibin Hou, Ming-Ming Cheng, Xiaowei Hu, Ali Borji, Zhuowen Tu, and Philip Torr. Deeply supervised salient object detection with short connections. IEEE TPAMI, 41(4):815–828, 2019. 1, 2, 3, 5, 6, 7, 8. 

\[5]. Tiantian Wang, Ali Borji, Lihe Zhang, Pingping Zhang, and Huchuan Lu. A stagewise refinement model for detecting salient objects in images. In ICCV, pages 4019–4028, 2017. 1, 3, 6, 7, 8. 

\[6]. Tiantian Wang, Lihe Zhang, Shuo Wang, Huchuan Lu, Gang Yang, Xiang Ruan, and Ali Borji. Detect globally, refine locally: A novel approach to saliency detection. In CVPR, pages 3127–3135, 2018. 1, 3, 6, 7, 8. 

\[7]. Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In CVPR, 2017. 1, 3. 

\[8]. Qiong Yan, Li Xu, Jianping Shi, and Jiaya Jia. Hierarchical saliency detection. In CVPR, pages 1155–1162, 2013. 1, 5, 8.

\[9]. Yin Li, Xiaodi Hou, Christof Koch, James M Rehg, and Alan L Yuille. The secrets of salient object segmentation. In CVPR, pages 280–287, 2014. 5, 7, 8. 

\[10]. Chuan Yang, Lihe Zhang, Huchuan Lu, Xiang Ruan, and Ming-Hsuan Yang. Saliency detection via graph-based manifold ranking. In CVPR, pages 3166–3173, 2013. 5, 6, 7, 8.

[11]. Guanbin Li and Yizhou Yu. Visual saliency based on multiscale deep features. In CVPR, pages 5455–5463, 2015. 2, 5, 6, 7, 8. 

[12]. Vida Movahedi and James H Elder. Design and perceptual validation of performance measures for salient object segmentation. In CVPR, pages 49–56, 2010. 5, 6, 7, 8. 

[13]. Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, and Xiang Ruan. Learning to detect salient objects with image-level supervision. In CVPR, pages 136–145, 2017. 5, 7, 8.

[14]. Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In ICCV, pages 1395–1403, 2015. 6.


  [1]: http://static.zybuluo.com/harrytsz/9ucb9mzitidkb4hwfqz2ws8x/image_1dd2ot8q4vte1iht1dqm10km16ah9.png
  [2]: http://static.zybuluo.com/harrytsz/2mu5wi88k9avs4840vtvs2ga/image_1dd2ou06t1fm91srd1odv1grn2tbm.png
  [3]: http://static.zybuluo.com/harrytsz/eys1eczwfvpol13g22mmpi62/image_1dd2oui9c1v191ojukfhmeupve13.png
  [4]: http://static.zybuluo.com/harrytsz/o0ivhhcw31x3kh7k9czldoy3/image_1dd2ov1kupn25pgcne1v541en71g.png
  [5]: http://static.zybuluo.com/harrytsz/lze10fmm9ma5ik3h471fri79/image_1dd2ovegl15hm1ftn1mj27a810851t.png
  [6]: http://static.zybuluo.com/harrytsz/2qgg2u25r8g2adi4akr7fb2t/image_1dd2ovu3fcl81olejp0h4b12rv2a.png
  [7]: http://static.zybuluo.com/harrytsz/1yq352umjdmawgyoc3hfvour/image_1dd2p08s88571q9110cne85ed2n.png
  [8]: http://static.zybuluo.com/harrytsz/glvpazgc84b4tqfj1fev15tv/image_1dd2p0j4iphl1gourbf1mud1p4u34.png
  [9]: http://static.zybuluo.com/harrytsz/031tgsk4s61vzqs6hiox62oq/image_1dd2p129u3i9bsj19onqb5mub3h.png
  [10]: http://static.zybuluo.com/harrytsz/hhwwbxbsv2c67rsbym83eubo/image_1dd2p1aoqtn91k1si7b5li104b3u.png
