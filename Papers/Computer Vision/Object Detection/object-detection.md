# [object-detection](https://github.com/amusi/awesome-object-detection/blob/master/README.md)

[TOC]

This is a list of awesome articles about object detection. If you want to read the paper according to time, you can refer to [Date](Date.md).

- R-CNN
- Fast R-CNN
- Faster R-CNN
- Mask R-CNN
- Light-Head R-CNN
- Cascade R-CNN
- SPP-Net
- YOLO
- YOLOv2
- YOLOv3
- YOLT
- SSD
- DSSD
- FSSD
- ESSD
- MDSSD
- Pelee
- Fire SSD
- R-FCN
- FPN
- DSOD
- RetinaNet
- MegDet
- RefineNet
- DetNet
- SSOD
- CornerNet
- M2Det
- 3D Object Detection
- ZSD（Zero-Shot Object Detection）
- OSD（One-Shot object Detection）
- Weakly Supervised Object Detection
- Softer-NMS
- 2018
- 2019
- Other

Based on handong1587's github: https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html

# Survey

**Object Detection in 20 Years: A Survey**

- intro：This work has been submitted to the IEEE TPAMI for possible publication
- arXiv：<https://arxiv.org/abs/1905.05055>

**《Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks》**

- intro: awesome


- arXiv: https://arxiv.org/abs/1809.03193

**《Deep Learning for Generic Object Detection: A Survey》**

- intro: Submitted to IJCV 2018
- arXiv: https://arxiv.org/abs/1809.02165

# Papers&Codes

## R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation**

- intro: R-CNN
- arxiv: <http://arxiv.org/abs/1311.2524>
- supp: <http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf>
- slides: <http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf>
- slides: <http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf>
- github: <https://github.com/rbgirshick/rcnn>
- notes: <http://zhangliliang.com/2014/07/23/paper-note-rcnn/>
- caffe-pr("Make R-CNN the Caffe detection example"): <https://github.com/BVLC/caffe/pull/482>

## Fast R-CNN

**Fast R-CNN**

- arxiv: <http://arxiv.org/abs/1504.08083>
- slides: <http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf>
- github: <https://github.com/rbgirshick/fast-rcnn>
- github(COCO-branch): <https://github.com/rbgirshick/fast-rcnn/tree/coco>
- webcam demo: <https://github.com/rbgirshick/fast-rcnn/pull/29>
- notes: <http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/>
- notes: <http://blog.csdn.net/linj_m/article/details/48930179>
- github("Fast R-CNN in MXNet"): <https://github.com/precedenceguo/mx-rcnn>
- github: <https://github.com/mahyarnajibi/fast-rcnn-torch>
- github: <https://github.com/apple2373/chainer-simple-fast-rnn>
- github: <https://github.com/zplizzi/tensorflow-fast-rcnn>

**A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1704.03414>
- paper: <http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf>
- github(Caffe): <https://github.com/xiaolonw/adversarial-frcnn>

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

- intro: NIPS 2015
- arxiv: <http://arxiv.org/abs/1506.01497>
- gitxiv: <http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region>
- slides: <http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf>
- github(official, Matlab): <https://github.com/ShaoqingRen/faster_rcnn>
- github(Caffe): <https://github.com/rbgirshick/py-faster-rcnn>
- github(MXNet): <https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn>
- github(PyTorch--recommend): <https://github.com//jwyang/faster-rcnn.pytorch>
- github: <https://github.com/mitmul/chainer-faster-rcnn>
- github(Torch):: <https://github.com/andreaskoepf/faster-rcnn.torch>
- github(Torch):: <https://github.com/ruotianluo/Faster-RCNN-Densecap-torch>
- github(TensorFlow): <https://github.com/smallcorgi/Faster-RCNN_TF>
- github(TensorFlow): <https://github.com/CharlesShang/TFFRCNN>
- github(C++ demo): <https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus>
- github(Keras): <https://github.com/yhenon/keras-frcnn>
- github: <https://github.com/Eniac-Xie/faster-rcnn-resnet>
- github(C++): <https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev>

**R-CNN minus R**

- intro: BMVC 2015
- arxiv: <http://arxiv.org/abs/1506.06981>

**Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: <https://github.com/dmlc/mxnet/tree/master/example/rcnn>

**Contextual Priming and Feedback for Faster R-CNN**

- intro: ECCV 2016. Carnegie Mellon University
- paper: <http://abhinavsh.info/context_priming_feedback.pdf>
- poster: <http://www.eccv2016.org/files/posters/P-1A-20.pdf>

**An Implementation of Faster RCNN with Study for Region Sampling**

- intro: Technical Report, 3 pages. CMU
- arxiv: <https://arxiv.org/abs/1702.02138>
- github: <https://github.com/endernewton/tf-faster-rcnn>
- github: https://github.com/ruotianluo/pytorch-faster-rcnn

**Interpretable R-CNN**

- intro: North Carolina State University & Alibaba
- keywords: AND-OR Graph (AOG)
- arxiv: <https://arxiv.org/abs/1711.05226>

**Domain Adaptive Faster R-CNN for Object Detection in the Wild**

- intro: CVPR 2018. ETH Zurich & ESAT/PSI
- arxiv: <https://arxiv.org/abs/1803.03243>

## Mask R-CNN

- arxiv: <http://arxiv.org/abs/1703.06870>
- github(Keras): https://github.com/matterport/Mask_RCNN
- github(Caffe2): https://github.com/facebookresearch/Detectron
- github(Pytorch): <https://github.com/wannabeOG/Mask-RCNN>
- github(MXNet): https://github.com/TuSimple/mx-maskrcnn
- github(Chainer): https://github.com/DeNA/Chainer_Mask_R-CNN

## Light-Head R-CNN

**Light-Head R-CNN: In Defense of Two-Stage Object Detector**

- intro: Tsinghua University & Megvii Inc
- arxiv: <https://arxiv.org/abs/1711.07264>
- github(offical): https://github.com/zengarden/light_head_rcnn
- github: <https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784>

## Cascade R-CNN

**Cascade R-CNN: Delving into High Quality Object Detection**

- arxiv: <https://arxiv.org/abs/1712.00726>
- github: <https://github.com/zhaoweicai/cascade-rcnn>

## SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- arxiv: <http://arxiv.org/abs/1406.4729>
- github: <https://github.com/ShaoqingRen/SPP_net>
- notes: <http://zhangliliang.com/2014/09/13/paper-note-sppnet/>

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: PAMI 2016
- intro: an extension of R-CNN. box pre-training, cascade on region proposals, deformation layers and context representations
- project page: <http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html>
- arxiv: <http://arxiv.org/abs/1412.5661>

**Object Detectors Emerge in Deep Scene CNNs**

- intro: ICLR 2015
- arxiv: <http://arxiv.org/abs/1412.6856>
- paper: <https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf>
- paper: <https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf>
- slides: <http://places.csail.mit.edu/slide_iclr2015.pdf>

**segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection**

- intro: CVPR 2015
- project(code+data): <https://www.cs.toronto.edu/~yukun/segdeepm.html>
- arxiv: <https://arxiv.org/abs/1502.04275>
- github: <https://github.com/YknZhu/segDeepM>

**Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- keywords: NoC
- arxiv: <http://arxiv.org/abs/1504.06066>

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: <http://arxiv.org/abs/1504.03293>
- slides: <http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf>
- github: <https://github.com/YutingZhang/fgs-obj>

**DeepBox: Learning Objectness with Convolutional Networks**

- keywords: DeepBox
- arxiv: <http://arxiv.org/abs/1505.02146>
- github: <https://github.com/weichengkuo/DeepBox>

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection**

[![img](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

- arxiv: <http://arxiv.org/abs/1506.02640>
- code: <https://pjreddie.com/darknet/yolov1/>
- github: <https://github.com/pjreddie/darknet>
- blog: <https://pjreddie.com/darknet/yolov1/>
- slides: <https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p>
- reddit: <https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/>
- github: <https://github.com/gliese581gg/YOLO_tensorflow>
- github: <https://github.com/xingwangsfu/caffe-yolo>
- github: <https://github.com/frankzhangrui/Darknet-Yolo>
- github: <https://github.com/BriSkyHekun/py-darknet-yolo>
- github: <https://github.com/tommy-qichang/yolo.torch>
- github: <https://github.com/frischzenger/yolo-windows>
- github: <https://github.com/AlexeyAB/yolo-windows>
- github: <https://github.com/nilboy/tensorflow-yolo>

**darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++**

- blog: <https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp>
- github: <https://github.com/thtrieu/darkflow>

**Start Training YOLO with Our Own Data**

[![img](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)

- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: <http://guanghan.info/blog/en/my-works/train-yolo/>
- github: <https://github.com/Guanghan/darknet>

**YOLO: Core ML versus MPSNNGraph**

- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: <http://machinethink.net/blog/yolo-coreml-versus-mps-graph/>
- github: <https://github.com/hollance/YOLO-CoreML-MPSNNGraph>

**TensorFlow YOLO object detection on Android**

- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: <https://github.com/natanielruiz/android-yolo>

**Computer Vision in iOS – Object Detection**

- blog: <https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/>
- github:<https://github.com/r4ghu/iOS-CoreML-Yolo>

## YOLOv2

**YOLO9000: Better, Faster, Stronger**

- arxiv: <https://arxiv.org/abs/1612.08242>
- code: <http://pjreddie.com/yolo9000/>    https://pjreddie.com/darknet/yolov2/
- github(Chainer): <https://github.com/leetenki/YOLOv2>
- github(Keras): <https://github.com/allanzelener/YAD2K>
- github(PyTorch): <https://github.com/longcw/yolo2-pytorch>
- github(Tensorflow): <https://github.com/hizhangp/yolo_tensorflow>
- github(Windows): <https://github.com/AlexeyAB/darknet>
- github: <https://github.com/choasUp/caffe-yolo9000>
- github: <https://github.com/philipperemy/yolo-9000>
- github(TensorFlow): <https://github.com/KOD-Chen/YOLOv2-Tensorflow>
- github(Keras): <https://github.com/yhcc/yolo2>
- github(Keras): <https://github.com/experiencor/keras-yolo2>
- github(TensorFlow): <https://github.com/WojciechMormul/yolo2>

**darknet_scripts**

- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: <https://github.com/Jumabek/darknet_scripts>

**Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2**

- github: <https://github.com/AlexeyAB/Yolo_mark>

**LightNet: Bringing pjreddie's DarkNet out of the shadows**

<https://github.com//explosion/lightnet>

**YOLO v2 Bounding Box Tool**

- intro: Bounding box labeler tool to generate the training data in the format YOLO v2 requires.
- github: <https://github.com/Cartucho/yolo-boundingbox-labeler-GUI>

**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: **LRM** is the first hard example mining strategy which could fit YOLOv2 perfectly and make it better applied in series of real scenarios where both real-time rates and accurate detection are strongly demanded.
- arxiv: https://arxiv.org/abs/1804.04606

**Object detection at 200 Frames Per Second**

- intro: faster than Tiny-Yolo-v2
- arxiv: https://arxiv.org/abs/1805.06361

**Event-based Convolutional Networks for Object Detection in Neuromorphic Cameras**

- intro: YOLE--Object Detection in Neuromorphic Cameras
- arxiv:https://arxiv.org/abs/1805.07931

**OmniDetector: With Neural Networks to Bounding Boxes**

- intro: a person detector on n fish-eye images of indoor scenes（NIPS 2018）
- arxiv:https://arxiv.org/abs/1805.08503
- datasets:https://gitlab.com/omnidetector/omnidetector

## YOLOv3

**YOLOv3: An Incremental Improvement**

- arxiv:https://arxiv.org/abs/1804.02767
- paper:https://pjreddie.com/media/files/papers/YOLOv3.pdf
- code: <https://pjreddie.com/darknet/yolo/>
- github(Official):https://github.com/pjreddie/darknet
- github:https://github.com/mystic123/tensorflow-yolo-v3
- github:https://github.com/experiencor/keras-yolo3
- github:https://github.com/qqwweee/keras-yolo3
- github:https://github.com/marvis/pytorch-yolo3
- github:https://github.com/ayooshkathuria/pytorch-yolo-v3
- github:https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
- github:https://github.com/eriklindernoren/PyTorch-YOLOv3
- github:https://github.com/ultralytics/yolov3
- github:https://github.com/BobLiu20/YOLOv3_PyTorch
- github:https://github.com/andy-yun/pytorch-0.4-yolov3
- github:https://github.com/DeNA/PyTorch_YOLOv3

## YOLT

**You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery**

- intro: Small Object Detection


- arxiv:https://arxiv.org/abs/1805.09512
- github:https://github.com/avanetten/yolt

## SSD

**SSD: Single Shot MultiBox Detector**

[![img](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- intro: ECCV 2016 Oral
- arxiv: <http://arxiv.org/abs/1512.02325>
- paper: <http://www.cs.unc.edu/~wliu/papers/ssd.pdf>
- slides: [http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)
- github(Official): <https://github.com/weiliu89/caffe/tree/ssd>
- video: <http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973>
- github: <https://github.com/zhreshold/mxnet-ssd>
- github: <https://github.com/zhreshold/mxnet-ssd.cpp>
- github: <https://github.com/rykov8/ssd_keras>
- github: <https://github.com/balancap/SSD-Tensorflow>
- github: <https://github.com/a
