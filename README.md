# reading

# Machine Learning & Deep Learning Reading Material

#### Fully Convolutional Networks for Semantic Segmentation (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

- upsampling 을 이용한 fully connected network Image sesmentation
- https://github.com/shelhamer/fcn.berkeleyvision.org : 논문에서사용한 코드들 caffe framework를 
- upsampling 을 tensorflow 로 구현 : http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
- http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/ : CNN을 이용한 image segmentation

#### Multi-Dimensional Recurrent Neural Networks (https://arxiv.org/pdf/0705.2011.pdf)

- Multi Dimensions RNN (MDRNN) 을 이용한 image segmentation

#### https://github.com/tencia/video_predict

- video 를 판별 CNN + RNN


#### An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/pdf/1507.05717.pdf)

- 15년 논문이지만 기본적인 CRNN의 개념을 잘 설명하고 있다.
- https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py
- https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py -> bidirectional lstm 
- https://gist.github.com/kastnerkyle/90a6d0f6789e15c38e97 : multidimensional rnn

#### Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
- https://github.com/KaimingHe/deep-residual-networks : Resnet 을 카페로 
#### Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning(https://arxiv.org/pdf/1602.07261.pdf)

- https://github.com/xuyuwei/resnet-tf : resnet 모델을 tensorflow로 

#### Financial Market Time Series Prediction with Recurrent Neural (http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf)
- http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/ 
#### Fast R-CNN (http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

- http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/


#### Region-based Convolutional Networks for Accurate Object Detection and Segmentation (https://people.eecs.berkeley.edu/~rbg/papers/pami/rcnn_pami.pdf)

- https://github.com/AlpacaDB/selectivesearch : selective search library

#### http://laonple.blog.me/220873446440 (딥러닝 관련 블로그)

#### https://github.com/BVLC/caffe/wiki/Model-Zoo : caffe prototxt 로 작성된유명 모델

#### Large scale deep learning for computer aided detection of
mammographic lesions http://ac.els-cdn.com/S1361841516301244/1-s2.0-S1361841516301244-main.pdf?_tid=398e52c4-1db1-11e7-91e1-00000aacb361&acdnat=1491803447_ba29f99efa439280d5b6478108bff0c2

 - 유방암 진단 , data augmentation 시 elastic distortion 을 수행
 

#### Convolutional Neural Pyramid for Image Processing (https://arxiv.org/pdf/1704.02071.pdf)


#### A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection(https://arxiv.org/pdf/1704.03414.pdf)

#### Spatial Transformer Networks (http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)
- 구현 : https://github.com/skaae/transformer_network/blob/master/transformerlayer_test.py

#### Dataset Augmentation for Pose and Lighting Invariant Face Recognition(https://arxiv.org/pdf/1704.04326v1.pdf)

#### U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/pdf/1505.04597.pdf)

#### Knowledge Transfer for Melanoma Screening with Deep Learning (https://arxiv.org/pdf/1703.07479.pdf)
- 데이터수가 부족한 의료데이터면 transfer learning 이 효과적 -> pre-trained 를 적극 적극 활용할 것
- https://github.com/aleju/imgaug : data augmentation zip


#### A Review on Deep Learning Techniques Applied to Semantic Segmentation (https://arxiv.org/pdf/1704.06857v1.pdf)


#### Learning Deep Representations for Scene Labeling with Guided Supervision(https://arxiv.org/pdf/1706.02493v1.pdf)
 - CNN 을 적용할 때 Sub class 에 관한 paper


#### https://github.com/beamandrew/medical-data
 - medical data source 를 정리해 놓은 github


#### https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- VAE 튜토리얼 

#### https://arxiv.org/pdf/1706.00712.pdf
- Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?

#### https://arxiv.org/pdf/1606.00915.pdf
 - DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
 
#### https://arxiv.org/pdf/1509.01626.pdf
- char CNN
#### http://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220952339643
- atrous conv 설명한 

#### https://www.researchgate.net/publication/304163398_Deep_Learning_for_Identifying_Metastatic_Breast_Cancer
- Deep Learning for Identifying Metastatic Breast Cancer

#### https://arxiv.org/pdf/1702.01816.pdf
- Prediction of Kidney Function from Biopsy Images Using Convolutional Neural Networks

#### https://arxiv.org/pdf/1511.06434.pdf
- UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

#### https://web.stanford.edu/class/cs224n/reports/2761133.pdf
 - Text Generation using Generative Adversarial Training

#### http://cs.nyu.edu/~akv245/advtext/report.pdf
- Adversarial Objectives for Text Generation

#### https://summer.kics.or.kr/storage/paper/event/20170621_workshop/publish/9A-3.pdf
- Neural Network 기반의 Generative 모델 챗봇 기술 분석

#### https://arxiv.org/pdf/1606.03657.pdf
- InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

#### https://arxiv.org/pdf/1706.03850.pdf
- Adversarial Feature Matching for Text Generation

#### https://arxiv.org/pdf/1505.03540.pdf
 - Brain Tumor Segmentation with Deep Neural Networks
