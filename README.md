## Contents
- [Packages](#packages)
  - [Generic](#generic)
  - [Time Series](#time-series)
  - [Machine Learning](#machine-learning)
  - [AutoML](#automl)
  - [Visualization](#visualization)
- [Datasets](#datasets)
  - [General](#general)
  - [Economics & Finance](#economics--finance)
  - [Geography](#geography)
  - [Healthcare](#healthcare)
  - [Social Study](#social-study)
- [Books](#books)
- [Resources](#resources)
- [Articles](#articles)
## Packages
This session lists all machine learning related python packages.
### Generic
1. [numpy](www.numpy.org)
2. [numba](http://numba.pydata.org/): A high performance python compiler
3. [pandas](https://pandas.pydata.org/)
4. [pandarallel](https://github.com/nalepae/pandarallel): make pandas run in parallel mode
5. [numexpr](https://github.com/pydata/numexpr): fast evaluation of numpy/pandas expressions
6. [blaze](https://github.com/blaze/blaze): NumPy and Pandas interface to Big Data
7. [datatable](https://datatable.readthedocs.io/en/latest/quick-start.html): Super fast parallel data table handling
8. [vaex](https://github.com/vaexio/vaex): Yet another pandas with memory mapping technique, super fast on super big datasets
9. [scipy](https://www.scipy.org/): for any generic scientific solutions
10. [statsmodels](https://www.statsmodels.org): for any statistics related topics such as distributions, statistical tests, time series analysis
12. [networkx](https://networkx.github.io/): a comprehensive package for all graph algorithms
14. [TensorNetwork](https://github.com/google/TensorNetwork): a very efficient library for tensor calculations
15. [Euler](https://github.com/alibaba/euler): large-scale graph algorithms and models by Alibaba
16. [jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html): Google's NumPy on the CPU, GPU and TPU
17. [ray](https://ray.io/): Fast distributed computing with python
### Time Series
1. [pyflux](https://pyflux.readthedocs.io): an advanced time-series analysis library
2. [pyts](https://github.com/johannfaouzi/pyts): time-series analysis, decomposition and classification
3. [tslearn](https://github.com/tslearn-team/tslearn): machine learning toolkit for time-series analysis
4. [seglearn](https://github.com/dmbee/seglearn): sliding operations/transformations for time-series/sequence data
5. [stumpy](https://stumpy.readthedocs.io/en/latest/Tutorial_STUMPY_Basics.html): Find motifs and anomalies in time-series data
6. [sktime](https://github.com/alan-turing-institute/sktime): toolbox for machine learning with time-series
7. [tsfresh](https://github.com/blue-yonder/tsfresh): Time series transformation, feature extraction and selection
8. [cesium](https://github.com/cesium-ml/cesium): Time series feature extraction
### Machine Learning
1. [sklearn](https://scikit-learn.org): a comprehensive and all-around machine learning package
2. [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html): imbalanced learning API
3. [photon](https://github.com/photon-team/photon): high level machine learning pipelines
4. [shap](https://github.com/slundberg/shap): analyze and visualize shapley values
5. [lightgbm](https://lightgbm.readthedocs.io): super efficient implementation of gradient boosting trees
6. [tensorflow](https://www.tensorflow.org/): Google's generic, graph-based deep learning package
7. [keras](https://keras.io/): A high level deep learning interface, can use tensorflow as backend
8. [pytorch](https://pytorch.org/): A pythonic-style, generic deep learning package. Very easy to use
9. [fast.ai](https://github.com/fastai/fastai): A library to train fast and accurate neural nets. Based on pytorch
10. [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning): A lightweight pytorch wrapper
11. [paddlepaddle](https://github.com/PaddlePaddle/Paddle): Baidu's generic deep learning framework
12. [spacy](https://spacy.io/): Excellent industrial level NLP package. Easy to use and very efficient
13. [ernie](https://github.com/PaddlePaddle/ERNIE): Baidu's NLP package based on PaddlePaddle. Best 中文 NLP.
14. [gym](https://github.com/openai/gym): OpenAI's framework for reinforcement learning
15. [baselines](https://github.com/openai/baselines): OpenAI's high quality implementations of famous reinforcement learning algorithms
16. [dopamine](https://github.com/google/dopamine): Google's framework for reinforcement learning
17. [trfl](https://github.com/deepmind/trfl): deepmind's framework for reinforcement learning
18. [tensorforce](https://github.com/tensorforce/tensorforce): TensorFlow based reinforcement learning framework, bundled with models
19. [tianshou](https://github.com/thu-ml/tianshou): Tsinghua's super fast framework for reinforcement learning based on Pytorch
20. [warpdrive](https://github.com/salesforce/warp-drive): Extremely fast end-to-end deep multi-agent reinforcement learning engine
21. [MorphNet](https://github.com/google-research/morph-net): Google Research method to reduce neural network size while improving performance
22. [dalib](https://github.com/thuml/Transfer-Learning-Library): Tsinghua's transfer learning library based on pure Pytorch
23. [PyG](https://github.com/rusty1s/pytorch_geometric): Very efficient pytorch based GNN library
24. [DGL](https://www.dgl.ai/): Easy deep learning on graphs
25. [Plato](https://github.com/Tencent/plato): Tencent graph library
26. [CogDL](https://github.com/THUDM/cogdl): An extensive research toolkit for deep learning on graphs by TsingHua
27. [dgl-ke](https://github.com/awslabs/dgl-ke): Fast and scalable knowledge graph algorithms by Amazon AWS
28. [pyro](http://docs.pyro.ai/en/stable/): Deep Universal Probabilistic Programming
29. [interpretML](https://github.com/interpretml/interpret): Interpretable Machine Learning by Microsoft
30. [apex](https://github.com/NVIDIA/apex): A PyTorch extension for mixed precision and distributed training
31. [pycaret](https://github.com/pycaret/): low-code machine learning wrapper for many packages
### AutoML
1. [bohb](https://github.com/automl/HpBandSter): Hyper-parameter optimization with Bayesian method and bandit-based method combined
2. [auto-sklearn](https://github.com/automl/auto-sklearn): Automatically builds ensemble of sklearn models and optimizes hyper-parameters
3. [TransmogrifAI](https://github.com/salesforce/TransmogrifAI): Salesforce AutoML library
4. [BoTorch](https://github.com/pytorch/botorch): Bayesian optimisation based on PyTorch
5. [AutoGluo](https://github.com/awslabs/autogluon): Amazon's AutoML toolkit
### Visualization
1. [tqdm](https://github.com/tqdm/tqdm): A python progressbar
2. [matplotlib](https://matplotlib.org/): All-purpose plotting package
3. [seaborn](https://seaborn.pydata.org/): A matplotlib-based high level visualization interface
4. [SandDance](https://github.com/microsoft/SandDance): Microsoft's visualization tool
5. [jupyter](jupyter.org/): Interactive python notebook. Excellent for machine learning research
6. [jupytext](https://github.com/mwouts/jupytext): Manage Jupyter notebooks as plain text documents
7. [nbdev](https://github.com/fastai/nbdev): Fast.Ai's Jupyter notebook which can export to script
8. [jupyterlab/debugger](https://github.com/jupyterlab/debugger): Visual debugger for JupyterLab
9. [handcalcs](https://github.com/connorferster/handcalcs): Render native python expressions into Latex formula
10. [hiddenlayer](https://github.com/waleedka/hiddenlayer): A light-weight tool to visualize neural net graphs and training processes. It supports tensorflow, keras and pytorch
11. [VisualDL](https://github.com/PaddlePaddle/VisualDL): Visualization of neural network data, structure and training. By Baidu PaddlePaddle, support Pytorch
12. [manifold](https://github.com/uber/manifold): A model-agnostic debugging and visulisation tool for deep learning
13. [sweetviz](https://github.com/fbdesignpro/sweetviz): Automated dataset analysis and visualization
### Other
1. [PySnooper](https://github.com/cool-RR/PySnooper): a generic but powerful python debug tool
2. [DiffTaichi](https://github.com/yuanming-hu/difftaichi): Differentiable Programming for Physical Simulation (ICLR 2020)
3. [Machine Learning Toolbox](https://amitness.com/toolbox): A rather complete list of packages for all-phase machine learning
## Datasets
This session lists publicly available datasets or data tools
### General
1. [Google Dataset Search](https://toolbox.google.com/datasetsearch)
2. [Yahoo Datasets](https://webscope.sandbox.yahoo.com/#datasets)
3. [Kaggle Datasets](https://www.kaggle.com/datasets)
4. [Reddit Datasets](https://www.reddit.com/r/datasets/)
5. [Google Trends](https://trends.google.com/trends/explore)
6. [VisualData](https://www.visualdata.io/): 281 Computer Vision Datasets
7. [Open Images](https://storage.googleapis.com/openimages/web/index.html): Google's image dataset, with > 9 million images
8. [Datasets for Machine Learning](https://www.datasetlist.com/): 100+ Most famous machine learning datasets
9. [TensorFlow Datasets](https://www.tensorflow.org/datasets): collection of datasets ready to use with TensorFlow
10. [Awesome Python Login Model](https://github.com/CriseLYJ/awesome-python-login-model): Examples to scrape data from websites with login requirements
11. [magi.com](https://magi.com/): A powerful knowledge graph search engine, supports Chinese.
### Economics & Finance
1. [Quandl Financial Datasets](https://www.quandl.com/)
2. [Yahoo Finance](https://finance.yahoo.com/)
3. [Open Corporates](https://opencorporates.com/)
4. [US Census Bureau](https://www.census.gov/)
5. [US Government Data](https://www.data.gov/)
6. [US Bureau of Labor Statistics](https://www.bls.gov/data/)
7. [Eurostat Database](https://ec.europa.eu/eurostat/data/database)
8. [EU Open Data](https://data.europa.eu/euodp/en/data)
9. [UK Government Data](https://data.gov.uk/)
### Geography
1. [Google Satellite Image Datasets](https://cloud.google.com/storage/docs/public-datasets/)
2. [World Factbook](https://www.cia.gov/library/publications/the-world-factbook/)
3. [Nasa Earth Data](https://search.earthdata.nasa.gov/)
4. [NOAA Weather Data](https://www.ncdc.noaa.gov/data-access)
### Healthcare
1. [Health data](https://healthdata.gov/)
2. [CDC Data](https://www.cdc.gov/datastatistics/index.html)
### Social Study
1. [GroupLens Datasets](https://grouplens.org/datasets/)
2. [BuzzFeed News](https://github.com/BuzzFeedNews)
## Books
This session lists publicly downloadable machine learning books.
1. [Probabilistic and Statistical Models for Outlier Detection](https://www.springer.com/cda/content/document/cda_downloaddocument/9783319475776-c1.pdf?SGWID=0-0-45-1597574-p180317591) by Charu Aggarwal
2. [Foundations of Data Science](http://www.cs.cornell.edu/jeh/book.pdf) by Avrim Blum, John Hopcroft and Ravindran Kannan
3. [Pattern Recognition and Machine Learning](https://github.com/ctgk/PRML) by Christopher Bishop
4. [集成学习：基础与算法](https://re.jd.com/cps/item/12912974.html?cu=true&utm_source=read678.com&utm_medium=tuiguang&utm_campaign=t_322302035_&utm_term=b8e378159373482eb04e19cd57beff11) by 周志华
5. [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio and Aaron Courville
6. [Neural Network and Deep Learning](https://github.com/nndl/nndl.github.io) by Xipeng Qiu 邱锡鹏
7. [Machine Learning Yearning](https://www.mlyearning.org/) by Andrew Ng
8. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) by Daniel Jurafsky and James Martin
9. [Notes on Deep Learning for NLP](https://arxiv.org/abs/1808.09772) by Antoine J.-P. Tixier
10. [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) by Christopher Manning, Prabhakar Raghavan and Hinrich Schütze
11. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) by Richard Sutton and Andrew Barto
12. [Deep Reinforcement Learning](https://arxiv.org/abs/1810.06339) by Yuxi Li
13. [Network Science](http://networksciencebook.com) by Albert-László Barabási
14. [知识图谱与深度学习](https://re.jd.com/cps/item/12661347.html?cu=true&utm_source=www.read678.com&utm_medium=tuiguang&utm_campaign=t_322302035_&utm_term=f83033b51c2b46ed84ac7c4a79ce2e5c) by 孙茂松 and 刘知远
15. [人工智能：知识图谱前沿技术](https://item.jd.com/12911290.html) by 朱小燕
16. [迁移学习](https://item.jd.com/12930984.html?extension_id=eyJhZCI6IiIsImNoIjoiIiwic2hvcCI6IiIsInNrdSI6IiIsInRzIjoiIiwidW5pcWlkIjoie1wiY2xpY2tfaWRcIjpcIjUzYjEzYmIyLWMzOWItNGE0MC05NDUzLTYyZmVlYTU0OGNhOVwiLFwicG9zX2lkXCI6XCIyNjE3XCIsXCJzaWRcIjpcIjA5NjM2NDA0LWIzNWItNDljMS04ZDlhLWEzZTZmOWQyN2E0YVwiLFwic2t1X2lkXCI6XCIxMjkzMDk4NFwifSJ9&jd_pop=53b13bb2-c39b-4a40-9453-62feea548ca9) by 杨强 张宇 戴文渊 潘嘉林
17. [Lifted Probabilistic Inference](https://mitpress.mit.edu/books/introduction-lifted-probabilistic-inference) by Guy Van den Broeck, Kristian Kersting, Sriraam Natarajan and David Poole
18. [AutoML: Methods, Systems, Challenges](https://www.automl.org/book/) edited by Frank Hutter, Lars Kotthoff and Joaquin Vanschoren
19. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar
20. [Limitations of Interpretable Machine Learning](https://compstat-lmu.github.io/iml_methods_limitations/)
21. [Evolutionary Learning: Advances in Theories and Algorithms](https://link.springer.com/book/10.1007/978-981-13-5956-9) by Zhi-Hua Zhou， Yang Yu and Chao Qian
## Resources
This session includes online resources such as websites, video courses, code samples etc.
1. [Advanced Numpy](http://www.scipy-lectures.org/advanced/advanced_numpy/)
2. [Mathematical Optimization with Scipy](http://www.scipy-lectures.org/advanced/mathematical_optimization/index.html)
3. [Curve Fitting Explained](https://www.explainxkcd.com/wiki/index.php/2048:_Curve-Fitting)
4. [Time Series Analysis in Python](http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016)
5. [Introduction to Machine Learning](https://scikit-learn.org/stable/user_guide.html) by sklearn
6. [Awesome Jupyter](https://github.com/markusschanta/awesome-jupyter): All Jupyter resources
7. [Top 50 matplotlib Visualizations – The Master Plots](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/)
8. [Stanford CS229 Machine Learning Notes](https://github.com/afshinea/stanford-cs-229-machine-learning)
9. [Stanford CS230 (Andrew Ng) Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230.html)
10. [Stanford CS224n NLP Notes](https://zhuanlan.zhihu.com/p/31977759): Notes are in Chinese
11. [Stanford CS234 Reinforcement Learning](http://web.stanford.edu/class/cs234/schedule.html)
12. [Stanford CS224w Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
13. [Stanford CS520 Knowledge Graphs](https://web.stanford.edu/class/cs520/)
14. [Fast.ai Online Course](https://course.fast.ai/): Extremely useful and practical
15. [Fast.ai Code-First Introduction to NLP](https://www.fast.ai/2019/07/08/fastai-nlp/): Comprehensive NLP course by fast.ai
16. [Fast.ai Self Supervised Learning](https://www.fast.ai/2020/01/13/self_supervised/)
17. [Andrew Ng Course Notes](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes): Notes are in Chinese
18. [Reinforcement Learning Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0)
19. [李宏毅 deep learning channel](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ)
20. [TensorFlow Course](https://github.com/open-source-for-science/TensorFlow-Course#why-use-tensorflow)
21. [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
22. [TensorFlow Hub](https://www.tensorflow.org/hub/): Reusable machine learning models
23. [TensorFlow Cookbook](https://github.com/nfmcclure/tensorflow_cookbook)
24. [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization): Provide a suite of tools for optimizing tf models
25. [Awesome Pytorch Resources](https://github.com/bharathgs/Awesome-pytorch-list): Everything you need with PyTorch is here
26. [PyTorch Hub](https://pytorch.org/hub): Pre-trained PyTorch models by facebook
27. [PyTorch Cookbook](https://zhuanlan.zhihu.com/p/59205847?): Useful PyTorch code snippets. Article in Chinese
28. [TorchShard](https://github.com/KaiyuYue/torchshard): Light weight lib to reduce GPU memory cost when using PyTorch
29. [Machine Learning Mastery](https://machinelearningmastery.com/)
30. [Towards Data Science](https://towardsdatascience.com/)
31. [Google AI Blog](https://ai.googleblog.com/)
32. [OpenAI Blog](https://blog.openai.com/)
33. [Tsinghua NLP](https://github.com/thunlp): NLP Lab at Tsinghua University
34. [Papers with Code](https://paperswithcode.com/): Extensive latest papers with implementations
35. [Deep Learning Ocean](https://github.com/machinelearningmindset/deep-learning-ocean): Papers, books, datasets, courses, blogs, frameworks
36. [stateoftheart.ai](stateoftheart.ai): Summary of state-of-the-art AI models
37. [80 papers with notes](https://github.com/yassouali/ML_paper_notes)
38. [Andrej Karpathy AI Blog](http://karpathy.github.io/)
39. [Sebastian Ruder Blog](http://ruder.io/): Deepmind Research Scientist blog about Deep Learning and NLP
40. [Andrew Ng DeepLearning.ai Blog](https://www.deeplearning.ai/blog/category/resources/)
41. [Deep Learning Models](https://github.com/rasbt/deeplearning-models)
42. [Segmentation Models](https://github.com/qubvel/segmentation_models): Both pytorch and keras implementations available
43. [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
44. [NAACL Tutorial on NLP Transfer Learning](https://github.com/huggingface/naacl_transfer_learning_tutorial)
45. [27 Pre-trained NLP SOTA Models](https://github.com/huggingface/pytorch-transformers)
46. [Everything about 中文 NLP](https://github.com/crownpku/Awesome-Chinese-NLP)
47. [中文 NLP Corpus](https://github.com/brightmart/nlp_chinese_corpus)
48. [中文分词SOTA](https://github.com/SVAIGBA/WMSeg)
49. [Deep Reinforcement Learning Hands On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On): Implementations of many deep RL algorithms with real-world problems.
50. [Deep Reinforcement Learning Algorithms](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
51. [Graph Neural Network Papers](https://github.com/benedekrozemberczki/awesome-graph-classification)
52. [Must Read GNN Papers](https://github.com/thunlp/GNNPapers)
53. [Awesome resources on Graph Neural Networks](https://github.com/nnzhan/Awesome-Graph-Neural-Networks)
54. [Few-Shot Learning Papers](https://github.com/tata1661/FewShotPapers)
55. [CleanLab](https://github.com/cgnorthcutt/cleanlab): Learning with Noisy Labels
56. [Understanding NN](https://github.com/1202kbs/Understanding-NN?files=1): Interpretable neural networks
57. [xg2xg](https://github.com/jhuangtw-dev/xg2xg): Very useful summary of data science IT tool packages
## Articles
This session lists interesting articles such as blogs or research papers.
- [Hacker Laws](https://github.com/dwmkerr/hacker-laws)
- [Supercharging Jupyter Notebooks](https://towardsdatascience.com/supercharging-jupyter-notebooks-e22f5ad7ca18)
- [7 things to quickly improve your Data Analysis in Python](https://towardsdatascience.com/7-things-to-quickly-improve-your-data-analysis-in-python-3d434243da7)
- [时间序列预测方法总结](https://zhuanlan.zhihu.com/p/67832773)
- [Initializing Neural Networks](http://www.deeplearning.ai/ai-notes/initialization/)
- [Activation Functions Explained](https://mlfromscratch.com/activation-functions-explained/#/)
- [How to Handle Imbalanced Classification Problems in Machine Learning?](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
- [How to Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)
- [Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)
- [Deep Learning Parameter Tuning Tricks (Chinese)](https://www.zhihu.com/question/41631631)
- [深度学习debug常见问题]()https://zhuanlan.zhihu.com/p/158739701)
- [Pruning Techniques for Neural Networks](https://heartbeat.fritz.ai/research-guide-pruning-techniques-for-neural-networks-d9b8440ab10d)
- [A Developer-Friendly Guide To Mixed Precision Training With PyTorch](https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam)
- [juliuszh机器学习札记](https://zhuanlan.zhihu.com/juliuszh)
- [22张深度学习精炼图笔记总结](https://zhuanlan.zhihu.com/p/152362317)
- [TensorFlow Model Optimization Toolkit - Pruning API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)
- [4 Ways to Debug Your Deep Neural Network](https://blog.cardiogr.am/4-ways-to-debug-your-deep-neural-network-e5edb14a12d7)
- [神经网络训练trick](https://zhuanlan.zhihu.com/p/59918821)
- [代码详解：用Pytorch训练快速神经网络的9个技巧](https://zhuanlan.zhihu.com/p/78555755)
- [Efficient PyTorch](https://towardsdatascience.com/efficient-pytorch-part-1-fe40ed5db76c)
- [Pytorch有什么节省显存的小技巧?](https://www.zhihu.com/question/274635237)
- [PyTorch Cookbook（常用代码段整理合集）](https://zhuanlan.zhihu.com/p/59205847)
- [Training a NN to 99% accuracy on MNIST in 0.76 seconds](https://github.com/tuomaso/train_mnist_fast): 8 steps to optimize pytorch training end-to-end
- [Troubleshooting Deep Neural Networks](https://www.zhihu.com/question/25097993/answer/585714651): in Chinese
- [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
- [CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more](https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)
- [深度学习 cnn trick合集](https://zhuanlan.zhihu.com/p/137940586)
- [Image Segmentation: Tips and Tricks from 39 Kaggle Competitions](https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Tips for Training Recurrent Neural Networks](https://danijar.com/tips-for-training-recurrent-neural-networks/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [注意力机制综述](https://zhuanlan.zhihu.com/p/106662375)
- [浅谈Attention机制的理解](https://zhuanlan.zhihu.com/p/35571412)
- [nlp中的Attention注意力机制+Transformer详解](https://zhuanlan.zhihu.com/p/53682800)
- [Self Supervised Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
- [NLP Overview](https://nlpoverview.com/)
- [Introduction to Natural Language Processing](https://blog.algorithmia.com/introduction-natural-language-processing-nlp/)
- [8 Excellent Pretrained Models to get you Started with NLP](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)
- [The Best and Most Current of Modern Natural Language Processing](https://mc.ai/the-best-and-most-current-of-modern-natural-language-processing/)
- [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [Automating Data Augmentation: Practice, Theory and New Direction](http://ai.stanford.edu/blog/data-augmentation/)
- [12个NLP预训练模型的学习笔记](https://zhuanlan.zhihu.com/p/111738085)
- [超详细中文预训练模型ERNIE使用指南](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/97703165)
- [NLP民工的乐园: 几乎最全的中文NLP资源库](https://github.com/fighting41love/funNLP)
- [莫愁前路，一文读懂中文语料预处理](https://zhuanlan.zhihu.com/p/166199114)
- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)
- [强化学习路在何方？](https://zhuanlan.zhihu.com/p/39999667)
- [GAN，没错，生成对抗网络也可以被用于时间序列预测](https://zhuanlan.zhihu.com/p/121771459)
- [Advances in Few Shot Learning: A Guided Tour](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
- [Few Shot Learning 小白入门笔记](https://zhuanlan.zhihu.com/p/396593319)
- [Few Shot Learning 综述](https://zhuanlan.zhihu.com/p/290011942)
- [贝叶斯与元学习](https://zhuanlan.zhihu.com/c_1186629504699731968)
- [Logging in Tensorboard with PyTorch](https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e)
- [Analyze Hinton's Capsule Plan](https://zhuanlan.zhihu.com/p/29435406): Article in Chinese
- [Introduction to Knowledge Graph](https://zhuanlan.zhihu.com/p/53753234): Article in Chinese
- [知识图谱从哪里来：实体关系抽取的现状与未来](https://zhuanlan.zhihu.com/p/91762831): Article in Chinese
- [少样本关系抽取技术](https://zhuanlan.zhihu.com/p/159438322)
- [漫谈图神经网络](https://zhuanlan.zhihu.com/p/108294485)
- [A Gentle Introduction to Graph Neural Networks](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)
- [Expressive power of graph neural networks and the Weisfeiler-Lehman test](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49)
- [Towards Explainable Graph Neural Networks](https://towardsdatascience.com/towards-explainable-graph-neural-networks-45f5e3912dd0)
- [带噪学习和协作学习，不完美场景下的神经网络优化策略](https://zhuanlan.zhihu.com/p/141085295)
- [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
