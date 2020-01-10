---
title: About Me
permalink: /about/
---

# YuKai Wu 
### email: ywu048@fiu.edu 
### EDUCATION
- 2009.09—2013.07 AnQing Normal University (bachelor degree), major Information and Computing Science 
- 2018.05—2019.12 Florida International University (master degree), major computer science 

### PROFESSIONAL EXPERIENCE
- ##### 2015.9 – 2018.1   HangZhou ShuYun Information Science Corporation (Algorithm Engineer)
  - **Design and Development of Recommend System for commodities**  
    Develop a framework for Recommendation System, It can train model from users features data and generates a list of recommendation commodities. In this project, My duty is to develop a back-end part that can get data from several recommendation list and merge them by specific algorithm, final push them to user end. I’m also in charge with developing some recommendation algorithms suit for a particular scenario.  This recommendation framework mainly serves the advertising business.
  - **Design and Development of TaoBao, TMall User Portrait System**  
    Depend on user shopping records, I establish a label system for each user to represent user feature space. Such as Preference for Clothes, Preference for Cosmetics, Preference for Food  etc. This system can easily group users into several category and improve the precise of recommendation system.  
  - **Analysis user comment data with NLP**  
    The company need extract short label from user comment to help our customer well understand what is the user feeling about their product. I use RNN with attention model to extract the key information in the short comment, and then analyse these data and display the result. 

- ##### 2015.6 – 2015.8	   ShangHai YiChen Information Science Corporation (Big Data Development Engineer)
  -  **Design and Development of Recommendation System for short videos**  
   We store user’s history data in Hive, and then we train the Recommendation Models with Spark and store them in HDFS. We develop a frame that automatically deploy the trained models that load from HDFS, and recommend short videos for users, putting the result into Hbase. The frame have RESTful API for responding the request from other front-end programme.     

- ##### 2014.5 – 2015.5	    ANHUI XiangXing Information Science Corporation (Data Mining Engineer)
  - **Development of Data Mining System Based on Spark Framework**  
  The company want to develop a cloud computing platform for the enterprise to help them analyse huge amount of data. So our task is to integrate Spark into our System, and extend the function and algorithm of MLlib. 
  - **Data Modeling of Customer Loss with Anhui Telecom User Data**  
  Help our customer to build a ML model to predict whether the user would lost or not in the next mouth.  The Telecom company give us lots of user  behavior data without sensitive part.  The label is binary, meaning whether the user would lost in next month.  We use previous four month data to extract the user’s feature, and then input the features into model to predict the user situation in next month.
 
### RESEARCH / FIELD WORK EXPERIENCE
- [**Reinforcement Learning Project:**](https://github.com/wyk2796/reinforcement_learning)
  - **Navigation**: Training a agent to learn how to get maximum bananas in one episode,  using the Deep Q-Network(DQN) Algorithm. 
  - **Reacher**: Training a robot arm to quickly catch a target.  In this project, I use the Deep Deterministic Policy Gradient(DDPG) with continuous action space to train 20 virtual agents simultaneously to reduce the training time.  
  - **Tennis**: Two players compete each other. The project is training two agents to play the tennis game and against each other. I learn how to use the DDPG to train multi-agents in competitive environment.
  - **Half Field Offense in [Robocup 2D Soccer](https://github.com/LARG/HFO)**: Training a agent to kick the ball to the goal. This environment action space is parameterized action space. It combine the discrete value and the continuous value. 

- [**NLP Project:**](https://github.com/wyk2796/NLP_learning)
We use four kind of algorithms to complete Name Entity Recognition(NER) task and compare them. [(paper)](https://github.com/wyk2796/NLP_learning/blob/master/doc/ner.pdf)

- [**Image Process:**](https://github.com/wyk2796/image_processes_style_transform)
For this project, our target is to display the intermediate processes of image-style transformation, how to transform a photo from a original photo to a style photo and what the processes look like. We use VGG19 and transform net (Johnson, J. et al. 2016) to train model for image-style transformation. In each layer of transform net, we use DeconvNet to get visible image from intermediate data to display the changing process. [(report)](https://github.com/wyk2796/image_processes_style_transform/blob/master/The%20visualization%20of%20Deep%20Neural%20Network.pdf)

- **Distributed Framework:**
It is a distributed system framework and has a master and various pluggable components. The pluggable components can design to divers function components. The master manage all components. The system has a pub-sub system to support communication between all components. The component can publish and subscribe topics. This project based on AKKA concurrency framework with Scala programm language. Here is a Distributed Web Crawler based on this distributed framework,  **["crawlnet".](https://github.com/wyk2796/crawlnet)**

- **Recommend System:**
The project target is to design and develop a recommend system to recommend commodity for users.  Our project’s architecture has four part. We use Hive as our data warehouse. It store all the order, user and commodity data. We use Spark to generate modes that used to recommend commodity and store the result into Hbase database. And then we design restful API to receive requests and return the recommendation commodities for users.   

### PROFESSIONAL SKILLS
- **Deep learning** development skills:
  - **Tensorflow**, **pyTorch** and relevant **Python lib**, Such as **Numpy**, **Pandas** etc.
  - **Reinforcement learning**: 
    - Get the Nanodegree in Udacity online course. 
    - **MDP**, **DQN**, **DDPG**, **PPO**, **A2C**,**A3C**, . 
  - **NLP**
    - **word2vec**, **RNN**, **LSTM**, **seq2seq**, and **attention model**.
  - **Image Process**
    - **CNN**, **VGG**, **ResNet**, **DeconvNet**, **Image Style_Transfer**
- **Big Data** development skills:
  - **Spark** distributed computing system, experience for dealing with **large-scale data sets** and the **spark-mllib** machine learning algorithm library.
  - **Recommendation System:**
    - **Framework** design and development
    - Design **Recommendation Algorithm**
    - Design **User Portrait**  
  - **AKKA concurrency framework**, and **Real-time stream processing** with Spark.  
  - **Big Data** related systems, such as: **Hadoop**, **Spark**, **Hive**, **Hbase**, **Kafka**, **Elasticsearch** etc.
  - Relational database **Mysql** and non-relational database, such as : **HBase**, **Mongodb** and **Redis**
- **Development Skills**:
  - Programming language **Scala**, **Python**, **Java**, **C/C++**, **SQL**, etc.
  - Tools: **git**, **vscode**, **vs**, **JetBrains**, **sbt**, **Maven**, **CMake**,   **Amazon Web Services(AWS)** 

### CERTIFICATE
[Udacity Nanodegree:](https://confirm.udacity.com/3QPL96K)

### WEBSITE
[Github](https://github.com/wyk2796), [Linkedin](https://www.linkedin.com/in/yukai-wu-b50ba7b8)
