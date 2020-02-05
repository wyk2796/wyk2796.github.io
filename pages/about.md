---
title: About Me
permalink: /about/
---

# YuKai Wu 
email: ywu048@fiu.edu 
call: 7866083426

### EDUCATION
- 2009.09—2013.07 AnQing Normal University (bachelor degree), major Computing & Info Science 
- 2018.05—2019.12 Florida International University (master degree), major computer science 

### PROFESSIONAL EXPERIENCE
- ##### 2015.9 – 2018.1   HangZhou ShuYun Information Science Corporation (Algorithm Engineer)
  - **Design and Development of Recommendation System for commodities**
    The company started to develop a framework for Recommendation System which can train model from users features data and generates a list of recommendation commodities. In this project, My duty is to develop a back-end part that can get data from several recommendation lists and merge them by specific algorithm, and then push them to user end. I’m also in charge with developing some recommendation algorithms for some particular scenarios.  This recommendation framework mainly serves the advertising business.
  - **Design and Development of TaoBao, TMall User Portrait System**  
    Depending on user shopping records, I establish a label system, representing user feature space, for all of them, such as Preference for Clothes, Preference for Cosmetics, Preference for Food etc. It is advantageous to divide group users into several categories and improve the precise of recommendation system.  
  - **Analysis user comment data with NLP**  
    The company needs to extract short labels from user comments to help our customers fully understand what the user feel about their product. I use RNN with attention model to extract the key information in the short comment, analyse these data and display the results. 

- ##### 2015.6 – 2015.8	   ShangHai YiChen Information Science Corporation (Big Data Development Engineer)
  -  **Design and Development of Recommendation System for short videos**  
  The company need to establish recommendation system for short videos. My work is to design and develop recommendation system. My design is to store user’s history data with Hive, train the Recommendation Models with Spark and storing those models in HDFS, use these models to generate recommendation sets and save them into Hbase, and has a service process that gets data from Hbase and responses the recommendation requests. I develop a framework that automatically deploys the trained models in HDFS and control the version of these models. The framework uses RESTful API to respond requests from other front-end programme.    

- ##### 2014.5 – 2015.5	    ANHUI XiangXing Information Science Corporation (Data Mining Engineer)
  - **Development of Data Mining System Based on Spark Framework**  
  The company creates a cloud computing platform for enterprise customers that helps companies analyze large amounts of business data. Therefore, my tasks is to integrate Spark into this System and extend the function and algorithm. 
  - **Data Modeling of Customer Loss with Anhui Telecom User Data**  
  Help our customer to build a ML model to predict whether the user would lost or not in the next mouth. The Telecom company gives me lots of user behavior data without sensitive part. The label is binary, meaning whether the user would lost in next month. I use previous four-month data to extract the user’s features, and then input the features into model to predict the user's situation in next month. In this project, I try many ML models and finally decide to use Random Forest that can reach the best precision and recall.   
 
### RESEARCH / PROJECT EXPERIENCE DURING UNIVERSITY 
- 2019 **Reinforcement Learning Project:** ([Git-Repo](https://github.com/wyk2796/reinforcement_learning)) **:**
  - **Navigation**: Training an agent to learn how to get maximum bananas in one episode,  using the Deep Q-Network(DQN) Algorithm. 
  - **Reacher**: Training a robot arm to quickly catch a target object. In this project, I use the Deep Deterministic Policy Gradient(DDPG) with continuous action space to train 20 virtual agents simultaneously to reduce the training time.  
  - **Tennis**: Two players compete with each other. The project is training two agents to play the tennis game. I use the DDPG to train multi-agents in competitive environment.
  - **Half Field Offense in [Robocup 2D Soccer](https://github.com/LARG/HFO)**: Training an agent to kick the ball to the goal. This environment action space is parameterized action space which combines the discrete value and the continuous value. 
<br>

- 2018 **NLP Project** ([Git-Repo](https://github.com/wyk2796/NLP_learning), [Paper](https://github.com/wyk2796/NLP_learning/blob/master/doc/ner.pdf)) **:**
We use four kinds of algorithms to complete Name Entity Recognition(NER) task and compare them. 

<br>

- 2019 **The Visualization of Transferring Processes of Image Style Transfer** ([Git-Repo](https://github.com/wyk2796/image_processes_style_transform), [Report](https://github.com/wyk2796/image_processes_style_transform/blob/master/The%20visualization%20of%20Deep%20Neural%20Network.pdf)) **:**
For this project, My target is to display the intermediate processes of image-style transformation, how to transform a photo from a original photo to a style photo and what the processes look like. I use VGG19 and transform-net (Johnson, J. et al. 2016) to train model for image-style transformation. In each layer of transform-net, I use DeconvNet to get visible image from intermediate data to display the changing process. 

<br>

- 2019 **Distributed Framework:**
It is a distributed system framework and has a master and various plumbable components. The plumbable components can be designed to divers function components. The master manages all components. The system has a pub-sub system to support communication between all components. The component can publish and subscribe topics. This project is based on AKKA concurrency framework with Scala programm language. Here is a Distributed Web Crawler based on this distributed framework,  **"crawlnet"** ([Git-Repo](https://github.com/wyk2796/crawlnet)).  

<br>  

- 2015 **Recommendation System** ([Git-Repo](https://github.com/wyk2796/recommender)) **:**
The project target is to design and develop a recommendation system to recommend commodity for users. The architecture of the project includes four parts. I use Hive as my data warehouse that stores all order, user and commodities data. I use Spark to generate recommendation models and use them to recommend commodities and store the result into Hbase database. And then we design restful API to receive requests and return the recommendation commodities for users. 

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
[Github](https://github.com/wyk2796), [Linkedin](https://www.linkedin.com/in/yukai-wu-b50ba7b8), [Personal Website](https://wyk2796.github.io)
