# Activation-Functions-NLP
Final Senior Intensive Project

Activation functions are often overlooked the tuning of hyperparameters in neural networks. Most deep learning researchers and engineers simply use the most popular activation functions, such as ReLU, without much consideration. In my senior capstone project, I investigated the performance of five different activation functions (ReLU, Swish, Mish, TAct, and mTAct) in many different neural network architectures in the context of image classification. In this project, I continue the investigation into the performance of activation functions -- but now in the context of neural network architectures designed for NLP. The task is simple sentiment analysis using a movie review dataset. 

In this repository, there are 3 different architectures: BERT (root level), GPT-2 (`gpt-2` directory), and a reduced version of GPT-2 (`fine-grained-sentiment` directory). To run any of these, first navigate to the corresponding directory, create a Python virtual environment, and then load the necessary dependencies:
