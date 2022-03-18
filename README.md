# Activation-Functions-NLP
<i>Final Senior Intensive Project</i>

Activation functions are often overlooked the tuning of hyperparameters in neural networks. Most deep learning researchers and engineers simply use the most popular activation functions, such as ReLU, without much consideration. In my senior capstone project, I investigated the performance of five different activation functions (ReLU, Swish, Mish, TAct, and mTAct) in many different neural network architectures in the context of image classification. In this project, I continue the investigation into the performance of activation functions -- but now in the context of neural network architectures designed for NLP. The main task is simple sentiment analysis with the BERT architecture using an IMDb movie review dataset. 


## Installation

In this repository, there are 3 different architectures: BERT (root level), GPT-2 (`gpt-2` directory), and a reduced version of GPT-2 (`fine-grained-sentiment` directory). To run any of these, first navigate to the corresponding directory, create a Python virtual environment, and then load the necessary dependencies:


First, set up virtual environment and install from ```requirements.txt```:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
Deactivate the virtual environment: `deactivate`

Re-activate the existing virtual environment:

    source venv/bin/activate
    
Within the virtual environment, run the Python module with this command:
`python activation_functions_in_transformer_model.py`
Feel free to experiment with the different activation functions for yourself by replacing the value fo "gelu" in line 294 with any of the five activation functions defined above (ReLU, Swish, Mish, TAct, and mTAct).


## GPT-2

The submodules use variations of the GPT-2 architecture. The dataset is a different movie review dataset than the IMDb one used for BERT testing: SST-5 (Stanford Sentiment Treebank). Each submodule has its own set of instructions in their respective directories.
