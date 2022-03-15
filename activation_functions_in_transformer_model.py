

"""---

# NEW APPROACH
"""

# # Install packages needed
# import sys
# import subprocess

# # implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install',
# 'datasets'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install',
# 'transformers'])

# # process output with an API in the subprocess module:
# reqs = subprocess.check_output([sys.executable, '-m', 'pip',
# 'freeze'])
# installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

# print(installed_packages)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam,lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import datasets
import transformers


"""## Custom Activation Functions"""

def f_mtact(input, alpha, beta, inplace = False):
    '''
    Applies the mtact function element-wise:
    mtact(x) = ----
    '''
    A = 0.5*(alpha**2)
    B = 0.5 - A
    #B=(1-alpha**2)/2
    #C = (1+beta**2)/2
    C = 0.5*(1+beta**2)

    return (A*input + B)*(torch.tanh(C*input)+1)

def f_tact(input, alpha, beta, inplace = False):
    '''
    Applies the tact function element-wise:
    tact(x) = ----
    '''
    A = 0.5*alpha
    B = 0.5 - A
    #B=(1-alpha)/2
    C = 0.5*(1+beta)

    return (A*input + B)*(torch.tanh(C*input)+1)

# implement class wrapper for mtact activation function
class mTACT(nn.Module):
    '''
    Applies the mTACT function element-wise:
    mtact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mtact()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = Parameter(torch.tensor(self.alpha,requires_grad=True))

        self.beta = beta
        self.beta = Parameter(torch.tensor(self.beta,requires_grad=True))

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mtact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)

# implement class wrapper for tact activation function
class TACT(nn.Module):
    '''
    Applies the TACT function element-wise:
    tact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> t = tact()
        >>> input = torch.randn(2)
        >>> output = t(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = Parameter(torch.tensor(self.alpha,requires_grad=True))

        self.beta = beta
        self.beta = Parameter(torch.tensor(self.beta,requires_grad=True))

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_tact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)

# !pip install datasets

from datasets import load_dataset

raw_datasets = load_dataset("imdb")

# !pip install transformers

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT model configuration """
from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/config.json",
    "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    # See all BERT models at https://huggingface.co/models?filter=bert
}


class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout



class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})])

# from transformers import BertModel
from transformers import TFBertModel

# Initializing a BERT configuration
configuration = BertConfig()

# Initializing a model configuration
bert_model = TFBertModel(configuration).from_pretrained("bert-large-uncased")

# Accessing the model configuration
# configuration = bert_model.config

def bert_encode(data,maximum_length) :
  input_ids = []
  attention_masks = []
  

  for i in range(len(data["text"])):
      encoded = tokenizer.encode_plus(
        
        data["text"][i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        # padding=True,
        # truncation=True,
        
        return_attention_mask=True,
        
      )
      
      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])
  return np.array(input_ids),np.array(attention_masks)

print(raw_datasets['train'])

len(raw_datasets["train"])



MAX_LENGTH = 512

train_input_ids, train_attention_masks = bert_encode(raw_datasets["train"][:1000], MAX_LENGTH)
test_input_ids, test_attention_masks = bert_encode(raw_datasets["test"][:1000], MAX_LENGTH)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
def create_model(bert_model):
  input_ids = tf.keras.Input(shape=(MAX_LENGTH,),dtype='int32')
  attention_masks = tf.keras.Input(shape=(MAX_LENGTH,),dtype='int32')
  
  output = bert_model([input_ids,attention_masks])
  output = output[1]
  output = tf.keras.layers.Dense(32,activation='relu')(output)
  output = tf.keras.layers.Dropout(0.2)(output)

  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
  model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = create_model(bert_model)
model.summary()



X_train = [train_input_ids, train_attention_masks]
Y_train = np.array(raw_datasets['train']['label'][:1000])
# Y_test first 5,000

print(type(raw_datasets['train']['label']))

print(type(raw_datasets['train']['label'][:5000]))

print(type(train_input_ids))

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=2, batch_size=50)

result = model.predict([test_input_ids,test_attention_masks])
result = np.round(result).astype(int)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()