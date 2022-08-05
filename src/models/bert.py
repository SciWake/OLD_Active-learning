import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
import transformers

from transformers import BertModel, BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textwrap import wrap
from tqdm.notebook import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 777
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
