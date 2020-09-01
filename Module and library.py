#......import module and library.................
import os
import pandas as pd
import numpy as np

import json
import argparse


from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
