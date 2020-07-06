from flask import Flask, jsonify, request, render_template, json
import pandas as pd
import numpy as np
from glob import glob
from ast import literal_eval
from tqdm import tqdm_notebook
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from pythainlp.ulmfit import *
from pythainlp.tokenize import Tokenizer
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
app = Flask(__name__)
print(Flask(__name__))

model_path = '/home/az2-user/model' 
data_cls = load_data(model_path, "/home/az2-user/model/data_cls.pkl")
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
learn.load('/home/az2-user/model/model_ml')
def sentiment_convert(p_val):
        if str(p_val) == "1":
            return "pos"
        elif str(p_val) == "2":
            return "neg"
        elif str(p_val) == "3":
            return "neu"
        elif str(p_val) == "4":
            return "q"
        else:
            return "ERROR: Unknown Type"

@app.route('/')
def index():
    return render_template('test.html')
@app.route('/test')
def test():
   text = request.args.get('getinput')
   predicted = learn.predict(text)
   p_val = sentiment_convert(predicted[0])
   resp = {
       "getinput" : text,
 #      "status":200,
       "message":"OK",
       "data":{
            "text": text,
            "mood": p_val,
        }
   }
   return json.dumps(resp, ensure_ascii=False)