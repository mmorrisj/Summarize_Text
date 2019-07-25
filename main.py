# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:18:56 2019

@author: 589636
"""
from MatTools import build_summaries,build_labels,visualize_labels
import pandas as pd
import json

filename="Deep Learning results"
filetype='.json'
filepath='C:\\Users\\589636\\Documents\\Jupyter_Notebook\\Deep Learning\\'

with open(filepath+filename+filetype) as f: 
    text=json.load(f)

df=pd.DataFrame(text[400:600])

df=df.head(100)

summaries=build_summaries(df)

labels=build_labels(df)

df=df.merge(summaries,how='left',on='title',copy=False)

df=df.merge(labels,how='left',on='title',copy=False)

df.to_excel('./'+filename+'_output.xlsx')

visualize_labels(df)

print('Complete. Processed '+str(len(df))+' records.')