import torch
import pickle
import numpy as np

dir = 'query.pkl'
with open(dir, "rb") as file:
    data = pickle.load(file)
    print(np.shape(data['image']))