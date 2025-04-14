# === File: hmm_model.py ===
from hmmlearn.hmm import GaussianHMM
import numpy as np

def add_hmm_states(data, n_states=3):
    X = data[['Return']].values
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(X)
    data['HMM_state'] = model.predict(X)
    return data, model
