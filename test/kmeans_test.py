
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def kmeans_test():
    num_vectors = 1000
    num_clusters = 3
    num_steps = 100
    vector_values = []

    for i in xrange(num_vectors):
        if np.random.random() > 0.5:
            vector_values.append([np.random.normal(0.5, 0.6), np.random.normal(0.3, 0.9)])
        else:
            vector_values.append([np.random.normal(2.5, 0.4), np.random.normal(0.8, 0.5)])

    df = pd.DataFrame({"x": [v[0] for v in vector_values], "y": [v[1] for v in vector_values]})
    sns.lmplot("x", "y", data=df, fit_reg=False, size=7)
    plt.show()
    return