#Ryan Karl

#My Python 2.7.15 environment contains the following packages:
#Keras	2.2.0
#Keras-Applications	1.0.2
#Keras-Preprocessing	1.0.1
#Markdown	3.0.1
#Pillow	5.4.1
#PyWavelets	1.0.1
#PyYAML	3.13
#TBB	0.1
#Werkzeug	0.14.1
#absl-py	0.7.0
#anytree	2.7.0
#astor	0.7.1
#cloudpickle	0.8.0
#cycler	0.10.0
#dask	1.1.1
#decorator	4.3.2
#gast	0.2.2
#graphviz	0.13
#grpcio	1.18.0
#h5py	2.9.0
#kiwisolver	1.0.1
#matplotlib	3.0.2
#mock	2.0.0
#networkx	2.2
#numpy	1.16.1
#opencv-python	4.0.0.21
#pandas	0.25.1
#pbr	5.1.2
#pip	19.1.1
#protobuf	3.6.1
#pyparsing	2.3.1
#python-dateutil	2.8.0
#pytz	2019.2
#scikit-image	0.14.2
#scikit-learn	0.20.2
#scipy	1.2.1
#seaborn	0.9.0
#setuptools	41.0.1
#six	1.12.0
#sklearn	0.0
#tensorboard	1.12.2
#tensorflow	1.13.0rc2
#tensorflow-estimator	1.13.0rc0
#termcolor	1.1.0
#toolz	0.9.0
#wheel	0.33.4

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier#, export_graphviz
import graphviz

df = pd.read_csv('data-film.csv')

df_mod = df.copy()
targets = df_mod["GENRE"].unique()
map_to_int = {name: n for n, name in enumerate(targets)}
df_mod["Target"] = df_mod["GENRE"].replace(map_to_int)
features = list(df_mod.columns[1:5])

Y = df_mod["Target"]
X = df_mod[features]
clf = DecisionTreeClassifier()
clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("clf")
