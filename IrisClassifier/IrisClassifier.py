import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris();
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["target"]=iris.target
df["species"]=df["target"].apply(lambda x:iris.target_names[x])
print(df.head(100))