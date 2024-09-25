import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# データの読み込み
data = pd.read_csv("node.csv")

print(data)

# Label Encoderで品種と産地を数値に変換
le_variety = LabelEncoder()
le_region = LabelEncoder()

data["Variety_Encoded"] = le_variety.fit_transform(data["Variety"])
data["Region_Encoded"] = le_region.fit_transform(data["Region"])

print(data)

# 特徴ベクトル（品種と産地）の作成
features = data[["Variety_Encoded", "Region_Encoded"]].values

print(features)

# 特徴ベクトル間のユークリッド距離を計算
distance_matrix = euclidean_distances(features, features)

print(distance_matrix)

# 距離の最大値で正規化して関連度スコアを計算（1から距離を引くことで関連度を逆転）
similarity_scores = 1 - distance_matrix / distance_matrix.max()

# X, Y座標として各点の位置を決定
x_positions = similarity_scores[:, 0]
y_positions = similarity_scores[:, 1]

# Z軸は評価値
z_positions = data["Rating"]

# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(x_positions, y_positions, z_positions, cmap="Greys", edgecolor="none")

ax.set_xlabel("Variety Similarity")
ax.set_ylabel("Region Similarity")
ax.set_zlabel("Rating")

plt.show()
