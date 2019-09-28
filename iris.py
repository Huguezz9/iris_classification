import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#importing the dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# set the dataset to a varible
iris = load_iris()

# print(iris.data) # prints arrays with each row of data in the dataset.

features = iris.data.T 

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

plt.scatter(sepal_length, sepal_width, c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
# print(X_new.shape)

# prediction = knn.predict(X_new)
# print(prediction)


# print(knn.score(X_test, y_test))
# 97% of accuracy
