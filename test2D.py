# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
from NN import NN
from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

# Generate a dataset and plot it
np.random.seed(0)
n_classes = 3
X, y = make_classification(n_samples=250, n_classes=n_classes, n_features=2, n_informative=2, n_redundant=0, random_state=0, n_clusters_per_class=1)
#n_classes = 2
#X, y = sklearn.datasets.make_moons(250, noise=0.20)

test_fraction = 0.2
X, features_test, y, labels_test = cross_validation.train_test_split(X, y, random_state = 0, test_size = test_fraction)

Y = []

for i in range(len(y)):
    y_new = []

    for j in range(n_classes):
        y_new.append(0)

    y_new[y[i]] = 1
    Y.append(y_new)

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

plt.figure(figsize=(16, 60))

hidden_layer_dimensions = [1, 2, 4, 8, 16, 32, 64, 128]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(4, 2, i+1)
    clf = NN(nn_hdim, 2, n_classes)
    print("\nTraining NN with " + repr(nn_hdim) + " hidden layer nodes...")
    clf.train(X, Y, num_passes=2000, print_loss=True)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    plt.title('Hidden Layer size ' + repr(nn_hdim) + ' , Accuracy: ' + repr(round(accuracy,3)))
    plot_decision_boundary(lambda x: clf.predict(x))

matplotlib.pyplot.tight_layout()
plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
plt.show()