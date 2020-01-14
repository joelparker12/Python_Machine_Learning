from sklearn import datasets
import numpy as np
import pandas as pd

iris = datasets.load_iris()

X = iris.data[: , [2,3]]
y = iris.target

print('class labels', np.unique(y))




##### SPlit data into test and train data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = .3, random_state = 1, stratify =y)


print('label counts in y', np.bincount(y))
print('label counts in y_train', np.bincount(y_train))
print('label counts in y_test', np.bincount(y_test))



######Standardize the Features.

from sklearn.preprocessing import StandardScaler

std_X = StandardScaler()
std_X.fit(x_train)
X_train_STD = std_X.transform(x_train)
X_test_STD = std_X.transform(x_test)





##### Run the percepfroon for the 3 classes

from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0= .01, random_state = 1)
ppn.fit(X_train_STD, y_train)

### predict method
y_pred = ppn.predict(X_test_STD)

print('Number of wrong predictions out of 45', (y_test != y_pred).sum())

print('accuracy: ' , (1-(2/45)))





from sklearn.metrics import accuracy_score



print('accuracy Score: ', ppn.score(X_test_STD, y_test))
print('accuracy Score: ', accuracy_score(y_pred, y_test))

# create plot decision regions.


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt





# create plot decision region
def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02):
    # set up marker generator and color map.
    markers = ("s", "x", "o", "^")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #### Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')




X_combined_std = np.vstack((X_train_STD, X_test_STD))
y_combined = np.hstack((y_train, y_test))


plot_decision_region(X= X_combined_std, y= y_combined, classifier= ppn, test_idx= range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()







#### using SKlearn Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C= 100.0, random_state= 1, solver= 'lbfgs', multi_class= 'ovr')

lr.fit(X_train_STD, y_train)

plot_decision_region(X_combined_std, y_combined, classifier= lr, test_idx= range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('Logistic Regression Classifier')
plt.tight_layout()
plt.show()





##### Looking at the consequences of increasing reularization strength.
import numpy as np

weights, params = [], []

for c in np.arange(-5,5):
    lr= LogisticRegression(C= 10.**c, random_state= 1, solver = 'lbfgs', multi_class = 'ovr')
    lr.fit(X_train_STD, y_train)
    weights.append((lr.coef_[1]))
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label = 'petal length')
plt.plot(params, weights[: , 1], label = 'petal width', linestyle = '--')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.title('L2 regularization path for Iris, Petal length/width.')

