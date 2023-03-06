from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def classifier_short_version(train_x, train_y, test_x, test_y):
    # Train a Support Vector Machine (SVM) classifier
    svm = SVC()
    svm.fit(train_x, train_y)

    # Train a k-Nearest Neighbors (k-NN) classifier
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(random_state=0)
    rf.fit(train_x, train_y)

    # Evaluate the performance of the classifiers on the test set
    return svm.score(test_x, test_y), knn.score(test_x, test_y), rf.score(test_x, test_y)
