from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# CHALLENGE - create 3 more classifiers...
clf = tree.DecisionTreeClassifier()
clf_gauss = GaussianNB()
clf_kn = KNeighborsClassifier(n_neighbors=3)
clf_random = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf_gauss = clf_gauss.fit(X,Y)
clf_kn = clf_kn.fit(X,Y)
clf_random = clf_random.fit(X,Y)

prediction_furkan = clf.predict([[182, 87, 41]])
prediction_mehmet = clf.predict([[183, 121, 45]])

# CHALLENGE compare their results and print the best one!
pred_clf = clf.predict(X)
acc_decision = accuracy_score(Y,pred_clf) 

pred_clf_gauss = clf_gauss.predict(X)
acc_gauss = accuracy_score(Y,pred_clf_gauss)

pred_clf_kn = clf_kn.predict(X)
acc_kn = accuracy_score(Y,pred_clf_kn)

pred_clf_random = clf_random.predict(X)
acc_random = accuracy_score(Y,pred_clf_random)

accuracies = {"GaussianNB" : acc_gauss,
              "Desicion Tree" : acc_decision,
              "KNeighborsClassifier" : acc_gauss,
              "RandomForestClassifier" : acc_kn,
}
print(max(zip(accuracies.values(), accuracies.keys())))
