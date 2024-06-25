import pandas as pd

# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Import svm model
from sklearn import svm

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Dung PCA
filepath_pca = "C:/Pycharm Projects/PCA_test/lda_dataset.csv"

df = pd.read_csv(filepath_pca, names=['principal component 1', 'principal component 2', 'Label'])

features = ['principal component 1', 'principal component 2']

# Khong dung PCA
# file_path = "C:/Pycharm Projects/PCA_Test/dataset_demo2.csv"
#
# df = pd.read_csv(file_path, names=['EtOH', 'VOC1', 'VO2', 'CH4', 'H2S', 'CO', 'Odor', 'NH3', 'Label'])
#
# features = ['EtOH', 'VOC1', 'VO2', 'CH4', 'H2S', 'CO', 'Odor', 'NH3']

# Separating out the features
x = df.loc[:, features].values
# x = df[features].values

# Separating out the Label
y = df.loc[:, ['Label']].values
# y = df[['Label']].values

# Since y is categorical Data, You will need to one hot encode
# label_encoder = LabelEncoder()
# y_encode = label_encoder.fit_transform(y)

# Standardizing the features
# x_scale = StandardScaler().fit_transform(x)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=109)  # 80% training and 20% test
# X_train, X_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=109)  # 80% training and 20% test


# Create a svm Classifier
# clf = svm.LinearSVC(max_iter=4000)  # Linear Kernel #94
# clf = svm.SVC(kernel= 'linear')  # Linear Kernel      #98
clf = svm.SVC(C=10, gamma=0.1)  # RBF Kernel         #98
# clf = svm.SVC(kernel= 'poly', degree=1)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))

print("Matrix:", metrics.confusion_matrix(y_test, y_pred))
