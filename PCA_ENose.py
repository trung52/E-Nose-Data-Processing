import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

file_path = "C:/Pycharm Projects/PCA_Test/dataset_demo2.csv"

# load dataset into Pandas DataFrame
df = pd.read_csv(file_path, names=['EtOH', 'VOC1', 'VO2', 'CH4', 'H2S', 'CO', 'Odor', 'NH3', 'Label'])

features = ['EtOH', 'VOC1', 'VO2', 'CH4', 'H2S', 'CO', 'Odor', 'NH3']

# Separating out the features
x = df.loc[:, features].values

# Separating out the Label
y = df.loc[:, ['Label']].values

# # Xem xet viec encode label khi su dung pca/lda
# label_encoder = LabelEncoder()
# y_encode = label_encoder.fit_transform(y)

# Standardizing the features
x_scale = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x_scale)

principalDf = pd.DataFrame(data = principalComponents
             , columns=['principal component 1', 'principal component 2'])

finalDf1 = pd.concat([principalDf, df[['Label']]], axis=1)
finalDf1.to_csv('C:/Pycharm Projects/PCA_test/pca_dataset.csv')

fig = plt.figure(figsize = (16,16))
ax = fig.add_subplot(1,2,1)
ax.set_xlabel('Principal Component 1 (' + str(round(pca.explained_variance_ratio_[0]*100,2)) + '%)', fontsize=15)
ax.set_ylabel('Principal Component 2 (' + str(round(pca.explained_variance_ratio_[1]*100,2)) + '%)', fontsize=15)
ax.set_title('2 component PCA', fontsize = 20)

labels = ['Dalatmilk','Mocchau', 'Thtruemilk', 'Vinamilk', 'Dutchlady']
colors = ['r', 'g', 'b', 'c', 'm']
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', '#FF5733', '#33FF57', '#3357FF']

for label, color in zip(labels,colors):
    indicesToKeep = finalDf1['Label'] == label
    ax.scatter(finalDf1.loc[indicesToKeep, 'principal component 1']
               , finalDf1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(labels)
ax.grid()
print(pca.explained_variance_ratio_)
# LDA
lda = LinearDiscriminantAnalysis(n_components=2)

ldaComponents = lda.fit_transform(x, y)

ldaDf = pd.DataFrame(data = ldaComponents
                        , columns= ['lda component 1','lda component 2'])
finalDf2 = pd.concat([ldaDf, df[['Label']]], axis = 1)
finalDf2.to_csv('C:/Pycharm Projects/PCA_test/lda_dataset.csv')

ax2 = fig.add_subplot(1,2,2)

ax2.set_xlabel('LDA Component 1 (' + str(round(lda.explained_variance_ratio_[0]*100,2)) + '%)', fontsize=15)
ax2.set_ylabel('LDA Component 2 (' + str(round(lda.explained_variance_ratio_[1]*100,2)) + '%)', fontsize=15)
ax2.set_title('2 component LDA', fontsize=20)

for label, color in zip(labels,colors):
    indicesToKeep = finalDf2['Label'] == label
    ax2.scatter(finalDf2.loc[indicesToKeep, 'lda component 1']
               , finalDf2.loc[indicesToKeep, 'lda component 2']
               , c = color
               , s = 50)
ax2.legend(labels)
ax2.grid()
print(lda.explained_variance_ratio_)
plt.show()