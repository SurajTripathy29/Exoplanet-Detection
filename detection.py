import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

#uploading the dataset
train_df=pd.read_csv("exoTrain.csv")
train_df.head(10)
#The "flux" in this context would be a measure of how much light (energy) is reaching a specific area at that distance.

#checking the shape of the dataset
train_df.shape

#display the rows in the null values
train_df[train_df.isnull().any(axis=1)]

#display null values (if any)
sns.heatmap(train_df.isnull())

#finding how may labels are present
train_df['LABEL'].unique()

#extract the index of stars whose label are 2
list(train_df[train_df['LABEL']==2].index)

# Visualise these values using countplot
plt.figure(figsize=(3, 5))
ax = sns.countplot(x='LABEL', data=train_df)
ax.bar_label(ax.containers[0])  
plt.show()

#replace 2 with 1
#replace 1 with 0
train_df=train_df.replace({"LABEL":{2:1,1:0}})
train_df.LABEL.unique()

plot_df=train_df.drop(["LABEL"],axis=1)
plot_df

#plot random star from the plot df
time=range(1,3198)
flux_val=plot_df.iloc[3,:].values
plt.figure(figsize=(15,5))
plt.plot(time,flux_val,linewidth=1)

time=range(1,3198)
flux_val=plot_df.iloc[43,:].values
plt.figure(figsize=(15,5))
plt.plot(time,flux_val,linewidth=1)

time=range(1,3198)
flux_val=plot_df.iloc[98,:].values
plt.figure(figsize=(15,5))
plt.plot(time,flux_val,linewidth=1)

time=range(1,3198)
flux_val=plot_df.iloc[198,:].values
plt.figure(figsize=(15,5))
plt.plot(time,flux_val,linewidth=1)

#as we can see even stars without exoplanet is also giving a huge spike(outliers) which is not requied
#so we will be using knn(as it is sensitive to outliers)

for i in range(1,4):
    plt.subplot(1,4,i)
    sns.boxplot(data=train_df,x='LABEL',y='FLUX.' + str(i))
    
#dropping the outliers
train_df.drop(train_df[train_df['FLUX.2']>0.25e6].index,axis=0,inplace=True)
sns.boxplot(data=train_df,x='LABEL',y='FLUX.'+str(np.random.randint(1000)))
    
#extracting independent (x) and dependent (y) features from df
x=train_df.drop(['LABEL'],axis=1)
y=train_df.LABEL

#Splitting this data into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)   
    
#feature scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.fit_transform(X_test) 

# Fiting the KNN Classifier Model on to the training data
from sklearn.neighbors import KNeighborsClassifier as KNC

# Choosing K = 10
knn_classifier = KNC(n_neighbors=5,metric='minkowski',p=2)  
'''metric is to be by default minkowski for p = 2 to calculate the Eucledian distances'''

# Fit the model
knn_classifier.fit(X_train_sc, y_train)

# Predict
y_pred = knn_classifier.predict(X_test_sc)

# Results
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

print('\nValidation accuracy of KNN is', accuracy_score(y_test,y_pred))
print("\n-------------------------------------------------------")
print ("\nClassification report :\n",(classification_report(y_test,y_pred)))

#Confusion matrix
plt.figure(figsize=(15,11))
plt.subplots_adjust(wspace = 0.3)
plt.suptitle("KNN Performance before handling the imbalance in the data", color = 'r', weight = 'bold')
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="Set2",fmt = "d",linewidths=3, cbar = False,
           xticklabels=['nexo', 'exo'], yticklabels=['nexo','exo'], square = True)
plt.xlabel("True Labels", fontsize = 15, weight = 'bold', color = 'tab:pink')
plt.ylabel("Predicited Labels", fontsize = 15, weight = 'bold', color = 'tab:pink')
plt.title("CONFUSION MATRIX",fontsize=20, color = 'm')

#ROC curve and Area under the curve plotting
predicting_probabilites = knn_classifier.predict_proba(X_test_sc)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("AUC :",auc(fpr,tpr)),color = "g")
plt.plot([1,0],[1,0],"k--")
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20, color = 'm')
plt.show() 
  









