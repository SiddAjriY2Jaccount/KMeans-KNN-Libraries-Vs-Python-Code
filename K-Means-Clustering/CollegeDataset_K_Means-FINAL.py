#!/usr/bin/env python
# coding: utf-8

# In[181]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import seaborn as sns


# In[182]:


dataset = pd.read_csv('College.csv')
dataset.head()


# In[183]:


dataset['Enrollment_Rate'] = (dataset['Enroll']/dataset['Accept'])*100.0
dataset.head()


# In[184]:


dataset['Acceptance Rate'] = (dataset['Accept']/dataset['Apps'])*100.0
dataset.head()


# In[185]:


dataset[dataset['Enrollment_Rate'] == 100]


# In[186]:


dataset.shape


# In[187]:


sns.set_style('whitegrid')
sns.lmplot('Enrollment_Rate','Expend',data=dataset, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[188]:


X = dataset.iloc[:, [17, 19]].values #Choosing Room.Board and Grad.Rate as Atrributes
X


# In[189]:


print(X.shape)

m = X.shape[0] #number of training examples
n = X.shape[1] #number of features. Here n will be 2, naturally

print("m,n",m,n)

epochs = 100 #max no. of iterations


# In[190]:


K = 2  # number of clusters, for now, change it later


# In[191]:


Centroids = np.array([]).reshape(n,0) #Centroids is a n x K dimensional matrix, where each column will be a centroid for one cluster


# In[192]:


print(np.array(Centroids))


# In[193]:


for i in range(K):
    rand = rd.randint(0,m-1)
    Centroids = np.c_[Centroids,X[rand]]

Centroids


# In[194]:


Output={} #Dict with key as Cluster no. and Values as each data point in the cluster


# In[195]:


'''
EuclidianDistance = np.array([]).reshape(m,0)
for k in range(K):
    tempDist = np.sum((X-Centroids[:,k])**2,axis=1)
    EuclidianDistance = np.c_[EuclidianDistance,tempDist]
C = np.argmin(EuclidianDistance,axis=1)+1

C
'''


# In[196]:


for i in range(epochs):
     #step 2.a
      EuclidianDistance = np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2(b)
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y
    
Output


# In[197]:


Centroids


# In[198]:


color=['red','blue','green','cyan','magenta']
labels=['Public','Private','cluster-3','cluster-4','cluster-5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=100,c='yellow',label='Centroids')
plt.xlabel('Enrollment_Rate')
plt.ylabel('Expend')
plt.legend()
plt.show()
#Output[k+1][:,0]
#Output[k+1][:,1]


# In[199]:


i2=[]
for i1 in Output[1]:
    i2.append(list(i1))


#i2 is a list of of all the data points clustered as Public


# In[200]:


dataset['Class_label'] = np.nan
dataset


# In[201]:


no_of_rows, no_of_cols = dataset.shape
print(no_of_rows)


# In[202]:


for kk in i2:
    dataset.loc[(dataset['Expend']==kk[0])&(dataset['Enrollment_Rate']==kk[1]),'Class_label'] = 'No'

    
    


# In[203]:


dataset


# In[204]:


i9=[]
for i1 in Output[2]:
    i9.append(list(i1))


#i9 is a list of of all the data points clustered as Private


# In[205]:


for kk in i9:
    dataset.loc[(dataset['Expend']==kk[0])&(dataset['Enrollment_Rate']==kk[1]),'Class_label'] = 'Yes'

dataset   


# In[206]:


dataset[dataset['Class_label'] == np.nan]


# In[207]:


dataset.shape


# In[208]:


dataset[dataset['Class_label'] == 'Yes'].shape
    


# In[209]:


'''
test=dataset.loc[(dataset['Expend']==i2[0][0])&(dataset['Enrollment_Rate']==i2[0][1])]
test['Label']
print(test)
'''
numRows, numCols = dataset.shape


# In[210]:


TP=TN=FP=FN=0
# for kk2 in range(numRows):
#     rowdf = dataset.loc[kk2,:]
#     print(rowdf)
    
factual_data=list(dataset['Private'])
test_data=list(dataset['Class_label'])

#print(factual_data,"\n",test_data)

for x,y in zip(factual_data,test_data):
    if x==y and x=='Yes':
        TP+=1
    if x==y and x=='No':
        TN+=1
    if x!=y and x=='Yes':
        FN+=1
    if x!=y and x=='No':
        FP+=1
    
#     if dataset.loc[kk2, dataset.columns.get_loc('Private')]:#.equals(dataset.loc[kk2, dataset.columns.get_loc('Class_label')]):
#         if dataset[kk2, dataset.columns.get_loc('Private')] == 'No':
#             TN+=1
#         else:
#             TP+=1
#     else:
#         if dataset[kk2, dataset.columns.get_loc('Private')] == 'No':
#             FP+=1
#         else:
#             FN+=1

print("ACCURACY by Confusion Matrix:")
print((TP+TN)/(numRows))
print()

print("Recall:")
Recall = (TP/(TP+FN))
print(Recall)
print()

print("Precision:")
Precision = (TP/(TP+FP))
print(Precision)
print()

print("F-Measure:")
print((2*Recall*Precision)/(Recall+Precision))
print()
print("$$$$$ TN : ", TN)
print("$$$$$ FN : ", FN)

#Low recall, high precision: Shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP) 


# In[211]:


######## USING SKLEARN - COMPARING RESULTS WITH LIBRARY FNS./APIs #########


# In[212]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[213]:


Y = dataset[['Enrollment_Rate']]
X = dataset[['Expend']]


# In[214]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
#kmeans


# In[215]:


score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
#score


# In[216]:


plt.scatter(Nc,score,c='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[217]:


pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)


# In[218]:


kmeans=KMeans(n_clusters=2)
kmeansoutput=kmeans.fit(Y)
kmeansoutput


# In[219]:


plt.figure('2 Cluster K-Means')
plt.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
plt.xlabel('Dividend Yield')
plt.ylabel('Returns')
plt.title('2 Cluster K-Means')
plt.show()


# In[220]:


######### ENTROPY CALCULATION ########## - Lower entropy, better cluster purity/validity
import math

clustersj = dataset[['Class_label']]
classi = dataset[['Private']]
count1 = 0

mij = [] #ij = YesYes, YesNo, NoYes, NoNo order
mj = [] #Yes, No order
pij = [] #ij = YesYes, YesNo, NoYes, NoNo order

mj1, mjr = clustersj[clustersj['Class_label'] == 'Yes'].shape
mj.append(mj1)
mj1, mjr = clustersj[clustersj['Class_label'] == 'No'].shape
mj.append(mj1)

mi1, mjr = classi[classi['Private'] == 'Yes'].shape
#print(mi1)

test1=dataset.loc[(dataset['Private']=='Yes')&(dataset['Class_label']=='Yes'),:]
count1, mjr = test1.shape
mij.append(count1)
test2=dataset.loc[(dataset['Private']=='Yes')&(dataset['Class_label']=='No'),:]
count2, mjr = test2.shape
mij.append(count2)
test3=dataset.loc[(dataset['Private']=='No')&(dataset['Class_label']=='Yes'),:]
count3, mjr = test3.shape
mij.append(count3)
test4=dataset.loc[(dataset['Private']=='No')&(dataset['Class_label']=='No'),:]
count4, mjr = test4.shape
mij.append(count4)

#print(mj)

#print(mij)

pij.append(float(mij[0])/float(mj[0]))
pij.append(float(mij[2])/float(mj[0]))
pij.append(float(mij[1])/float(mj[1]))
pij.append(float(mij[3])/float(mj[1]))
        
#print(pij)
#print("\n\n")

Entropy = 0.0

for i in pij:
    #print(i*(math.log(i,2)))
    Entropy += (i*(math.log(i,2))) #math.log(num, base)
Entropy = - Entropy
print("############# ---------- ###########\n")
print("ENTROPY :\n")
print(Entropy)


# In[ ]:




