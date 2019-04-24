{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import random as rd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>BALANCE</th>\n",
       "      <th>BALANCE_FREQUENCY</th>\n",
       "      <th>PURCHASES</th>\n",
       "      <th>ONEOFF_PURCHASES</th>\n",
       "      <th>INSTALLMENTS_PURCHASES</th>\n",
       "      <th>CASH_ADVANCE</th>\n",
       "      <th>PURCHASES_FREQUENCY</th>\n",
       "      <th>ONEOFF_PURCHASES_FREQUENCY</th>\n",
       "      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>\n",
       "      <th>CASH_ADVANCE_FREQUENCY</th>\n",
       "      <th>CASH_ADVANCE_TRX</th>\n",
       "      <th>PURCHASES_TRX</th>\n",
       "      <th>CREDIT_LIMIT</th>\n",
       "      <th>PAYMENTS</th>\n",
       "      <th>MINIMUM_PAYMENTS</th>\n",
       "      <th>PRC_FULL_PAYMENT</th>\n",
       "      <th>TENURE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8949.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8637.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "      <td>8950.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1564.474828</td>\n",
       "      <td>0.877271</td>\n",
       "      <td>1003.204834</td>\n",
       "      <td>592.437371</td>\n",
       "      <td>411.067645</td>\n",
       "      <td>978.871112</td>\n",
       "      <td>0.490351</td>\n",
       "      <td>0.202458</td>\n",
       "      <td>0.364437</td>\n",
       "      <td>0.135144</td>\n",
       "      <td>3.248827</td>\n",
       "      <td>14.709832</td>\n",
       "      <td>4494.449450</td>\n",
       "      <td>1733.143852</td>\n",
       "      <td>864.206542</td>\n",
       "      <td>0.153715</td>\n",
       "      <td>11.517318</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2081.531879</td>\n",
       "      <td>0.236904</td>\n",
       "      <td>2136.634782</td>\n",
       "      <td>1659.887917</td>\n",
       "      <td>904.338115</td>\n",
       "      <td>2097.163877</td>\n",
       "      <td>0.401371</td>\n",
       "      <td>0.298336</td>\n",
       "      <td>0.397448</td>\n",
       "      <td>0.200121</td>\n",
       "      <td>6.824647</td>\n",
       "      <td>24.857649</td>\n",
       "      <td>3638.815725</td>\n",
       "      <td>2895.063757</td>\n",
       "      <td>2372.446607</td>\n",
       "      <td>0.292499</td>\n",
       "      <td>1.338331</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.019163</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>128.281915</td>\n",
       "      <td>0.888889</td>\n",
       "      <td>39.635000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.083333</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1600.000000</td>\n",
       "      <td>383.276166</td>\n",
       "      <td>169.123707</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>873.385231</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>361.280000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>89.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.083333</td>\n",
       "      <td>0.166667</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>3000.000000</td>\n",
       "      <td>856.901546</td>\n",
       "      <td>312.343947</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2054.140036</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1110.130000</td>\n",
       "      <td>577.405000</td>\n",
       "      <td>468.637500</td>\n",
       "      <td>1113.821139</td>\n",
       "      <td>0.916667</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.750000</td>\n",
       "      <td>0.222222</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>6500.000000</td>\n",
       "      <td>1901.134317</td>\n",
       "      <td>825.485459</td>\n",
       "      <td>0.142857</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>19043.138560</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>49039.570000</td>\n",
       "      <td>40761.250000</td>\n",
       "      <td>22500.000000</td>\n",
       "      <td>47137.211760</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>123.000000</td>\n",
       "      <td>358.000000</td>\n",
       "      <td>30000.000000</td>\n",
       "      <td>50721.483360</td>\n",
       "      <td>76406.207520</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            BALANCE  BALANCE_FREQUENCY     PURCHASES  ONEOFF_PURCHASES  \\\n",
       "count   8950.000000        8950.000000   8950.000000       8950.000000   \n",
       "mean    1564.474828           0.877271   1003.204834        592.437371   \n",
       "std     2081.531879           0.236904   2136.634782       1659.887917   \n",
       "min        0.000000           0.000000      0.000000          0.000000   \n",
       "25%      128.281915           0.888889     39.635000          0.000000   \n",
       "50%      873.385231           1.000000    361.280000         38.000000   \n",
       "75%     2054.140036           1.000000   1110.130000        577.405000   \n",
       "max    19043.138560           1.000000  49039.570000      40761.250000   \n",
       "\n",
       "       INSTALLMENTS_PURCHASES  CASH_ADVANCE  PURCHASES_FREQUENCY  \\\n",
       "count             8950.000000   8950.000000          8950.000000   \n",
       "mean               411.067645    978.871112             0.490351   \n",
       "std                904.338115   2097.163877             0.401371   \n",
       "min                  0.000000      0.000000             0.000000   \n",
       "25%                  0.000000      0.000000             0.083333   \n",
       "50%                 89.000000      0.000000             0.500000   \n",
       "75%                468.637500   1113.821139             0.916667   \n",
       "max              22500.000000  47137.211760             1.000000   \n",
       "\n",
       "       ONEOFF_PURCHASES_FREQUENCY  PURCHASES_INSTALLMENTS_FREQUENCY  \\\n",
       "count                 8950.000000                       8950.000000   \n",
       "mean                     0.202458                          0.364437   \n",
       "std                      0.298336                          0.397448   \n",
       "min                      0.000000                          0.000000   \n",
       "25%                      0.000000                          0.000000   \n",
       "50%                      0.083333                          0.166667   \n",
       "75%                      0.300000                          0.750000   \n",
       "max                      1.000000                          1.000000   \n",
       "\n",
       "       CASH_ADVANCE_FREQUENCY  CASH_ADVANCE_TRX  PURCHASES_TRX  CREDIT_LIMIT  \\\n",
       "count             8950.000000       8950.000000    8950.000000   8949.000000   \n",
       "mean                 0.135144          3.248827      14.709832   4494.449450   \n",
       "std                  0.200121          6.824647      24.857649   3638.815725   \n",
       "min                  0.000000          0.000000       0.000000     50.000000   \n",
       "25%                  0.000000          0.000000       1.000000   1600.000000   \n",
       "50%                  0.000000          0.000000       7.000000   3000.000000   \n",
       "75%                  0.222222          4.000000      17.000000   6500.000000   \n",
       "max                  1.500000        123.000000     358.000000  30000.000000   \n",
       "\n",
       "           PAYMENTS  MINIMUM_PAYMENTS  PRC_FULL_PAYMENT       TENURE  \n",
       "count   8950.000000       8637.000000       8950.000000  8950.000000  \n",
       "mean    1733.143852        864.206542          0.153715    11.517318  \n",
       "std     2895.063757       2372.446607          0.292499     1.338331  \n",
       "min        0.000000          0.019163          0.000000     6.000000  \n",
       "25%      383.276166        169.123707          0.000000    12.000000  \n",
       "50%      856.901546        312.343947          0.000000    12.000000  \n",
       "75%     1901.134317        825.485459          0.142857    12.000000  \n",
       "max    50721.483360      76406.207520          1.000000    12.000000  "
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_csv('CC-GENERAL.csv')\n",
    "dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:, [1, 6]].values #Choosing Balance and Cash_Advance as Atrributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(8950, 2)\n",
      "m,n 8950 2\n"
     ]
    }
   ],
   "source": [
    "print(X.shape)\n",
    "\n",
    "m = X.shape[0] #number of training examples\n",
    "n = X.shape[1] #number of features. Here n will be 2, naturally\n",
    "\n",
    "print(\"m,n\",m,n)\n",
    "\n",
    "epochs = 100 #max no. of iterations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "K = 2  # number of clusters, for now, change it later"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Centroids = np.array([]).reshape(n,0) #Centroids is a n x K dimensional matrix, where each column will be a centroid for one cluster"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[]\n"
     ]
    }
   ],
   "source": [
    "print(np.array(Centroids))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  80.198389, 2837.962185],\n",
       "       [   0.      ,  958.26211 ]])"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i in range(K):\n",
    "    rand = rd.randint(0,m-1)\n",
    "    Centroids = np.c_[Centroids,X[rand]]\n",
    "\n",
    "Centroids"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Output={} #Dict with key as Cluster no. and Values as each data point in the cluster"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nEuclidianDistance = np.array([]).reshape(m,0)\\nfor k in range(K):\\n    tempDist = np.sum((X-Centroids[:,k])**2,axis=1)\\n    EuclidianDistance = np.c_[EuclidianDistance,tempDist]\\nC = np.argmin(EuclidianDistance,axis=1)+1\\n\\nC\\n'"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "EuclidianDistance = np.array([]).reshape(m,0)\n",
    "for k in range(K):\n",
    "    tempDist = np.sum((X-Centroids[:,k])**2,axis=1)\n",
    "    EuclidianDistance = np.c_[EuclidianDistance,tempDist]\n",
    "C = np.argmin(EuclidianDistance,axis=1)+1\n",
    "\n",
    "C\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(epochs):\n",
    "     #step 2.a\n",
    "      EuclidianDistance=np.array([]).reshape(m,0)\n",
    "      for k in range(K):\n",
    "          tempDist=np.sum((X-Centroids[:,k])**2,axis=1)\n",
    "          EuclidianDistance=np.c_[EuclidianDistance,tempDist]\n",
    "      C=np.argmin(EuclidianDistance,axis=1)+1\n",
    "     #step 2.b\n",
    "      Y={}\n",
    "      for k in range(K):\n",
    "          Y[k+1]=np.array([]).reshape(2,0)\n",
    "      for i in range(m):\n",
    "          Y[C[i]]=np.c_[Y[C[i]],X[i]]\n",
    "     \n",
    "      for k in range(K):\n",
    "          Y[k+1]=Y[k+1].T\n",
    "    \n",
    "      for k in range(K):\n",
    "          Centroids[:,k]=np.mean(Y[k+1],axis=0)\n",
    "      Output=Y\n",
    "    \n",
    "Output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "color=['red','blue','green','cyan','magenta']\n",
    "labels=['cluster-1','cluster-2','cluster-3','cluster-4','cluster-5']\n",
    "for k in range(K):\n",
    "    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])\n",
    "plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')\n",
    "plt.xlabel('Balance')\n",
    "plt.ylabel('Cash_Advance')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
