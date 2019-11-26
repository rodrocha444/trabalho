import matplotlib.pyplot as plt
from sklearn import datasets, metrics
iris=datasets.load_iris()
plt.figure(figsize=(15, 4))
plt.legend('o',loc=0,bbox_to_anchor=(0.5, 0., 0.5, 0.5),scatterpoints=150)
plt.subplot(1,2,1)
plt.scatter(iris.data[:,0],iris.data[:,2],marker="o",c=iris.target,s=30,)
plt.xlabel("Largura da pétala")
plt.ylabel("Largura da sépala")
plt.subplot(1,2,2)
plt.scatter(iris.data[:,1],iris.data[:,3],marker="o",c=iris.target,s=30,)
plt.xlabel("Comprimento da pétala")
plt.ylabel("Comprimento da sépala")
plt.show()