import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

#rearranjando 30 amostras para treino de cada tipo de iris
iris=datasets.load_iris()
data=np.append(iris.data[:30],iris.data[50:80],axis=0)
data=np.append(data,iris.data[100:130],axis=0)
target=np.append(iris.target[:30],[iris.target[50:80],iris.target[100:130]])

clf.fit(data,target)

#rearranjando 20 amostras para teste de cada tipo de iris
data_for_predict=np.append(iris.data[30:50],iris.data[80:100],axis=0)
data_for_predict=np.append(data_for_predict,iris.data[130:150],axis=0)
expected=np.append(iris.target[30:50],[iris.target[80:100],iris.target[130:150]])
predicted=clf.predict(data_for_predict)


acerto=round(metrics.accuracy_score(expected,predicted)*100,2)
predicted
fig,ax = plt.subplots()
rects1=ax.bar(['Erro','Acerto'],[round(100-acerto,2),acerto],color=['red','green'])
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
plt.show()

m_conf=metrics.confusion_matrix(expected, predicted)

dados=[m_conf[0,0],m_conf[0,1]+m_conf[0,2]]
plt.subplot(3,1,1)
plt.title('Tipo 0')
plt.pie(dados,labels=dados,labeldistance=0.3,colors=['green','red'])

dados=[m_conf[1,1],m_conf[1,0]+m_conf[1,2]]
plt.subplot(3,1,2)
plt.title('Tipo 1')
plt.pie(dados,labels=dados,labeldistance=0.3,colors=['green','red'])

dados=[m_conf[2,2],m_conf[2,1]+m_conf[2,0]]
plt.subplot(3,1,3)
plt.title('Tipo 2')
plt.pie(dados,labels=dados,labeldistance=0.3,colors=['green','red'])
plt.show()