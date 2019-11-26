import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
# A Base de dados de dígitos
digits = datasets.load_digits()

#numero de dados
n_samples = len(digits.images)

data = digits.data

# Criando um classificador: Support vector classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Colocando a primeira metade de dados para treinamento
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Agora testamos a outra metade e analisamos a resposta do nosso modelo com relacao ao resultado real:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
acerto=metrics.accuracy_score(expected,predicted)*100

print(acerto)
imagens_e_valores = list(zip(digits.images, digits.target))
j=1
plt.figure(figsize=(25,7))
for image, label in imagens_e_valores[:6]:
    plt.subplot(2, 6, j)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Imagem enviada\n no Treino: %i' % label).set_fontsize('large')
    j+=1
imagens_e_predicoes=list(zip(digits.images[-6:],classifier.predict(digits.data[-6:])))
for image, label in imagens_e_predicoes:
    plt.subplot(2, 6, j)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Valor reconhecido\npelo modelo:{}' .format(label)).set_fontsize('large')
    j+=1
plt.show()
fig,ax = plt.subplots()
rects1=ax.bar(['Erro','Acerto'],[round(100-acerto,2),round(acerto,2)],color=['red','green'])
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