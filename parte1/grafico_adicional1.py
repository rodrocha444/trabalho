import matplotlib.pyplot as plt
from sklearn import datasets
digits=datasets.load_digits()
j=1
plt.figure(figsize=(10,4))
for image in digits.images[:16]:
    plt.subplot(2, 8, j)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    j+=1
plt.show()