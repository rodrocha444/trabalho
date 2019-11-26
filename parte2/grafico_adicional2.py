import matplotlib.pyplot as plt
fig,ax = plt.subplots()
rects1=ax.bar(['SVM','KNN','DecisionTree','AdaBoostClassifier'],[93.33,98.33,98.33,96.67],color=['green','blue','blue','green'])
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