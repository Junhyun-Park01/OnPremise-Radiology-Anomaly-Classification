from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import matplotlib as mpl


####### Plotting AUC Curve and obtain sensitive, specificity, best threshold ########
plt.figure(figsize=(6,6))

y = np.load('./RESULTS/pubmedbert_sentence_gd.npy')
pred = np.load('./RESULTS/roberta_paragraph_abn_prob_full.npy')
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label= 0)
roc_auc = metrics.auc(fpr, tpr)

J = tpr - fpr
ix = np.argmax(J)
#plot roc and best threshold
sens, spec = 1-fpr[ix], tpr[ix]

best_thresh = thresholds[ix]
plt.plot(fpr, tpr, label ='sentence-level estimator (AUC='+str(round(roc_auc, 3))+')')
plt.scatter(fpr[ix], tpr[ix], marker='+', s=100, color='r',
            label='Best threshold = %.3f, \nSensitivity = %.3f, \nSpecificity = %.3f' % (best_thresh, sens, spec))
print(best_thresh, sens, spec)

plt.legend(fontsize= 12)
plt.show()

###### Get Confusion matrix and accuracy on best threshold ########
def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size = 16)
    plt.xlabel('Predicted label', size = 16)
    plt.show()

for i in range(len(pred)):
    if pred[i] > best_thresh:
        pred[i] = 0
    else:
        pred[i] = 1

cm = confusion_matrix(y, pred)
plot_confusion_matrix(cm, target_names= ['abnormal', 'normal'], cmap = plt.get_cmap('YlGn'), normalize=False, title="")
plt.show()
print(accuracy_score(y, pred))
