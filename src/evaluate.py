from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from train_model import preds, y_val

cm = confusion_matrix(y_val, preds)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()