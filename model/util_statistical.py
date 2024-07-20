import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


class Counter_MultiLabel_CM():

    def __init__(self, num_of_label) -> None:
        self.label_num = num_of_label
        self.cm = np.zeros((self.label_num, 2, 2))

    def clear(self):
        self.cm = np.zeros((self.label_num, 2, 2))

    def record(self, y_true, y_pred):
        tmp_cm = multilabel_confusion_matrix(y_true, y_pred)
        self.cm += tmp_cm

    def output_F1(self):
        for i in range(len(self.cm)):
            tp = self.cm[i][1][1]
            fp = self.cm[i][0][1]
            tn = self.cm[i][0][0]
            fn = self.cm[i][1][0]

            accuracy = (tp + tn)/(tp + fp + tn + fn)
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            precision_r = tn / (tn + fn) if tn + fn != 0 else 0
            recall_r = tn / (tn + fp) if tn + fp !=0 else 0
            f1_score_r = 2 * precision_r * recall_r / (precision_r + recall_r) if precision_r + recall_r != 0 else 0

            print(f"Label {i+1}:Acc={accuracy:.4f}, " +
                  f"Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}, " +
                  f"Precision(R)={precision_r:.4f}, Recall(R)={recall_r:.4f}, F1 Score(R)={f1_score_r:.4f}")