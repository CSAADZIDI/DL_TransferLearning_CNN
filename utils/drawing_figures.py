import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score,auc, roc_curve,accuracy_score
import matplotlib.pyplot as plt
import mlflow
import os
import sys
import numpy as np
from .functions_model import load_models
from .functions_data import get_data
def draw_confusion_matrix(y_test, y_pred,path:str):
    """ draw confusion matrix
    
    Args:
        path (str): 
        
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,annot=True)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    mlflow.log_artifact(os.path.join(path, 'confusion_matrix.png'))
    print("Confusion Matrix:\n", cm)
    plt.show()
    
def draw_roc_curve(y_test, y_pred_prob,path:str):
    """draw roc_curve

    Args:
        path (str): 
        
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve score (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path,'ROC.png'))
    mlflow.log_artifact(os.path.join(path,'ROC.png'))
    plt.show()

def draw_train_val_acc_loss(history,path):
    """ draw training accuracy and loss vs validation accuracy and loss respectively

    Args:
        path (str): destination folder
        history (): output of the fit() method
    """

# Extract metrics
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    # Create subplots
    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='s')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(path,"training_val_curves.png"))
    mlflow.log_artifact(os.path.join(path,"training_val_curves.png"))
    plt.show()

def draw_accuracy_train_val_test(model,history, X_test,y_test,dual:bool, path):
    """draw model accuracy train vs val vs test
    """
    train_acc = history.history['accuracy'][-1]        # Final training accuracy
    val_acc = history.history['val_accuracy'][-1]      # Final validation accuracy
    if dual is True:
        test_loss, test_acc = model.evaluate([X_test,X_test], y_test, verbose=0)
    else:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies = [train_acc, val_acc, test_acc]
    labels = ['Training', 'Validation', 'Test']

    plt.figure(figsize=(6, 4))
    plt.bar(labels, accuracies, color=['skyblue', 'orange', 'green'])
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(path,"accuracy_comparison.png"))
    mlflow.log_artifact(os.path.join(path,"accuracy_comparison.png"))
    plt.show()
    
    


    
def draw_models_acc(models_path, figures_path, X_test,y_test):
    """ load and draw models accuracy on test data
    """
    models = []
    test_accuracies =[]
    models_avg = []
    # load models
    
    models,labels = load_models(models_path)
    print("models loaded")
    print("labels",labels)
    for model,label in zip(models,labels):
        print("label",label)
        if "dual" in label:
            print("dual model",model, "label",label)
            test_loss, test_acc = model.evaluate([X_test,X_test], y_test, verbose=0)
        elif ("densenet" in label) or ("resnet" in label):
            models_avg.append(model)
            print(" for average ")
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        else:
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_accuracies.append(test_acc)
    
    # for the ensemble learning using average
    print("len",len(models_avg))
    # Predict on test data
    preds1 = models_avg[0].predict(X_test)
    preds2 = models_avg[1].predict(X_test)

    # Average predictions
    ensemble_preds = (preds1+preds2) / 2
    ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)


    # Compute ensemble accuracy
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds_binary)
    test_accuracies.append(ensemble_accuracy)
    labels.append("ensemble (avg)")
    
    
    
    plt.figure(figsize=(6, 4))
    

    
    bars = plt.bar(labels, test_accuracies)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    #plt.yticks(np.arange(0, 1.05, 0.05))  # More precise tick marks (every 0.05)
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')  # ha='right' aligns labels nicely
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
    # Ajouter les valeurs sur chaque barre
    for bar in bars:
        height = bar.get_height()
        plt.text(
        bar.get_x() + bar.get_width() / 2,  # position horizontale (centre de la barre)
        height + 0.02,                      # position verticale (un peu au-dessus de la barre)
        f'{height:.2f}',                    # texte (valeur arrondie à 2 décimales)
        ha='center', va='bottom'
        )
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "models_acc.png"))
    plt.show()
    
    
if __name__ == "__main__":
    """ test func"""


    path= "models/"
    figures_path = "figures/"
    X_test,y_test = get_data("data/test/",['PNEUMONIA','NORMAL'])
    draw_models_acc(path, figures_path, X_test,y_test)
    
    
    

    
