import datetime 
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, average_precision_score


def write_log(w):
    file_name = 'logs/' + datetime.date.today().strftime('%m%d')+"_{}.log"
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f: 
        f.write(info + '\n') 

def calculate_metrics(true_values, predictions, epoch, num_epochs, train_loss, val_loss, label):
    fpr, tpr, _ = roc_curve(true_values, predictions)
    auc_value = auc(fpr, tpr)
    ks = max(tpr - fpr)
    gini = 2 * auc_value - 1
    val_preds_class = [1 if x > 0.5 else 0 for x in predictions]
    accuracy = accuracy_score(true_values, val_preds_class)
    recall = recall_score(true_values, val_preds_class)
    f1 = f1_score(true_values, val_preds_class)
    auc_pr = average_precision_score(true_values, predictions)
    msg = (f'{label}: Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
           f'Val AUC_roc score: {auc_value:.4f}, Val KS score: {ks:.4f}, Val gini score: {gini:.4f}, '
           f'Val accuracy score: {accuracy:.4f}, Val recall score: {recall:.4f}, Val f1 score: {f1:.4f}, '
           f'Val auc_pr score: {auc_pr:.4f}')
    write_log(msg)
    return ks   