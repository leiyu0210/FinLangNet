from Model.sentences_load import train_dataloader, val_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_loss import FocalTverskyLoss, DiceBCELoss, MultiLoss, DynamicWeightAverage
from Model.FinLangNet import *
from metric_record import *
from torch.optim.lr_scheduler import StepLR
from other_models.gru_deepfm import MyModel_GRU
from other_models.gru_attention_deepfm import MyModel_GRU_Attention
from other_models.lstm_deepfm import MyModel_LSTM
from other_models.transformer_deepfm import MyModel_Transformer
from other_models.stack_gru_deepfm import MyModel_StackGRU

# Automatically select device based on availability of CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    early_stop_patience = 10  
    best_val_loss = float('inf')
    best_epoch = 0
    # Dice and  Focal Tversky
    loss_weighter = DynamicWeightAverage(2)
    criterion1 = DiceBCELoss() 
    criterion2 = FocalTverskyLoss()
    # You can change other models
    model = MyModel_FinLangNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size = 3, gamma=0.2)
    num_epochs = 12
    best_ks = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        loss_weights = [1.0, 1.0]  
        multi_loss = MultiLoss(loss_weights)
        model.train()
        train_loss = 0.0
        train_cnt = 0
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(epoch + 1))
        for step,(dz_categorica_feature, dz_numeric_feature, person_feature, labels, len_dz, cfrnid, bv1_score,date_time_credit, inquery_feature, creditos_feature, len_inquery, len_creditos) in enumerate(train_dataloader):
            dz_categorica_feature = dz_categorica_feature.to(device)
            dz_numeric_feature = dz_numeric_feature.to(device)
            person_feature = person_feature.to(device)
            labels = labels.float().to(device)
            inquery_feature = inquery_feature.float().to(device)
            creditos_feature = creditos_feature.float().to(device)
            label_stander = labels[:,[3,5,7,9]]
            label_gt = labels[:,[0,1,2,4,6,8,10]]
            optimizer.zero_grad()
            outputs = model(dz_categorica_feature,dz_numeric_feature,person_feature,len_dz,inquery_feature, creditos_feature, len_inquery, len_creditos)
            total_loss = 0
            loss_log = []
            for idx,output in enumerate(outputs):
                label = label_gt[:,idx].unsqueeze(-1)
                Dice_loss = criterion1(output, label)
                Focal_Tversky_loss = criterion2(output, label)
                Dice_grad = torch.autograd.grad(Dice_loss, output, retain_graph=True, only_inputs=True)[0]
                Focal_Tversky_grad = torch.autograd.grad(Focal_Tversky_loss, output, retain_graph=True, only_inputs=True)[0]
                loss_weighter.update([Dice_grad, Focal_Tversky_grad])
                loss = Dice_loss * loss_weighter.weights[0] + Focal_Tversky_loss * loss_weighter.weights[1]
                loss_log.append(loss.item())   
                total_loss += multi_loss(Dice_loss, Focal_Tversky_loss)
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            if step % 100 == 0:
                msg = f'Epoch [{epoch+1}/{num_epochs}], Step [{step}],dob45dpd7_loss: {loss_log[0]:.4f}, dob90dpd7_loss: {loss_log[1]:.4f},dob90dpd30_loss: {loss_log[2]:.4f}, dob120dpd7_loss: {loss_log[3]:.4f},dob120dpd30_loss: {loss_log[4]:.4f}, dob180dpd7_loss: {loss_log[5]:.4f},dob180dpd30_loss: {loss_log[6]:.4f},  Loss: {total_loss.item():.4f}'
                write_log(msg)
            train_cnt += 1        
        scheduler.step()
        train_loss /= train_cnt

        model.eval()
        val_loss = 0.0
        val_preds_dob45dpd7 = []
        val_true_dob45dpd7 = []
        val_preds_dob90dpd7 = []
        val_true_dob90dpd7 = []
        val_preds_dob90dpd30 = []
        val_true_dob90dpd30 = []
        val_preds_dob120dpd7 = []
        val_true_dob120dpd7 = []
        val_preds_dob120dpd30 = []
        val_true_dob120dpd30 = []
        val_preds_dob180dpd7 = []
        val_true_dob180dpd7 = []
        val_preds_dob180dpd30 = []
        val_true_dob180dpd30 = []
        val_cnt = 0
        with torch.no_grad():
            for val_step,(dz_categorica_feature, dz_numeric_feature, person_feature, labels, len_dz, cfrnid, bv1_score,date_time_credit, inquery_feature, creditos_feature, len_inquery, len_creditos) in enumerate(val_dataloader):
                dz_categorica_feature = dz_categorica_feature.to(device)
                dz_numeric_feature = dz_numeric_feature.to(device)
                person_feature = person_feature.to(device)
                labels = labels.float().to(device)
                inquery_feature = inquery_feature.float().to(device)
                creditos_feature = creditos_feature.float().to(device)
                label_stander = labels[:,[3,5,7,9]]
                label_gt = labels[:,[0,1,2,4,6,8,10]]
                outputs = model(dz_categorica_feature,dz_numeric_feature,person_feature,len_dz,inquery_feature, creditos_feature, len_inquery, len_creditos)
                total_loss = 0
                loss_log = []
                for idx,output in enumerate(outputs):
                    label = label_gt[:,idx].unsqueeze(-1)
                    Dice_loss = criterion1(output, label)
                    Focal_Tversky_loss = criterion2(output, label)
                    loss_log.append(multi_loss(Dice_loss, Focal_Tversky_loss).item())
                    total_loss += multi_loss(Dice_loss, Focal_Tversky_loss)
                val_loss += total_loss.item()
                preds = outputs
                if val_step % 1000 == 0:
                    msg = f'Valid runing,Epoch [{epoch+1}/{num_epochs}],Step [{val_step}],dob45dpd7_loss: {loss_log[0]:.4f}, dob90dpd7_loss: {loss_log[1]:.4f},dob90dpd30_loss: {loss_log[2]:.4f},dob120dpd7_loss: {loss_log[3]:.4f},dob120dpd30_loss: {loss_log[4]:.4f}, dob180dpd7_loss: {loss_log[5]:.4f},dob180dpd30_loss: {loss_log[6]:.4f},  Loss: {total_loss.item():.4f}'
                    write_log(msg)
                
                val_preds_dob45dpd7.extend(preds[0].tolist())
                val_true_dob45dpd7.extend(label_gt[:,0].unsqueeze(-1).tolist())
                val_preds_dob90dpd7.extend(preds[1].tolist())
                val_true_dob90dpd7.extend(label_gt[:,1].unsqueeze(-1).tolist())
                val_preds_dob90dpd30.extend(preds[2].tolist())
                val_true_dob90dpd30.extend(label_gt[:,2].unsqueeze(-1).tolist()) 
                
                indices_of_ones = [index for index, value in enumerate(label_stander[:,0].tolist()) if value == 1]
                val_preds_dob120dpd7.extend([preds[3].tolist()[i] for i in indices_of_ones])
                val_true_dob120dpd7.extend([label_gt[:,3].unsqueeze(-1).tolist()[i] for i in indices_of_ones])
                
                indices_of_ones = [index for index, value in enumerate(label_stander[:,1].tolist()) if value == 1]
                val_preds_dob120dpd30.extend([preds[4].tolist()[i] for i in indices_of_ones])
                val_true_dob120dpd30.extend([label_gt[:,4].unsqueeze(-1).tolist()[i] for i in indices_of_ones])
                
                indices_of_ones = [index for index, value in enumerate(label_stander[:,2].tolist()) if value == 1]
                val_preds_dob180dpd7.extend([preds[5].tolist()[i] for i in indices_of_ones])
                val_true_dob180dpd7.extend([label_gt[:,5].unsqueeze(-1).tolist()[i] for i in indices_of_ones])
                
                indices_of_ones = [index for index, value in enumerate(label_stander[:,3].tolist()) if value == 1]
                val_preds_dob180dpd30.extend([preds[6].tolist()[i] for i in indices_of_ones])
                val_true_dob180dpd30.extend([label_gt[:,6].unsqueeze(-1).tolist()[i] for i in indices_of_ones])
                
                val_cnt += 1

        ks_dob45dpd7 = calculate_metrics(val_true_dob45dpd7, val_preds_dob45dpd7, epoch, num_epochs, train_loss, val_loss, "dob45dpd7")
        ks_target = calculate_metrics(val_true_dob90dpd7, val_preds_dob90dpd7, epoch, num_epochs, train_loss, val_loss, "dob90dpd7")
        ks_dob90dpd30 = calculate_metrics(val_true_dob90dpd30, val_preds_dob90dpd30, epoch, num_epochs, train_loss, val_loss, "dob90dpd30")
        ks_dob120dpd7 = calculate_metrics(val_true_dob120dpd7, val_preds_dob120dpd7, epoch, num_epochs, train_loss, val_loss, "dob120dpd7")
        ks_dob180dpd7 = calculate_metrics(val_true_dob180dpd7, val_preds_dob180dpd7, epoch, num_epochs, train_loss, val_loss, "dob180dpd7")
        ks_dob120dpd30 = calculate_metrics(val_true_dob120dpd30, val_preds_dob120dpd30, epoch, num_epochs, train_loss, val_loss, "dob120dpd30")
        ks_dob180dpd30 = calculate_metrics(val_true_dob180dpd30, val_preds_dob180dpd30, epoch, num_epochs, train_loss, val_loss, "dob180dpd30")
        
        if ks_target > best_ks:
            best_ks = ks_target
            best_epoch = epoch
            torch.save(model.state_dict(), "./model.pth")
            print('save best model in epoch {}, ks is {}'.format(epoch,ks_target))
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch 
        else:
            if epoch - best_epoch >= early_stop_patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()