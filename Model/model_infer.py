import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from Model.sentences_load import val_dataloader
from Model.FinLangNet import MyModel_FinLangNet
from other_models.gru_deepfm import MyModel_GRU
from other_models.gru_attention_deepfm import MyModel_GRU_Attention
from other_models.lstm_deepfm import MyModel_LSTM
from other_models.transformer_deepfm import MyModel_Transformer
from other_models.stack_gru_deepfm import MyModel_StackGRU


def initialize_model():
    """Initialize and return the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel_FinLangNet().to(device)
    model.load_state_dict(torch.load('./results/FinLangNet.pth'))
    model.eval()
    return model, device

def evaluate_model(model, device, val_dataloader):
    """Evaluate the model and return lists of predictions, actuals, and other information."""
    val_preds, val_true, val_score = ([[] for _ in range(7)] for _ in range(3))
    cfrnid_list, slice_list, bcardv6_list = [], [], []
    
    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data]
            outputs, labels, additional_info = process_data_step(model, device, data)
            
            update_results_lists(val_preds, val_true, val_score, labels, outputs)
            cfrnid_list.extend(additional_info['cfrnid'])
            slice_list.extend(additional_info['date_time_credit_formatted'])
            bcardv6_list.extend(additional_info['bv1_score'])

            if step % 100 == 0:
                print(f'Step [{step}]')
    
    return val_preds, val_true, val_score, cfrnid_list, slice_list, bcardv6_list

def process_data_step(model, device, data):
    """Process a batch of data and return model outputs and additional information."""
    processed_data = unpack_data(data)
    
    outputs = model(*processed_data['inputs'])
    additional_info = {
        'cfrnid': [str(i) for i in processed_data['cfrnid'].tolist()],
        'bv1_score': [float(i) for i in processed_data['bv1_score'].tolist()],
        'date_time_credit_formatted': [f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}" for date in processed_data['date_time_credit'].tolist()]
    }
    return outputs, processed_data['labels'], additional_info

def unpack_data(data):
    """Unpack data from dataloader and organize it for processing."""
    # Assume data follows a specific structure as per input example
    return {
        'inputs': data[:-6],  # Assuming first sets are inputs to the model
        'labels': data[3],  # Assuming labels are at a specific index
        'cfrnid': data[-6],
        'bv1_score': data[-5],
        'date_time_credit': data[-4],
        # Extend or modify as per actual data structure
    }

def update_results_lists(val_preds, val_true, val_score, labels, outputs):
    """Update results lists with predictions, true values, and scores."""
    # This function would contain the logic for updating the prediction,
    # true, score lists from the outputs (assuming a similar structure)
    pass  # Placeholder for actual logic
