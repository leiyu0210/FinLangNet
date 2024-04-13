import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class DependencyLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DependencyLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dependent = False
    
    def forward(self, x, dependency=None):
        x = self.linear(x)
        if self.dependent and dependency is not None:
            x = x + dependency
        return x
    
class MaskedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.1, bidirectional=False):
        super(MaskedGRU, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens):
        # input_tensor shape: batch_size*time_step*dim , seq_lens: (batch_size,)  when batch_first = True
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = rnn_utils.pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
        y_padded, length = rnn_utils.pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        return y_padded, hidden


class MyModel_GRU(nn.Module):
    def __init__(self,embedding_dim=16, hidden_dim=256, num_layers=2, bidirectional=False):
        super(MyModel_GRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_direction = 2 if bidirectional else 1   
        self.inquery_category_index = list(range(3+2))
        self.inquery_fe_len = 3+2
        self.creditos_category_index = list(range(8+21))
        self.creditos_fe_len = 8+21
        self.encode_dim = 128

        #  embedding layers
        self.inquery_embedding_layers = nn.ModuleList([
                                    nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim) for _ in range(self.inquery_fe_len)
                                               ])
        self.creditos_embedding_layers = nn.ModuleList([
                                    nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim) for _ in range(self.creditos_fe_len)
                                        ])
       
        self.num_layers_circulo = self.num_layers * 2
        self.lstm_inquery = MaskedGRU(input_size= self.inquery_fe_len * self.embedding_dim , hidden_size = self.hidden_dim, num_layers= self.num_layers_circulo, batch_first=True, bidirectional= self.bidirectional)

        
        self.lstm_creditos = MaskedGRU(input_size= self.creditos_fe_len * self.embedding_dim   , hidden_size= self.hidden_dim, num_layers=self.num_layers_circulo, batch_first=True, bidirectional= self.bidirectional)
        self.inquery_hidden_dropout =  nn.Dropout(0.2)
        self.creditos_hidden_dropout =  nn.Dropout(0.2)
        self.inquery_lstm_linear = nn.Linear(self.hidden_dim * self.num_layers_circulo * num_direction, 256)
        self.creditos_lstm_linear = nn.Linear(self.hidden_dim * self.num_layers_circulo * num_direction, 256)
        
        # category_feature
        
        self.category_fe_len = 4 + 237
        self.category_embedding_layers = nn.ModuleList([
                        nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim) for _ in range(self.category_fe_len)
                        ])
        
        # Embedding and encoder for time feature
         # time model
        self.lstm = MaskedGRU(input_size= self.category_fe_len * self.embedding_dim , hidden_size= self.hidden_dim, num_layers= self.num_layers, batch_first=True, bidirectional= self.bidirectional)  
        self.lstm_hidden_dropout =  nn.Dropout(0.2)
        self.lstm_linear = nn.Linear(self.hidden_dim * self.num_layers * num_direction, 256)
        
        # Activation function
        self.act = F.gelu
           
        # personal_feature
        self.person_fe_len = 3 + 8
        self.personal_embedding_layers = nn.ModuleList([
                                nn.Embedding(num_embeddings=32, embedding_dim=self.embedding_dim) for _ in range(self.person_fe_len)
                                ])
        self.personal_feature_linear = nn.Linear(self.person_fe_len*self.embedding_dim, 128)
        
        """FM"""
        # one layer
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(32, 1) for _ in range(self.person_fe_len)])  
        
        # two layer
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(32, self.embedding_dim) for _ in range(self.person_fe_len)])  
        
        """DNN"""
        hid_dims=[256, 128]
        self.all_dims = [self.person_fe_len * self.embedding_dim] + hid_dims
        # for DNN 
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(0.2))
        
        self.person_final_linear = nn.Linear( self.person_fe_len+self.embedding_dim+128, 256)
           
        # Definition for combining layers
        self.multihead_name = ['dob45dpd7','dob90dpd7','dob90dpd30','dob120dpd7','dob120dpd30','dob180dpd7','dob180dpd30']
        
        in_size = 256 + 256 + 256 + 256
        hidden_sizes = [512, 256, 128, 32]
        self.dropout_prob = 0.2
        self.multihead_dict = nn.ModuleDict()        
        branch_layers = {}
        for hidden_size in hidden_sizes:
            layers = [
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=self.dropout_prob),
            ]
            branch_layers[hidden_size] = nn.Sequential(*layers)
            in_size = hidden_size
        # setting labels relation
        for name in self.multihead_name:
            layers = []
            dpb, dpd = name.split('dpd')
            if 'dpd' in name:
                for hs in hidden_sizes:
                    layers.append(branch_layers[hs])
                dependent_layer = DependencyLayer(hidden_sizes[-1], 1)
                dependent_layer.dependent = dpd != '7'
                layers.append(dependent_layer)
            else:
                out_size = hidden_sizes[-1]
                layers.append(nn.Linear(out_size, 1))
            
            # head - Sequential
            self.multihead_dict[name] = nn.Sequential(*layers)
        
        self.sigmoid = nn.Sigmoid()       
        

    def forward(self, dz_categorica_feature, dz_numeric_feature, person_feature,len_dz,x_inquery, x_creditos,inquery_length,creditos_length):
        # x_inquery : BS * feature_dim * length
        inquery_embedded_features = []
        for i in range(self.inquery_fe_len):
            embedded_feature = self.inquery_embedding_layers[i](torch.as_tensor(x_inquery[:, i, :], dtype=torch.long) ) 
            inquery_embedded_features.append(embedded_feature)
        inquery_concatenated_features = torch.cat(inquery_embedded_features, dim=2)  # BS*length*emb_dim
        out_inquery, hidden_inquery = self.lstm_inquery(inquery_concatenated_features, inquery_length)
        hidden_inquery = self.inquery_hidden_dropout(hidden_inquery)
        hidden_inquery = torch.transpose(hidden_inquery, 0, 1).contiguous()
        hidden_inquery = hidden_inquery.view(hidden_inquery.shape[0], -1)
        inquery_out = self.inquery_lstm_linear(hidden_inquery)
        
        creditos_embedded_features = []
        for i in range(self.creditos_fe_len):
            embedded_feature = self.creditos_embedding_layers[i](torch.as_tensor(x_creditos[:, i, :], dtype=torch.long) ) 
            creditos_embedded_features.append(embedded_feature)
        creditos_concatenated_features = torch.cat(creditos_embedded_features, dim=2)
        out_creditos, hidden_creditos= self.lstm_creditos(creditos_concatenated_features,creditos_length)
        hidden_creditos = self.creditos_hidden_dropout(hidden_creditos)
        hidden_creditos = torch.transpose(hidden_creditos, 0, 1).contiguous()
        hidden_creditos = hidden_creditos.view(hidden_creditos.shape[0], -1)
        creditos_out = self.creditos_lstm_linear(hidden_creditos)
        
        # category_feature_encoder
        dz_feature_input =  torch.cat((dz_categorica_feature,dz_numeric_feature), dim=1) 
        category_embedded_features = []
        for i in range(self.category_fe_len):
            embedded_feature = self.category_embedding_layers[i](torch.as_tensor(dz_feature_input[:, i, :], dtype=torch.long) ) 
            category_embedded_features.append(embedded_feature)
        category_embedded_features = torch.cat(category_embedded_features, dim=2)  # BS*length*(emb_dim*cate_len)
        time_input_features = category_embedded_features
        time_state, time_hidden = self.lstm(time_input_features, len_dz)
        time_hidden = self.lstm_hidden_dropout(time_hidden)
        time_hidden = torch.transpose(time_hidden, 0, 1).contiguous()
        time_hidden = time_hidden.view(time_hidden.shape[0], -1)
        time_out = self.lstm_linear(time_hidden)
        
        person_feature_cat = person_feature
        
        """FM 1st"""
        fm_1st_sparse_res = [emb(torch.as_tensor(person_feature_cat[:, i], dtype=torch.long).unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_part = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs,  1]
        
        """FM 2st"""
        fm_2nd_order_res = [emb(torch.as_tensor(person_feature_cat[:, i], dtype=torch.long).unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  
        
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        
        
        """DNN"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]
        
        for i in range(1, len(self.all_dims)): 
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
            
        person_cat_out = torch.cat((fm_1st_sparse_res,sub,dnn_out),dim=1)   # [bs, self.person_fe_len+self.embedding_dim+128]
        person_out = self.person_final_linear(person_cat_out)

        x_concat = torch.cat((time_out, person_out,inquery_out, creditos_out), dim=1)

        final_output = []
        output_history = {}

        for name in self.multihead_name:
            parts = name.replace('dob', '').split('dpd')
            dob = int(parts[0])
            dpd = int(parts[1])

            head_output = self.multihead_dict[name](x_concat)
            for prev_name in self.multihead_name:
                prev_dob, prev_dpd = prev_name.replace('dob','').split('dpd')
                prev_dob = int(prev_dob)
                prev_dpd = int(prev_dpd)

                if dob > prev_dob and dpd == prev_dpd:
                    head_output += output_history[prev_name]

            head_output = self.sigmoid(head_output)
            final_output.append(head_output)
            output_history[name] = head_output
  
        return final_output

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



    