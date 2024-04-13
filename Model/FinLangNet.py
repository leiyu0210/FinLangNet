import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.encode import *
from Modules.attention import *
from Modules.embedding import *

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
       
class MyModel_FinLangNet(nn.Module):
    def __init__(self,embedding_dim=64,hidden_dim=64,num_layers=2,num_head = 8):
        super(MyModel_FinLangNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_head = num_head
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
        # Classification token embedding 
        self.inquery_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.num_layers_circulo = self.num_layers * 2
        self.inquery_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head = self.num_head), self.hidden_dim, self.num_head),
                    self.hidden_dim,
                    self.num_head
                ) for l in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim)
        )      
        
        self.creditos_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head = self.num_head), self.hidden_dim, self.num_head),
                    self.hidden_dim,
                    self.num_head
                ) for l in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim)
        )     
        self.inquery_dropout =  nn.Dropout(0.1)
        self.creditos_dropout =  nn.Dropout(0.1)
        self.inquery_lstm_linear = nn.Linear(self.hidden_dim, 256)
        self.creditos_lstm_linear = nn.Linear(self.hidden_dim, 256)
        
        # category_feature
        
        self.creditos_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.category_fe_len = 4 + 237
        self.category_embedding_layers = nn.ModuleList([
                        nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim) for _ in range(self.category_fe_len)
                        ])
        
        # Embedding and encoder for time feature
        self.time_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.dz_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head = self.num_head), self.hidden_dim, self.num_head),
                    self.hidden_dim,
                    self.num_head
                ) for l in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim)
        )
        self.dz_dropout =  nn.Dropout(0.1)
        self.dz_lstm_linear = nn.Linear(self.hidden_dim, 256)  
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
        inquery_cls_tokens = self.inquery_cls_token_embedding.expand(x_inquery.size(0), -1, -1)
        inquery_embedded_features = torch.stack(inquery_embedded_features)
        inquery_embedded_features = torch.sum(inquery_embedded_features, dim=0)  #len*emb_dim
        inquery_embedded_features = torch.cat((inquery_cls_tokens, inquery_embedded_features), dim=1)  
        inquery_transformer_output, _ = self.inquery_encoder(inquery_embedded_features, attn_mask=None) # BS*length*hidden_dim
        inquery_transformer_output = self.act(inquery_transformer_output)
        inquery_transformer_output = self.inquery_dropout(inquery_transformer_output)
        inquery_out = self.inquery_lstm_linear(inquery_transformer_output[:,0,:])
        
        creditos_concatenated_features = []
        for i in range(self.creditos_fe_len):
            embedded_feature = self.creditos_embedding_layers[i](torch.as_tensor(x_creditos[:, i, :], dtype=torch.long) ) 
            creditos_concatenated_features.append(embedded_feature)
        creditos_cls_tokens = self.creditos_cls_token_embedding.expand(x_creditos.size(0), -1, -1)
        
        creditos_concatenated_features = torch.stack(creditos_concatenated_features)
        creditos_concatenated_features = torch.sum(creditos_concatenated_features, dim=0)  #len*emb_dim
        creditos_concatenated_features = torch.cat((creditos_cls_tokens, creditos_concatenated_features), dim=1)
        creditos_transformer_output, _ = self.creditos_encoder(creditos_concatenated_features, attn_mask=None) # BS*length*hidden_dim
        creditos_transformer_output = self.act(creditos_transformer_output)
        creditos_transformer_output = self.creditos_dropout(creditos_transformer_output)
        creditos_out = self.creditos_lstm_linear(creditos_transformer_output[:,0,:])
        
        # category_feature_encoder
        dz_feature_input =  torch.cat((dz_categorica_feature,dz_numeric_feature), dim=1) 
        dz_concatenated_features = []
        for i in range(self.category_fe_len):
            embedded_feature = self.category_embedding_layers[i](torch.as_tensor(dz_feature_input[:, i, :], dtype=torch.long) ) 
            dz_concatenated_features.append(embedded_feature)
            
        time_cls_tokens = self.time_cls_token_embedding.expand(dz_feature_input.size(0), -1, -1)   
        dz_concatenated_features = torch.stack(dz_concatenated_features)
        dz_concatenated_features = torch.sum(dz_concatenated_features, dim=0)  #len*emb_dim
        dz_concatenated_features = torch.cat((time_cls_tokens, dz_concatenated_features), dim=1)
        dz_transformer_output, _ = self.dz_encoder(dz_concatenated_features, attn_mask=None) # BS*length*hidden_dim
        dz_transformer_output = self.act(dz_transformer_output)
        dz_transformer_output = self.dz_dropout(dz_transformer_output)
        dz_out = self.dz_lstm_linear(dz_transformer_output[:,0,:])
        
        person_feature_cat = person_feature
        
        """FM one layer"""
        fm_1st_sparse_res = [emb(torch.as_tensor(person_feature_cat[:, i], dtype=torch.long).unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_part = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs,  1]
        
        """FM two layer"""
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
        x_concat = torch.cat((dz_out, person_out,inquery_out, creditos_out), dim=1)

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



    
    
