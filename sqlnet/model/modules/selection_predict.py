import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        # two layer bi-lstm dropout -> 0.2
        # self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h / 2,num_layers=N_depth, batch_first=True,dropout=0.3, bidirectional=True)
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,num_layers=N_depth, batch_first=True,dropout=0.2, bidirectional=True)
        if use_ca:
            print "Using column attention on selection predicting"
            # self.sel_att = nn.Linear(N_h, N_h)

            # new attention model (additive attention)
            self.sel_att_out = nn.Sequential(nn.Tanh(),nn.Linear(N_h,1))
            self.sel_W1 = nn.Linear(N_h,N_h)
            self.sel_W2 = nn.Linear(N_h,N_h)
        else:
            print "Not using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, 1)

        # E_col
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.2, bidirectional=True)

        # U_q_sel
        self.sel_out_K = nn.Linear(N_h, N_h)
        # U_c_sel
        self.sel_out_col = nn.Linear(N_h, N_h)
        # nn.Linear is (u_a_sel).T
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()


    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        # compute the E_col
        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)

        if self.use_ca:
            # compute the hidden states output of lstm corresponding to question
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            # compute v
            # using multiplicative attention
            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))

            h_ext = h_enc.unsqueeze(1).unsqueeze(1)
            col_ext = e_col.unsqueeze(2).unsqueeze(2)
            # additive attention
            att_val = self.sel_att_out(self.sel_W1(h_ext) + self.sel_W2(col_ext)).squeeze()



            # print()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            # L-dimension attention weight
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)
            # compute E_Q|col
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            # normal method without column attention
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)

        # compute P_selcol(i|Q)
        sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col) ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score
