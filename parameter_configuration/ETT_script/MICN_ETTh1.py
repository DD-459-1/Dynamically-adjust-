

def MICN_ETTh1_96(args,  pred_len = 96, seq_len = 96):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_"+str(seq_len)+"_"+str(pred_len)
    args.model = "MICN"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = seq_len
    args.label_len = 96
    args.pred_len = pred_len
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.itr = 1


def MICN_ETTh1_192(args,  pred_len = 192, seq_len = 96):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_"+str(seq_len)+"_"+str(pred_len)
    args.model = "MICN"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = seq_len
    args.label_len = 96
    args.pred_len = pred_len
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.itr = 1

def MICN_ETTh1_336(args,  pred_len = 336, seq_len = 96):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_"+str(seq_len)+"_"+str(pred_len)
    args.model = "MICN"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = seq_len
    args.label_len = 96
    args.pred_len = pred_len
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.itr = 1

def MICN_ETTh1_720(args,  pred_len = 720, seq_len = 96):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_"+str(seq_len)+"_"+str(pred_len)
    args.model = "MICN"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = seq_len
    args.label_len = 96
    args.pred_len = pred_len
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.itr = 1
