

def TimeXer_ETTh1_96(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_96_96"
    args.model = "TimeXer"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.e_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 256
    args.d_ff = 1024
    args.batch_size = 64
    args.des = "Exp"
    args.itr = 1

def TimeXer_ETTh1_192(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_96_192"
    args.model = "TimeXer"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 192
    args.e_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 256
    args.batch_size = 64
    args.d_ff = 1024
    args.des = "Exp"
    args.itr = 1

def TimeXer_ETTh1_336(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_96_336"
    args.model = "TimeXer"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 336
    args.e_layers = 2
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 512
    args.batch_size = 64
    args.d_ff = 1024
    args.des = "Exp"
    args.itr = 1

def TimeXer_ETTh1_720(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh1.csv"
    args.model_id = "ETTh1_96_720"
    args.model = "TimeXer"
    args.data = "ETTh1"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 720
    args.e_layers = 2
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 256
    args.batch_size = 64
    args.d_ff = 1024
    args.des = "Exp"
    args.itr = 1