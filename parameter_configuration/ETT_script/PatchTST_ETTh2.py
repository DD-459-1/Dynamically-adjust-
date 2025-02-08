


def PatchTST_ETTh2_96(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh2.csv"
    args.model_id = "ETTh2_96_96"
    args.model = "PatchTST"
    args.data = "ETTh2"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.n_heads = 4
    args.itr = 1

def PatchTST_ETTh2_192(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh2.csv"
    args.model_id = "ETTh2_96_192"
    args.model = "PatchTST"
    args.data = "ETTh2"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 192
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.n_heads = 8
    args.itr = 1

def PatchTST_ETTh2_336(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh2.csv"
    args.model_id = "ETTh2_96_336"
    args.model = "PatchTST"
    args.data = "ETTh2"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 336
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.n_heads = 8
    args.itr = 1

def PatchTST_ETTh2_720(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh2.csv"
    args.model_id = "ETTh2_96_720"
    args.model = "PatchTST"
    args.data = "ETTh2"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 720
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.des = "Exp"
    args.n_heads = 4
    args.itr = 1
