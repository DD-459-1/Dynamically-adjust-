

def PatchTST_ECL_96(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/electricity/"
    args.data_path = "electricity.csv"
    args.model_id = "ECL_96_96"
    args.model = "PatchTST"
    args.data = "custom"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
    args.des = "Exp"
    args.n_heads = 8
    args.batch_size = 64
    args.itr = 1

def PatchTST_ECL_192(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/electricity/"
    args.data_path = "electricity.csv"
    args.model_id = "ECL_96_192"
    args.model = "PatchTST"
    args.data = "custom"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 192
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
    args.des = "Exp"
    args.n_heads = 8
    args.batch_size = 64
    args.itr = 1

def PatchTST_ECL_336(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/electricity/"
    args.data_path = "electricity.csv"
    args.model_id = "ECL_96_336"
    args.model = "PatchTST"
    args.data = "custom"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 336
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
    args.des = "Exp"
    args.n_heads = 8
    args.batch_size = 64
    args.itr = 1

def PatchTST_ECL_720(args):
    args.task_name = "long_term_forecast"
    args.is_training = 1
    args.root_path = "./dataset/electricity/"
    args.data_path = "electricity.csv"
    args.model_id = "ECL_96_720"
    args.model = "PatchTST"
    args.data = "custom"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 720
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
    args.des = "Exp"
    args.n_heads = 8
    args.batch_size = 64
    args.itr = 1