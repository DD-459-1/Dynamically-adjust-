import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
from torch.multiprocessing import Process, set_start_method
import sys



from parameter_configuration.ETT_script.PatchTST_ETTh1 import *
from parameter_configuration.ETT_script.PatchTST_ETTh2 import *
from parameter_configuration.ETT_script.PatchTST_ETTm1 import *
from parameter_configuration.ETT_script.PatchTST_ETTm2 import *
from parameter_configuration.ECL_script.PatchTST_ECL import *
from parameter_configuration.weather_script.PatchTST_weather import *


from parameter_configuration.ETT_script.TimesNet_ETTh1 import *
from parameter_configuration.ETT_script.TimesNet_ETTh2 import *
from parameter_configuration.ETT_script.TimesNet_ETTm1 import *
from parameter_configuration.ETT_script.TImesNet_ETTm2 import *
from parameter_configuration.ECL_script.TimesNet_ECL import *
from parameter_configuration.weather_script.TimesNet_weather import *

from parameter_configuration.ETT_script.TimeXer_ETTh1 import *
from parameter_configuration.ETT_script.TimeXer_ETTh2 import *
from parameter_configuration.ETT_script.TimeXer_ETTm1 import *
from parameter_configuration.ETT_script.TimeXer_ETTm2 import *
from parameter_configuration.ECL_script.TimeXer_ECL import *
from parameter_configuration.weather_script.TimeXer_weather import *


from parameter_configuration.ETT_script.MICN_ETTh1 import *
from parameter_configuration.ETT_script.MICN_ETTh2 import *
from parameter_configuration.ETT_script.MICN_ETTm1 import *
from parameter_configuration.ETT_script.MICN_ETTm2 import *
from parameter_configuration.ECL_script.MICN_ECL import *
from parameter_configuration.weather_script.MICN_weather import *



def Train_model(args, device, optimizer, patience, load_model_edition = "", print_to_file = False, log_file = ""):

    if print_to_file:
        sys.stdout = open(log_file, 'w')

    args.gpu = device
    args.optimizer = optimizer
    args.patience = patience

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.optimizer,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            min_val = exp.train(setting, best_optim_edition= load_model_edition)
            print("min_vali:",{min_val})

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
            return min_val
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.optimizer,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        sys.stdout.close()






if __name__ == '__main__':
    fix_seed = 123
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--load_model',default = False, help = "whether using pre_train model")

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--out_patience', type=int, default=2, help='early stopping patience ,when using DN-Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='select your optimizer, options: [DN-Adam, DN-Adam-lookahead, Adam, SGD, Lookahead-Adam, AMSGrad]')
    parser.add_argument('--explore', type=str, default=False, help="is explore method?")

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")



    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')



    func_dict = {
        "PatchTST_ETTh1": [PatchTST_ETTh1_96, PatchTST_ETTh1_192, PatchTST_ETTh1_336, PatchTST_ETTh1_720],
        "PatchTST_ETTh2": [PatchTST_ETTh2_96, PatchTST_ETTh2_192, PatchTST_ETTh2_336, PatchTST_ETTh2_720],
        "PatchTST_ETTm1": [PatchTST_ETTm1_96, PatchTST_ETTm1_192, PatchTST_ETTm1_336, PatchTST_ETTm1_720],
        "PatchTST_ETTm2": [PatchTST_ETTm2_96, PatchTST_ETTm2_192, PatchTST_ETTm2_336, PatchTST_ETTm2_720],
        "PatchTST_ECL": [PatchTST_ECL_96, PatchTST_ECL_192, PatchTST_ECL_336, PatchTST_ECL_720],
        "PatchTST_weather":[PatchTST_weather_96, PatchTST_weather_192, PatchTST_weather_336, PatchTST_weather_720],


        "TimesNet_ETTh1": [TimesNet_ETTh1_96, TimesNet_ETTh1_192, TimesNet_ETTh1_336, TimesNet_ETTh1_720],
        "TimesNet_ETTh2": [TimesNet_ETTh2_96, TimesNet_ETTh2_192, TimesNet_ETTh2_336, TimesNet_ETTh2_720],
        "TimesNet_ETTm1": [TimesNet_ETTm1_96, TimesNet_ETTm1_192, TimesNet_ETTm1_336, TimesNet_ETTm1_720],
        "TimesNet_ETTm2": [TimesNet_ETTm2_96, TimesNet_ETTm2_192, TimesNet_ETTm2_336, TimesNet_ETTm2_720],
        "TimesNet_ECL": [TimesNet_ECL_96, TimesNet_ECL_192, TimesNet_ECL_336 ,TimesNet_ECL_720],
        "TimesNet_weather": [TimesNet_weather_96, TimesNet_weather_192, TimesNet_weather_336, TimesNet_weather_720],

        "TimeXer_ETTh1": [TimeXer_ETTh1_96, TimeXer_ETTh1_192, TimeXer_ETTh1_336, TimeXer_ETTh1_720],
        "TimeXer_ETTh2": [TimeXer_ETTh2_96, TimeXer_ETTh2_192, TimeXer_ETTh2_336, TimeXer_ETTh2_720],
        "TimeXer_ETTm1": [TimeXer_ETTm1_96, TimeXer_ETTm1_192, TimeXer_ETTm1_336, TimeXer_ETTm1_720],
        "TimeXer_ETTm2": [TimeXer_ETTm2_96, TimeXer_ETTm2_192, TimeXer_ETTm2_336, TimeXer_ETTm2_720],
        "TimeXer_ECL": [TimeXer_ECL_96, TimeXer_ECL_192, TimeXer_ECL_336, TimeXer_ECL_720],
        "TimeXer_weather": [TimeXer_weather_96, TimeXer_weather_192, TimeXer_weather_336, TimeXer_weather_720],


        "MICN_ETTh1":[MICN_ETTh1_96, MICN_ETTh1_192, MICN_ETTh1_336, MICN_ETTh1_720],
        "MICN_ETTh2":[MICN_ETTh2_96, MICN_ETTh2_192, MICN_ETTh2_336, MICN_ETTh2_720],
        "MICN_ETTm1":[MICN_ETTm1_96, MICN_ETTm1_192, MICN_ETTm1_336, MICN_ETTm1_720],
        "MICN_ETTm2":[MICN_ETTm2_96, MICN_ETTm2_192, MICN_ETTm2_336, MICN_ETTm2_720],
        "MICN_ECL": [MICN_ECL_96, MICN_ECL_192, MICN_ECL_336, MICN_ECL_720],
        "MICN_weather": [MICN_weather_96, MICN_weather_192, MICN_weather_336, MICN_weather_720],
    }

    model_list = ["TimesNet"] # Choose the models to be used. options:[PatchTST, TimeXer, TimesNet, MICN]
    dataset_list = ["ETTh1"] # Select dataset
    optimizers_list = ["Adam", "DA_Adam", "DAs_Adam"] # Choose the optimizers to be used.
    # opions:[Adam, DA_Adam, DAs_Adam, AdaBelief, DA_AdaBelief, DAs_AdaBelief, Yogi, DA_Yogi, DAs_Yogi]

    for model_name in model_list:
        for datasetname in dataset_list:
            for i, optimizer in enumerate(optimizers_list):
                args = parser.parse_args()
                args.use_gpu = True if torch.cuda.is_available() else False
                args.model_name = model_name
                args.dataset = datasetname
                func_key = args.model_name + "_" + args.dataset
                func_list = func_dict[func_key]
                for fun in func_list:
                    fun(args)
                    Train_model(args, 0, optimizer, patience=3) # default training on cuda:0
                    print("++++++++++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++++++++++")
                if i < len(optimizers_list) - 1:
                    print("====================The optimizer has been switched====================")
            print("====================The dataset has been changed====================")
        print("====================The model has been changed====================")









