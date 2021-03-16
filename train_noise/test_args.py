import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', action='store', default='cuda:0',
        help='input the device you want to use')
parser.add_argument('--BS', action='store', type=int, default=128,
        help='input the batch size')
parser.add_argument('--fname_head', action='store', default="GCONV_state_dict",
        help='input the filename')
parser.add_argument('--method', action='store', choices = ["normal", "noise", "adv"], default="adv", 
        help='input the training method')
parser.add_argument('--epochs', action='store', type=int, default=5, 
        help='input the number of epochs for training')
parser.add_argument('--adv_num', action='store', type=int, default=10000, 
        help='input the number of samples for adv train')
parser.add_argument('--test_run', action='store', type=int, default=10, 
        help='input the number of runs for noisy test')
args = parser.parse_args()

print(args)