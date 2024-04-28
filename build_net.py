import torch
import torch.nn as nn

from Net import *


def build_net(args):
    print("[Build Net]")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 모델 초기화
    input_dim = 7  # 입력 차원
    hidden_dim = 100  # 은닉 차원
    layer_dim = 1  # LSTM 레이어 수
    net = LSTM_net.LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim=3).to(device)

    # # load pretrained parameters
    # if args.mode == 'train':
    #     param = torch.load(f"./pretrained/{args.train_subject[0]-1}/checkpoint/500.tar")
    #     net.load_state_dict(param['net_state_dict'])
    #
    # # test only
    # else:
    #     param = torch.load(f"./tl/{args.train_subject[0]-1}/checkpoint/50.tar")
    #     net.load_state_dict(param['net_state_dict'])

    # Set GPU
    if args.gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if args.gpu == "multi":
            device = args.gpu
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
        net.cuda()

    # Set CPU
    else:
        device = torch.device("cpu")

    # Print
    print(f"device: {device}")
    print("")

    return net
