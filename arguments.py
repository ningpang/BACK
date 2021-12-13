import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='huffpost', type=str)
    parser.add_argument('--max_length', default=40, type=int)
    parser.add_argument('--case_sensitive', default=False, type=bool)
    parser.add_argument('--batch_size', default=1, type=int)


    # setting
    parser.add_argument('--N_for_train', default=15, type=int)
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--K', default=5, type=int)
    parser.add_argument('--Neg_K', default=10, type=int)
    parser.add_argument('--Q', default=2, type=int)

    # model
    parser.add_argument('--model', default='back', type=str)
    parser.add_argument('--embedding_size', default=50, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)


    #training
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_step_size', default=10000, type=int)
    parser.add_argument('--train_iter', default=50000, type=int)
    parser.add_argument('--val_iter', default=1000)
    parser.add_argument('--val_step', default=2000)
    parser.add_argument('--test_iter', default=2000)

    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    args.save_dict = './save_dict/'+ args.model

    return args