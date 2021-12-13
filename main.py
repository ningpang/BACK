import torch
from model.back_model import BACK
from arguments import get_args_parser
from framework import framework
from dataloader import get_loader, get_word_vec

if __name__ == '__main__':
    args = get_args_parser()
    print('{}-way-{}-shot Few-shot Text Classification on {}'.format(args.N, args.K, args.dataset))
    base_dir = './datasets/'
    word2id, word_vec_mat = get_word_vec(args, base_dir)
    train_data_loader = get_loader(args, base_dir, 'train', word2id)
    val_data_loader = get_loader(args, base_dir, 'test', word2id)

    model = BACK(args, word_vec_mat)

    FTCFramework = framework(args, train_data_loader, val_data_loader, val_data_loader)
    print('Begin training ...')
    FTCFramework.train(model)
