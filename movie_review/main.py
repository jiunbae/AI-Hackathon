# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

from dataset import MovieReviewDataset
from process import preprocess
from model import CNNReg

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        data = preprocess(raw_data, config.vocasize, config.minlen, config.maxlen)
        model.eval()

        prediction = model(data)
        point = prediction.data.squeeze(dim=1).tolist()

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=1001)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--vocasize', type=int, default=20000)
    args.add_argument('--embedding', type=int, default=200)
    args.add_argument('--lr', type=float, default=.01)
    args.add_argument('--maxlen', type=int, default=100)
    args.add_argument('--minlen', type=int, default=3)
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--cudaseed', type=int, default=1234)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.cudaseed)

    model = CNNReg(config.vocasize, config.embedding, config.maxlen, GPU_NUM)
    if GPU_NUM: model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause: nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        dataset = MovieReviewDataset(DATASET_PATH, config.vocasize, config.minlen, config.maxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=lambda data: zip(*data),
                                  num_workers=2)
        total_batch = len(train_loader)
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)

                labels = Variable(torch.from_numpy(np.array(labels)))
                if GPU_NUM: labels = labels.cuda()

                loss = criterion(predictions, labels)
                if GPU_NUM: loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.data[0]

            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            if not (epoch % 100):
                print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
                nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
