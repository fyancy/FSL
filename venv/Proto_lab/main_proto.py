import torch
import os
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import visdom
from my_utils import Euclidean_Distance, t_sne
from proto_model import Protonet
from Data_generator_normalize import data_generate

generator = data_generate()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(net, save_path, train_x, vis, scenario='proto', ls_threshold=1e-5,
          n_epochs=100, n_way=3, n_episodes=30, shot=3, skip_lr=0.005):
    net.train()
    optimizer1 = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # nesterov=True; lr初始值设为0.1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)  # lr=lr∗gamma^epoch
    optimizer2 = torch.optim.Adam(net.parameters())
    optimizer = optimizer1

    n_shot = n_query = shot
    img_w = 2048
    chn = 1
    n_examples = train_x.shape[1]
    n_class = train_x.shape[0]
    assert n_class >= n_way
    assert n_examples >= n_shot + n_query

    print('train_data set Shape:', train_x.size())
    print('n_way=>', n_way, 'n_shot=>', n_shot, ' n_query=>', n_query)
    print("---------------------Training----------------------\n")
    counter = 0
    opt_flag = False
    avg_ls = torch.zeros([n_episodes]).to(device)

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            # permutation不改变原数组，copy并重新生成，不同于shuffle
            epi_classes = torch.randperm(n_class)[:n_way]
            support = torch.zeros([n_way, n_shot, img_w, chn], dtype=torch.float32)
            query = torch.zeros([n_way, n_query, img_w, chn], dtype=torch.float32)

            for i, epi_cls in enumerate(epi_classes):
                selected = torch.randperm(n_examples)[:n_shot + n_query]
                support[i] = train_x[epi_cls, selected[:n_shot]]
                query[i] = train_x[epi_cls, selected[n_shot:]]

            support, query = support.to(device), query.to(device)
            losses, ls_ac = net.forward(support, query)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            ls, ac = ls_ac['loss'], ls_ac['acc']
            avg_ls[epi] = ls
            if (epi + 1) % 5 == 0:
                vis.line(Y=[[ac, ls]], X=[counter],
                         update=None if counter == 0 else 'append', win=scenario,
                         opts=dict(legend=['accuracy', 'loss'], title=scenario))
                counter += 1
            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}' \
                      .format(ep + 1, n_epochs, epi + 1, n_episodes, ls, ac))

        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))
        if ls_ < skip_lr and opt_flag is False:
            optimizer = optimizer2
            print('Optimizer ==> Adam')
            opt_flag = True
        if (ls_ <= ls_threshold and ep + 1 >= 50) or ls_ < 0.5 * ls_threshold:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break
        scheduler.step(epoch=ep // 5)

    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, test_x, vis, scenario='test_proto', n_way=3, shot=2, eval_=False,
         n_episodes=200, n_epochs=1):
    # --------------------修改-------------------------
    net = Protonet().to(device)
    net.load_state_dict(torch.load(save_path))
    if eval_:
        net.eval()
    print("Load the model Successfully！\n%s" % save_path)
    n_s = n_q = shot
    img_w = 2048
    chn = 1

    n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert n_class >= n_way
    assert n_examples >= n_s + n_q

    print('test_data set Shape:', test_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, img_w))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, img_w))
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    sne_state = False
    counter = 0

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        for epi in range(n_episodes):
            epi_classes = torch.randperm(n_class)[:n_way]  # training
            support = torch.zeros([n_way, n_s, img_w, chn], dtype=torch.float32)
            query = torch.zeros([n_way, n_q, img_w, chn], dtype=torch.float32)

            for i, epi_cls in enumerate(epi_classes):
                selected = torch.randperm(n_examples)[:n_s + n_q]
                support[i] = test_x[epi_cls, selected[:n_s]]
                query[i] = test_x[epi_cls, selected[n_s:]]
            if epi == n_episodes - 1:  # if (epi+1) % 20 == 0:
                sne_state = True
            support, query = support.to(device), query.to(device)
            _, ls_ac = net.forward(xs=support, xq=query, vis=vis, sne_state=sne_state)
            sne_state = False

            ls, ac = ls_ac['loss'], ls_ac['acc']
            avg_acc += ac
            avg_loss += ls
            vis.line(Y=[[ac, ls]], X=[counter],
                     update=None if counter == 0 else 'append', win=scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=scenario))
            counter += 1
        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        print('[epoch {}/{}] => avg_loss: {:.8f}, avg_acc: {:.8f}'.format(ep + 1, n_epochs, avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    vis.text('Average Accuracy: {:.6f}  Average Loss:{:.6f}'.format(avg_acc_, avg_loss_), win='Test result')
    print('------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.6f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}\n'.format(avg_loss_))


def main(save_path, scenario, split, n_way=3, shot=2, samples=50, eval_=False,
         ls_threshold=1e-5, normalization=True):
    net = Protonet().to(device)
    print('%d GPU is available.' % torch.cuda.device_count())
    vis = visdom.Visdom(env='yancy_env')
    n_s = n_q = shot
    img_w = 2048

    train_x, test_x = generator.SQ_data_generator2(examples=100, split=split, way=n_way,
                                                   normalize=normalization, label=False)
    # train_x, test_x = generator.Cs_data_generator2(examples=samples, split=split,
    #                                                normalize=normalization, label=False)
    # train_x, test_x = generator.SQ_data_generator3(examples=samples, split=split, way=n_way,
    #                                                normalize=normalization, label=False)
    # train_x = train_x[:n_way, :split]
    n_class = train_x.shape[0]
    assert n_class == n_way

    print('train_data shape:', train_x.shape)
    print('test_data shape:', test_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, img_w))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, img_w))
    train_x, test_x = torch.from_numpy(train_x).float(), torch.from_numpy(test_x).float()

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            print('The training file exists：%s' % save_path)
        else:
            train(net=net, save_path=save_path, train_x=train_x, vis=vis,
                  scenario=scenario, ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            test(save_path=save_path, test_x=test_x, vis=vis, n_way=n_way, shot=shot, eval_=eval_)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':  # train10, train13
    import matplotlib.pyplot as plt

    # save_dir = r'G:\comparison_code'
    save_dir = r'G:\comparison_code\comparison_add'
    model_name = 'sq7_proto_5shot_30_0.pkl'
    path = os.path.join(save_dir, model_name)
    way = 7
    shot = 5
    sample = 50  # cw: 50; sq: 100
    n_train_per_file = 30  # total number of training examples each way
    split = n_train_per_file  # split the data, split=>train, the rest=>test
    assert shot + shot <= split * 4
    print('File exist?', os.path.exists(path))

    main(save_path=path, scenario='proto', split=split, n_way=way,
         eval_=True, samples=sample, shot=shot, ls_threshold=1e-4)
    main(save_path=path, scenario='proto_', split=split, n_way=way,
         eval_=False, samples=sample, shot=shot, ls_threshold=1e-5)
    plt.show()
    # model_name = 'train_39_query15.pkl'
    # path = os.path.join(save_dir, model_name)
    # train(normalization=True, save_path=path, train_scenario='train-39')
    # plot(ac, ls)
    # graph_view()
