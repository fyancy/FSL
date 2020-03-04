from DaNN_model import DaNN
import torch
import numpy as np
import torch.nn as nn
# from tqdm import tqdm
import mmd
import visdom
import os
from Data_generator_normalize import data_generate

generator = data_generate()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# BATCH_SIZE = 5
# -----------------------------
# the hyper parameters follow the original paper as follows:
LAMBDA = 0.25
GAMMA = 10 ** 3
LEARNING_RATE = 0.02
MOMEMTUM = 0.05
L2_WEIGHT = 0.003


# 实验中发现，optimizer选择Adam比SGD效果更好
# ------------------------------


def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def compute_acc(out, y):
    """
    :param out: the result of classifier, didn't go through softmax layer
    :param y: labels
    :return: accuracy
    """
    prob = nn.functional.log_softmax(out, dim=-1)
    pre = torch.max(prob, dim=1)[1]
    # print('y_label:\n', y.cpu().numpy())
    # print('predicted:\n', pre.cpu().numpy())
    acc = torch.eq(pre, y).float().mean().cpu().item()
    return acc


def train(net, vis, save_path, train_x, train_y, tar_x, ls_threshold,
          scenario, n_way=3, shot=3, n_episodes=30, n_epochs=200, fine_tune=False):
    if fine_tune:
        net.load_state_dict(torch.load(save_path))
        print('load the model!')
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE,
    #                             momentum=MOMEMTUM, weight_decay=L2_WEIGHT)
    criterion = nn.CrossEntropyLoss()

    img_w = 2048
    chn = 1
    n_examples = train_x.shape[1]
    n_episodes = n_examples // shot
    tar_num = tar_x.shape[1]
    n_class = train_x.size(0)

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('train_x Shape:', train_x.shape)
    print('train_y Shape:', train_y.shape)
    print('train_x Shape:', train_x.shape)
    print('target_x Shape:', tar_x.shape)
    print("---------------------Training----------------------\n")
    avg_ls = torch.zeros([n_episodes]).to(device)
    avg_cls = torch.zeros([n_episodes]).to(device)
    avg_mls = torch.zeros([n_episodes]).to(device)
    counter = 0

    for ep in range(n_epochs):
        count = 0
        for epi in range(n_episodes):
            x, y = train_x[:, count:count + shot], train_y[:, count:count + shot]
            selected = torch.randperm(tar_num)[:shot]
            x_tar = tar_x[:, selected]
            x, y, x_tar = x.to(device), y.contiguous().view(-1).to(device), x_tar.to(device)
            count += shot

            # epi_classes = torch.randperm(n_class)[:n_way]  # training
            # x = torch.zeros([n_way, shot, img_w, chn], dtype=torch.float32)
            # x_tar = torch.zeros([n_way, shot, img_w, chn], dtype=torch.float32)
            # y = torch.zeros([n_way, shot], dtype=torch.long)
            # for i, epi_cls in enumerate(epi_classes):
            #     selected = torch.randperm(n_examples)[:shot]
            #     x[i] = train_x[epi_cls, selected[:shot]]
            #     y[i] = train_y[epi_cls, selected[:shot]]
            #     selected = torch.randperm(tar_num)[:shot]  # 选取目标域数据
            #     x_tar[i] = tar_x[epi_cls, selected[:shot]]

            x, y, x_tar = x.to(device), y.contiguous().view(-1).to(device), x_tar.to(device)
            y_src, x_src_mmd, x_tar_mmd = net.forward(src=x, tar=x_tar)
            # print(x_src_mmd.shape)
            # print(x_tar_mmd.shape)

            outputs = y_src
            loss_c = criterion(outputs, y)
            loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
            loss = loss_c + LAMBDA * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_ls[epi] = loss
            avg_cls[epi] = loss_c
            avg_mls[epi] = loss_mmd
            acc = compute_acc(outputs, y)
            if (epi + 1) % 5 == 0 or epi == 0:
                vis.line(Y=[[acc, loss.cpu().item()]], X=[counter],
                         update=None if counter == 0 else 'append', win=scenario + '_train',
                         opts=dict(legend=['accuracy', 'loss'], title=scenario + '_train'))
            counter += 1

            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, loss.cpu().item(), acc))

        ls_ = torch.mean(avg_ls).cpu().item()
        ls_c = avg_cls.mean().cpu().item()
        ls_m = avg_mls.mean().cpu().item()

        if ls_ <= ls_threshold and ep + 1 >= 40:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break

        print('[epoch {}/{}] => train_loss: {:.8f}, c_loss:{:.8f}, mmd_loss:{:.8f}\n'.format(
            ep + 1, n_epochs, ls_, ls_c, ls_m))

    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(vis, save_path, test_x, test_y, tar_x, scenario='DaNN_TEST', eval_=False, n_way=3, shot=3):
    net = DaNN(n_class=n_way).to(device)
    criterion = nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(save_path))
    if eval_:
        net.eval()
    print('load the model successfully!')
    n_examples = test_x.shape[1]
    tar_num = tar_x.shape[1]
    n_episodes = n_examples // shot
    n_epochs = 1

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    print('target_x Shape:', tar_x.shape)
    print("---------------------Testing----------------------\n")
    avg_ls = torch.zeros([n_episodes]).to(device)
    avg_acc_ = 0.
    avg_loss_ = 0.
    counter = 0

    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        count = 0
        for epi in range(n_episodes):
            x, y = test_x[:, count:count + shot], test_y[:, count:count + shot]
            selected = torch.randperm(tar_num)[:shot]
            t_x = tar_x[:, selected]
            x, y, t_x = x.to(device), y.contiguous().view(-1).to(device), t_x.to(device)
            count += shot

            x, y, x_tar = x.to(device), y.contiguous().view(-1).to(device), t_x.to(device)
            y_src, _, _ = net.forward(src=x, tar=t_x)

            outputs = y_src
            loss_c = criterion(outputs, y)
            # loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
            loss = loss_c  # + LAMBDA * loss_mmd
            acc = compute_acc(outputs, y)
            avg_loss += loss.cpu().item()
            avg_acc += acc

            vis.line(Y=[[acc, loss.cpu().item()]], X=[counter],
                     update=None if counter == 0 else 'append', win=scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=scenario))
            counter += 1
            print('[Episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(epi + 1, n_episodes, loss.cpu().item(), acc))

        avg_acc /= n_episodes
        avg_loss /= n_episodes
        avg_acc_ += avg_acc
        avg_loss_ += avg_loss
        print('[epoch {}/{}] => avg_loss: {:.8f}, avg_acc: {:.8f}'.format(ep + 1, n_epochs, avg_loss, avg_acc))
    avg_acc_ /= n_epochs
    avg_loss_ /= n_epochs
    vis.text('Average Accuracy: {:.6f}  Average Loss:{:.6f}'.format(avg_acc_, avg_loss_), win='Test result')
    print('\n------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.6f}'.format(avg_acc_))
    print('Average Test Loss: {:.6f}\n'.format(avg_loss_))


def main(save_path, scenario='DaNN', normalization=True, ls_threshold=1e-3,
         n_way=3, samples=20, shot=3, split=10, eval_=False, fine_tune=False):
    """
    train and test
    :param fine_tune:
    :param eval_:
    :param samples:
    :param scenario:
    :param save_path:
    :param normalization:
    :param ls_threshold:
    :param n_way:
    :param shot:
    :param split:
    """
    print('%d GPU is available.' % torch.cuda.device_count())
    net = DaNN(n_class=n_way).to(device)
    vis = visdom.Visdom(env='yancy_env')
    # train_x, train_y, test_x, test_y = generator.Cs_data_generator2(examples=samples, split=split, way=n_way,
    #                                                                 normalize=normalization, label=True)
    train_x, train_y, test_x, test_y = generator.SQ_data_generator2(examples=samples, split=split, way=n_way,
                                                                    normalize=normalization, label=True)
    # train_x, train_y, test_x, test_y = generator.SQ_data_generator3(examples=samples, split=split, way=n_way,
    #                                                                 normalize=normalization, label=True)
    # _, _, target_x, _ = generator.Cs_data_generator2(examples=samples, split=split,
    #                                                  normalize=normalization, label=True)

    # _, _, test_x, test_y = generator.SQ_data_generator2(examples=100, split=0, way=n_way,
    #                                                     normalize=normalization, label=True)
    # _, _, target_x, _ = generator.SQ_data_generator2(examples=samples*2, split=split, way=n_way,
    #                                                  normalize=normalization, label=True)
    target_x = test_x[:, :60]
    # target_x = target_x[:, :4*shot]
    print('train_x Shape:', train_x.shape)
    print('train_y Shape:', train_y.shape)
    print('target_x Shape:', target_x.shape)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()
    target_x = torch.from_numpy(target_x).float()

    if os.path.exists(save_path):
        print('The training file exists：%s' % save_path)
        fine_tune = os.path.exists(save_path)
    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        train(net=net, save_path=save_path, train_x=train_x, train_y=train_y, tar_x=target_x, vis=vis,
              scenario=scenario, ls_threshold=ls_threshold, n_way=n_way, shot=shot, fine_tune=fine_tune)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            test(save_path=save_path, test_x=test_x, test_y=test_y, tar_x=target_x,
                 vis=vis, n_way=n_way, shot=shot, eval_=eval_)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':
    save_dir = r'G:\comparison_code\comparison_add'
    model_name = 'sq7_DaNN_1shot_30_0.pkl'
    path = os.path.join(save_dir, model_name)
    way = 7
    n_shot = 1
    sample = 100  # samples from each file; cw: 50; sq: 100
    num_per_file = 30  # total number of examples per class
    split = num_per_file  # split the data, split=>train, the rest=>test

    print('File exist? ', os.path.exists(path))
    # main(save_path=path, n_way=way, split=split, samples=sample,
    #      shot=n_shot, eval_=True, ls_threshold=1e-5)
    main(save_path=path, n_way=way, split=split, samples=sample,
         shot=n_shot, eval_=False, ls_threshold=1e-3)
