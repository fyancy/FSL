import torch
import torch.nn as nn
import visdom
import os
from cnn_Data_generator_normalize import data_generate
from CNN_model import CNN

generator = data_generate()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_acc(out, y):
    """
    :param out: the result of classifier
    :param y: labels
    :return: accuracy
    """
    prob = nn.functional.log_softmax(out, dim=-1)
    pre = torch.max(prob, dim=1)[1]
    # print('y_label:\n', y.cpu().numpy())
    # print('predicted:\n', pre.cpu().numpy())
    acc = torch.eq(pre, y).float().mean().cpu().item()
    return acc


def train(net, save_path, train_x, train_y, vis, train_scenario, ls_threshold=1e-5, n_epochs=200,
          n_way=3, n_episodes=20, shot=3):
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    img_w = 2048
    chn = 1
    n_examples = train_x.size(1)
    n_class = train_x.size(0)

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('train_x Shape:', train_x.shape)
    print('train_y Shape:', train_y.shape)
    print("---------------------Training----------------------\n")
    counter = 0
    n_episodes = n_examples // shot
    avg_ls = torch.zeros([n_episodes]).to(device)

    for ep in range(n_epochs):
        count = 0
        for epi in range(n_episodes):
            x, y = train_x[:, count:count + shot], train_y[:, count:count + shot]
            x, y = x.to(device), y.contiguous().view(-1).to(device)
            count += shot

            # epi_classes = torch.randperm(n_class)[:n_way]  # training
            # x = torch.zeros([n_way, shot, img_w, chn], dtype=torch.float32)
            # y = torch.zeros([n_way, shot], dtype=torch.long)
            #
            # for i, epi_cls in enumerate(epi_classes):
            #     selected = torch.randperm(n_examples)[:shot]
            #     x[i] = train_x[epi_cls, selected[:shot]]
            #     y[i] = train_y[epi_cls, selected[:shot]]

            x, y = x.to(device), y.contiguous().view(-1).to(device)
            outputs = net.forward(x)
            losses = loss(outputs, y)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            avg_ls[epi] = losses
            acc = compute_acc(outputs, y)

            if (epi + 1) % 10 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, losses.cpu().item(), acc))
            if (epi + 1) % 5 == 0 or epi == 0:
                vis.line(Y=[[acc, losses.cpu().item()]], X=[counter],
                         update=None if counter == 0 else 'append', win=train_scenario,
                         opts=dict(legend=['accuracy', 'loss'], title=train_scenario))
            counter += 1
        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_train_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))
        if ls_ <= ls_threshold and ep + 1 >= 30:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break

    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, test_x, test_y, vis, test_scenario='test', n_way=3, shot=3, eval_mode=False):
    net = CNN(nc=n_way).to(device)
    net.load_state_dict(torch.load(save_path))
    if eval_mode:
        net.eval()
    print("Load the model Successfully！\n%s" % save_path)
    loss = nn.CrossEntropyLoss()

    n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert test_x.shape[1] == test_x.shape[1]
    assert n_way == n_class

    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    n_episodes = n_examples // shot
    n_epochs = 1
    counter = 0
    for ep in range(n_epochs):
        avg_acc = 0.
        avg_loss = 0.
        count = 0
        for epi in range(n_episodes):
            x, y = test_x[:, count:count + shot], test_y[:, count:count + shot]
            x, y = x.to(device), y.contiguous().view(-1).to(device)
            count += shot

            outputs = net.forward(x)
            losses = loss(outputs, y)
            ls = losses.cpu().item()
            ac = compute_acc(outputs, y)
            avg_acc += ac
            avg_loss += ls
            print('[test episode {}/{}] => loss:{:.6f}, acc: {:.6f}'.format(epi + 1, n_episodes, ls, ac))
            vis.line(Y=[[ac, ls]], X=[counter],
                     update=None if counter == 0 else 'append', win=test_scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=test_scenario))
            counter += 1
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


def main(save_path, scenario, normalization=True, ls_threshold=0.001,
         n_way=3, samples=20, shot=3, split=10, eval_=False):
    """
    train and test
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
    net = CNN(nc=n_way).to(device)
    vis = visdom.Visdom(env='yancy_env')
    # print(net.cpu())
    # train_x, train_y, test_x, test_y = generator.Cs_data_generator2(examples=samples, split=split, way=n_way,
    #                                                                 normalize=normalization, label=True)
    train_x, train_y, test_x, test_y = generator.SQ_data_generator2(examples=100, split=split, way=n_way,
                                                                    normalize=normalization, label=True)
    # train_x, train_y, test_x, test_y = generator.SQ_data_generator3(examples=samples, split=split, way=n_way,
    #                                                                 normalize=normalization, label=True)
    n_class = train_x.shape[0]
    assert n_class == n_way
    print('train_x Shape:', train_x.shape)
    print('train_y Shape:', train_y.shape)
    print('test_x Shape:', test_x.shape)
    print('test_y Shape:', test_y.shape)
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            print('The training file exists：%s' % save_path)
        else:
            train(net=net, save_path=save_path, train_x=train_x, train_y=train_y, vis=vis,
                  train_scenario=scenario, ls_threshold=ls_threshold, n_way=n_way, shot=shot)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            test(save_path=save_path, test_x=test_x, test_y=test_y, vis=vis,
                 n_way=n_way, shot=shot, eval_mode=eval_)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':
    save_dir = r'G:\comparison_code\comparison_add'
    model_name = 'sq7_CNN_1shot_30_0.pkl'
    path = os.path.join(save_dir, model_name)
    sample = 100  # samples from each file; cw: 50; sq: 100
    way = 7
    n_shot = 1
    num_per_file = 30  # total number of examples per class
    split = num_per_file  # split the data, split=>train, the rest=>test

    print('File exist?', os.path.exists(path))
    main(save_path=path, scenario='CNN', n_way=way, samples=sample, split=split,
         shot=n_shot, ls_threshold=1e-4, eval_=True)
    main(save_path=path, scenario='CNN_', n_way=way, samples=sample, split=split,
         shot=n_shot, ls_threshold=1e-4, eval_=False)
