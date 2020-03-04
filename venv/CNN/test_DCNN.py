import torch
import torch.nn as nn
import visdom
import os
from cnn_Data_generator_normalize import data_generate
from train_DCNN import CNN, compute_acc

generator = data_generate()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(save_path, test_scenario, normalization=True, eval_mode=True, ls_threshold=0.001,
         n_epochs=100, n_way=3, n_episodes=30, sample=20, IsSQ=True, shot=3):
    print('%d GPU is available.' % torch.cuda.device_count())
    net = CNN(nc=n_way).to(device)
    net.load_state_dict(torch.load(save_path))
    print('%d GPU is available.' % torch.cuda.device_count())
    vis = visdom.Visdom(env='yancy_env')
    if eval_mode:
        net.eval()
    print("Load the model Successfully！\n%s" % save_path)
    # print(net.cpu())
    # optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    eval_num = shot
    img_w = 2048
    chn = 1
    n_examples = sample

    if IsSQ:
        train_x, train_y = generator.SQ_data_generator(train=False, examples=n_examples,
                                                       normalize=normalization, label=True)
    else:
        train_x, train_y = generator.Cs_data_generator(train=False, examples=n_examples,
                                                       normalize=normalization, label=True)
    # eval_x, eval_y = generator.Cs_data_generator(train=False, examples=eval_num,
    #                                              normalize=normalization, label=True)

    # eval_x, eval_y = torch.from_numpy(eval_x).float(), torch.from_numpy(eval_y).long()
    train_x, train_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    # y: [nc, num]==>[nc*num]
    n_class = train_x.size(0)
    assert n_way == n_class

    print('test_data Shape:', train_x.size())
    print('test_label Shape:', train_y.size())
    print('n_way=>', n_way, 'n_shot/bsize=>', shot)
    print("---------------------Testing----------------------\n")
    avg_acc = 0.
    avg_loss = 0.

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            epi_classes = torch.randperm(n_class)[:n_way]  # training
            x = torch.zeros([n_way, shot, img_w, chn], dtype=torch.float32)
            y = torch.zeros([n_way, shot], dtype=torch.long)

            for i, epi_cls in enumerate(epi_classes):
                selected = torch.randperm(n_examples)[:shot]
                x[i] = train_x[epi_cls, selected[:shot]]
                y[i] = train_y[epi_cls, selected[:shot]]
            x, y = x.to(device), y.contiguous().view(-1).to(device)
            outputs = net.forward(x)
            losses = loss(outputs, y)
            ls = losses.cpu().item()
            ac = compute_acc(outputs, y)

            # ls_ = torch.mean(avg_ls).cpu().item()
            avg_acc += ac
            avg_loss += ls
            vis.line(Y=[[ac, ls]], X=[epi],
                     update=None if epi == 0 else 'append', win=test_scenario,
                     opts=dict(legend=['accuracy', 'loss'], title=test_scenario))

    avg_acc /= n_episodes
    avg_loss /= n_episodes
    # ls_ = torch.mean(avg_ls).cpu().item()
    vis.text('Average Accuracy: {:.6f}  Average Loss:{:.6f}'.format(avg_acc, avg_loss), win='Test result')
    print('------------------------Average Result----------------------------')
    print('Average Test Accuracy: {:.6f}'.format(avg_acc))
    print('Average Test Loss: {:.6f}\n'.format(avg_loss))


if __name__ == '__main__':
    save_dir = r'G:\comparison_code'
    model_name = 'case4_0_DCNN_5shot_adam_5_2.pkl'
    path = os.path.join(save_dir, model_name)
    if os.path.exists(path):
        test(save_path=path, test_scenario='eval-4way', n_epochs=1, n_episodes=100,
             n_way=4, sample=50, IsSQ=False, shot=5, ls_threshold=1e-5, eval_mode=True)
        test(save_path=path, test_scenario='test-4way', n_epochs=1, n_episodes=100,
             n_way=4, sample=50, IsSQ=False, shot=5, ls_threshold=1e-5, eval_mode=False)
    else:
        print('The path does NOT exist! Check it please:%s' % path)
