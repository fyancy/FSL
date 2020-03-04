"""
to train or test ssProto
"""
from semiPro import ssProto
from Data_generator_normalize import data_generate
import torch
import os
import visdom
from my_utils import Euclidean_Distance, vis_tSNE, plot_confusion_matrix, t_sne

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
generator = data_generate()


def train(net, save_path, train_x, un_x, vis, scenario, n_epochs=200, n_way=3, shot=2,
          n_episodes=30, ls_threshold=0.001, skip_lr=0.01, nu=1):
    """

    :param un_x:
    :param scenario:
    :param net:
    :param train_x:
    :param vis:
    :param shot:
    :param skip_lr: 0.001~0.01
    :param nu:
    :param save_path:
    :param n_epochs:
    :param n_way:
    :param n_episodes:
    :param ls_threshold: to decide whether to break
    """
    net.train()
    optimizer1 = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # nesterov=True; lr初始值设为0.1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)  # lr=lr∗gamma^epoch
    optimizer2 = torch.optim.Adam(net.parameters())
    optimizer = optimizer1
    optional_lr = skip_lr  # 经验参数：0.001~0.01

    n_s = n_q = shot
    n_u = nu
    img_w = 2048
    chn = 1
    n_examples = train_x.shape[1]
    n_unlabeled = un_x.shape[1]
    n_class = train_x.shape[0]

    print('train_set:', train_x.shape)
    print('unlabeled_set:', un_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, img_w))
    print('(n_way, n_unlabel, data_len) ==> ', (n_way, n_u, img_w))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, img_w))
    print("---------------------Training----------------------\n")
    avg_ls = torch.zeros([n_episodes]).to(device)
    opt_flag = False
    counter = 0

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            epi_classes = torch.randperm(n_class)[:n_way]  # training
            support = torch.zeros([n_way, n_s, img_w, chn], dtype=torch.float32)
            query = torch.zeros([n_way, n_q, img_w, chn], dtype=torch.float32)
            unlabel = torch.zeros([n_way, n_u, img_w, chn], dtype=torch.float32)

            for i, epi_cls in enumerate(epi_classes):
                selected = torch.randperm(n_examples)[:n_s + n_q]
                support[i] = train_x[epi_cls, selected[:n_s]]
                query[i] = train_x[epi_cls, selected[n_s:n_s + n_q]]
                selected = torch.randperm(n_unlabeled)[:n_u]
                unlabel[i] = un_x[epi_cls, selected[:n_u]]
            support, query, unlabel = support.to(device), query.to(device), unlabel.to(device)
            losses, ls_ac = net.forward(nc=n_way, x_s=support, x_u=unlabel, x_q=query)

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
                print('[epoch {}/{}, episode {}/{}] => loss: {:.8f}, acc: {:.8f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, ls, ac))

        ls_ = torch.mean(avg_ls).cpu().item()
        print('[epoch {}/{}] => avg_loss: {:.8f}\n'.format(ep + 1, n_epochs, ls_))

        if ls_ < optional_lr and opt_flag is False:
            optimizer = optimizer2
            print('Optimizer ==> Adam')
            opt_flag = True
        # elif (ls_ <= ls_threshold and ep + 1 >= 70) or ls_ < 0.5 * ls_threshold:
        elif ls_ <= ls_threshold:
            print('[ep %d] => loss = %.8f < %f' % (ep + 1, ls_, ls_threshold))
            break

        scheduler.step(epoch=ep // 5)
    print('train finished!')
    torch.save(net.state_dict(), save_path)
    print('This model saved at', save_path)


def test(save_path, test_x, vis, scenario='test_semipro', n_way=3, shot=2, iseval=False,
         n_episodes=400, n_epochs=1, nu=1):
    # --------------------修改-------------------------
    net = ssProto(device=device).to(device)
    net.load_state_dict(torch.load(save_path))
    if iseval:
        net.eval()
    print("Load the model Successfully！\n%s" % save_path)
    n_s = n_q = shot
    n_u = nu
    img_w = 2048
    chn = 1

    n_examples = test_x.shape[1]
    n_class = test_x.shape[0]
    assert n_class >= n_way
    assert n_examples >= n_s + n_q + n_u

    print('test_data set Shape:', test_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, img_w))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, img_w))
    print('(n_way, n_unlabel, data_len) ==> ', (n_way, n_u, img_w))
    print("---------------------Testing----------------------\n")
    avg_acc_ = 0.
    avg_loss_ = 0.
    counter = 0

    for ep in range(n_epochs):
        # For a new epoch, these params should be updated again.
        avg_acc = 0.
        avg_loss = 0.
        cm = False
        for epi in range(n_episodes):
            epi_classes = torch.randperm(n_class)[:n_way]  # training
            support = torch.zeros([n_way, n_s, img_w, chn], dtype=torch.float32)
            query = torch.zeros([n_way, n_q, img_w, chn], dtype=torch.float32)
            unlabel = torch.zeros([n_way, n_u, img_w, chn], dtype=torch.float32)

            for i, epi_cls in enumerate(epi_classes):
                selected = torch.randperm(n_examples)[:n_s + n_u + n_q]
                support[i] = test_x[epi_cls, selected[:n_s]]
                unlabel[i] = test_x[epi_cls, selected[n_s:n_s + n_u]]
                query[i] = test_x[epi_cls, selected[n_s + n_u:]]
            if epi == n_episodes - 1:
                cm = True
            support, query, unlabel = support.to(device), query.to(device), unlabel.to(device)
            with torch.no_grad():
                _, ls_ac = net.forward(nc=n_way, x_s=support, x_u=unlabel, x_q=query, cm=cm)

            ls, ac = ls_ac['loss'], ls_ac['acc']
            avg_acc += ac
            avg_loss += ls
            if (epi + 1) % 100 == 0:
                print('[Episode {}/{}] => avg_loss: {:.8f}, avg_acc: {:.8f}'.
                      format(epi + 1, n_episodes, avg_loss / (epi + 1), avg_acc / (epi + 1)))

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
    print('\n------------------------Average Result----------------------------')
    print('Average Test Loss: {:.6f}'.format(avg_loss_))
    print('Average Test Accuracy: {:.6f}\n'.format(avg_acc_))


def main(save_path, scenario, split, n_way=3, shot=2, eval_=False,
         ls_threshold=0.001, nu=1, normalization=True):
    net = ssProto(device=device).to(device)
    print('%d GPU is available.' % torch.cuda.device_count())
    vis = visdom.Visdom(env='yancy_env')
    n_s = n_q = shot
    n_u = nu
    img_w = 2048

    # case 1: sq7
    # train_x, test_x = generator.SQ_data_generator2(examples=samples, split=split, way=n_way,
    #                                                normalize=normalization, label=False)
    # un_x = test_x

    # case 2: multi_speed
    # train_x, test_x = generator.SQ_data_generator3(examples=samples, split=split, way=n_way,
    #                                                normalize=normalization, label=False)
    # un_x = test_x

    # case 3: cw2sq
    train_x, _ = generator.Cs_data_generator2(examples=50, split=split, way=n_way,
                                              normalize=normalization, label=False)
    _, test_x = generator.SQ_data_generator2(examples=100, split=0, way=n_way,
                                             normalize=normalization, label=False)
    un_x = train_x

    # -----------scratch 50 number but only 5 used for training-----------
    # train_x, test_x = generator.Cs_data_generator2(examples=samples, split=split,
    #                                                normalize=normalization, label=False)
    # train_x = train_x[:n_way, :split]

    # ----------------------
    n_class = train_x.shape[0]
    assert n_class == n_way

    print('train_data shape:', train_x.shape)
    print('test_data shape:', test_x.shape)
    print('(n_way, n_support, data_len)==> ', (n_way, n_s, img_w))
    print('(n_way, n_unlabel, data_len) ==> ', (n_way, n_u, img_w))
    print('(n_way, n_query, data_len) ==> ', (n_way, n_q, img_w))
    train_x, test_x = torch.from_numpy(train_x).float(), torch.from_numpy(test_x).float()
    un_x = torch.from_numpy(un_x).float()

    order = input("Train or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            print('The training file exists：%s' % save_path)
        else:
            train(net=net, save_path=save_path, train_x=train_x, un_x=un_x, vis=vis, scenario=scenario,
                  ls_threshold=ls_threshold, skip_lr=0.005, n_way=n_way, shot=shot, nu=nu)

    order = input("Test or not? Y/N\n")
    if order == 'Y' or order == 'y':
        if os.path.exists(save_path):
            test(save_path=save_path, test_x=test_x, vis=vis, n_way=n_way, shot=shot, nu=nu, iseval=eval_)
        else:
            print('The path does NOT exist! Check it please:%s' % save_path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    save_dir = r'G:\comparison_code\comparison_discussion_2'
    # model_name = 'sq7_semipro_5shot_4att_10_1.pkl'  # ls_threshold=0.1-0.01
    # model_name = 'sq3spd_semipro_5shot_0att_16_0.pkl'  # ls_threshold=0.2-0.1; samples: 16//4 = 4
    model_name = 'cw2sq_semipro_5shot_3att_30_1.pkl'  # ls_threshold=1e-4
    path = os.path.join(save_dir, model_name)
    way = 3
    shot = 5
    un = 5
    # samples = 50  # samples from each file; cw: 50; sq: 100
    samples = 100  # samples from each file; cw: 50; sq: 100
    n_train_per_file = 30  # total number of training examples each way
    split = n_train_per_file  # split the data, split=>train, the rest=>test
    # assert shot+shot+un <= split*4
    print('File exist?', os.path.exists(path))

    main(save_path=path, scenario='semipro', n_way=way, nu=un,
         shot=shot, split=split, eval_=True, ls_threshold=1e-4)
    main(save_path=path, scenario='semipro_', n_way=way, nu=un,
         shot=shot, split=split, eval_=False, ls_threshold=1e-4)
    # loss阈值可设1e-4~1e-3;在100个循环以内足以收敛到1e-4以下，一般70以下。
    plt.show()
