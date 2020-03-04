import numpy as np
from my_utils import my_normalization1, my_normalization2, my_normalization3
from SQdata_dir import train5_dir19, train5_dir39
# 3 classes
from SQdata_dir import test3_dir109, test3_dir119, test3_dir129, test3_dir139
from SQdata_dir import test3_dir209, test3_dir219, test3_dir229, test3_dir239
from SQdata_dir import test3_dir309, test3_dir319, test3_dir329, test3_dir339
from SQdata_dir import test3_dir09, test3_dir19, test3_dir29, test3_dir39  # outer1/2/3
from SQdata_dir import sq3_09, sq3_19, sq3_29, sq3_39
# 4 classes
from SQdata_dir import test4_dir09, test4_dir19, test4_dir29, test4_dir39
# 5 classes
from SQdata_dir import test5_dir09, test5_dir19, test5_dir29, test5_dir39
# 6 classes
from SQdata_dir import test6_dir09, test6_dir19, test6_dir29, test6_dir39
# 7 classes
from SQdata_dir import test7_dir09, test7_dir19, test7_dir29, test7_dir39
from SQdata_dir import sq3_39_0, sq3_39_1, sq7_39_0, sq7_39_1
# case_data
from Csdata_dir import case_3way_0, case_3way_1, case_3way_2, case_3way_3
from Csdata_dir import case_4way_0, case_4way_1, case_4way_2, case_4way_3


def data_label_shuffle(data, label):
    """
    要求input是二维数组array
    :param data: [num, data_len]
    :param label: [num]
    :return:
    """
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


def sample_shuffle(data):
    """
    required: data.shape [Nc, num, data_len...]
    :param data: [[Nc, num, data_len...]]
    """
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data


def SQ_data_get(file_dir, num):
    """
    :param file_dir:
    :param num: number of data extracted
    :return: list=> (examples*data_len, ) 即 (num, )
    """
    mat = []
    line_start = 20  # SQ_data要从txt文件第16行以后取数据
    line_end = line_start + num
    with open(file_dir) as file:  # 默认是r-read
        for line in file.readlines()[line_start:line_end]:
            line_cur = line.strip('\n').split('\t')[1]
            line_float = float(line_cur)  # 使用float函数直接把字符类data转化成为float
            mat.append(line_float)
    mat = np.array(mat)
    return mat


def Cs_data_get(file_dir, num):
    """

    :param file_dir:
    :param num:
    :return: [examples*data_len, ] 即 (num, )
    """
    mat = []
    line_start = 0  # case_data从txt文件第0行取数据即可
    line_end = line_start + num
    with open(file_dir) as file:  # 默认是r-read
        for line in file.readlines()[line_start:line_end]:
            line_cur = line.strip('\n')
            line_float = float(line_cur)  # 使用float函数直接把字符类data转化成为float
            mat.append(line_float)
    mat = np.array(mat)
    return mat


class data_generate():
    def __init__(self):
        # IsSQ=True: SQ
        self.train_dir = train5_dir39
        self.test_dir = test4_dir19
        self.sq3 = [sq3_39_0, sq3_39_1]
        self.sq7 = [sq7_39_0, sq7_39_1]
        self.sq3_speed = [sq3_09, sq3_19, sq3_29, sq3_39]

        # IsSQ=False: CASE, CWRU_data
        self.case_train_dir = case_4way_0
        self.case_test_dir = case_4way_1
        self.case4 = [case_4way_0, case_4way_1, case_4way_2, case_4way_3]

    def SQ_data_generator(self, train=True, examples=150, data_len=2048, normalize=True, label=False):
        """
        :param label:
        :param train: train or test
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,20,2048,1], [Nc, 20]
        """
        file_dir = self.train_dir if train is True else self.test_dir
        print('SQ_DATA loading ……')
        n_file = len(file_dir)
        data_size = examples * data_len
        data_set = None
        for i in range(n_file):
            data = SQ_data_get(file_dir=file_dir[i], num=data_size)
            data = np.reshape(data, [-1, data_len])  # [examples, 2048]
            if normalize:
                data = my_normalization1(data)
            if i == 0:
                data_set = data
            else:
                data_set = np.concatenate((data_set, data), axis=0)

        data_set = data_set.reshape([n_file, examples, data_len, 1])

        if label:
            label = np.arange(n_file)[:, np.newaxis]
            label = np.tile(label, (1, examples))  # [Nc, examples]
            return data_set, label  # [Nc,50,2048,1], [Nc, 50]
        else:
            return data_set  # [nc, num, 2048, 1]

    def Cs_data_generator(self, train=True, examples=50, data_len=2048, normalize=True, label=False):
        file_dir = self.case_train_dir if train is True else self.case_test_dir
        print('Case_DATA loading ……')
        n_file = len(file_dir)
        data_size = examples * data_len
        data_set = None
        for i in range(n_file):
            data = Cs_data_get(file_dir=file_dir[i], num=data_size)
            data = np.reshape(data, [-1, data_len])  # [examples, 2048]
            if normalize:
                data = my_normalization1(data)
            if i == 0:
                data_set = data
            else:
                data_set = np.concatenate((data_set, data), axis=0)

        data_set = data_set.reshape([n_file, examples, data_len, 1])

        if label:
            label = np.arange(n_file)[:, np.newaxis]
            label = np.tile(label, (1, examples))  # [Nc, examples]
            return data_set, label  # [Nc,50,2048,1], [Nc, 50]
        else:
            return data_set  # [nc, num, 2048, 1]

    def Cs_data_generator2(self, way=3, examples=50, split=30,
                           data_len=2048, normalize=True, label=False):
        file_dir = self.case4
        print('Case_DATA2 loading ……')
        n_way = len(file_dir[0])  # 4way
        n_file = len(file_dir)  # 4 files
        data_size = examples * data_len
        num_each_way = examples * n_file  # 200
        # print(data_.shape)
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, examples, data_len])
            for j in range(n_file):
                data = Cs_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = my_normalization1(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])[:way]
        data_set = sample_shuffle(data_set)
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def SQ_data_generator2(self, way=3, examples=100, split=30,
                           data_len=2048, normalize=True, label=False):
        """
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,20,2048,1], [Nc, 20]
        """
        file_dir = self.sq3
        if way == 3:
            file_dir = self.sq3
        elif way == 7:
            file_dir = self.sq7
        print('SQ_DATA2 loading ……')
        n_way = len(file_dir[0])  # 3/7 way
        # assert n_way == way
        n_file = len(file_dir)  # 2 files
        num_each_file = examples
        num_each_way = examples * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = SQ_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = my_normalization1(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, data_len, 1])
        data_set = sample_shuffle(data_set)
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,2048,1], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]

    def SQ_data_generator3(self, way=3, examples=100, split=3,
                           data_len=2048, normalize=True, label=False):
        """
        :param split:
        :param way: 3/7
        :param label:
        :param examples: examples of each class
        :param data_len: size of each example
        :param normalize: normalize data
        :return: [Nc,20,2048,1], [Nc, 20]
        """
        file_dir = self.sq3_speed
        print('SQ_DATA3_speed loading ……')
        n_way = len(file_dir[0])  # 3 way
        assert n_way == way
        n_file = len(file_dir)  # 4
        num_each_file = examples  # 100
        num_each_way = examples * n_file  # 400
        data_size = num_each_file * data_len  # 100*2048
        train_data = np.zeros([n_way, n_file, split, data_len])
        test_data = np.zeros([n_way, n_file, num_each_file-split, data_len])
        for i in range(n_way):
            for j in range(n_file):
                data = SQ_data_get(file_dir=file_dir[j][i], num=data_size)
                data = data.reshape([-1, data_len])
                if normalize:
                    data = my_normalization1(data)
                train_data[i, j] = data[:split]
                test_data[i, j] = data[split:]
        train_data = train_data.reshape([n_way, -1, data_len, 1])
        test_data = test_data.reshape([n_way, -1, data_len, 1])
        train_data = sample_shuffle(train_data)  # 第二维shuffle: 将4种转速shuffle

        if label:
            label = np.arange(n_way)[:, np.newaxis]
            label = np.tile(label, (1, num_each_way))  # [Nc, examples]
            train_lab, test_lab = label[:, :split*n_file], label[:, split*n_file:]
            return train_data, train_lab, test_data, test_lab  # [Nc, num_each_way, 2048,1], [Nc, num_each_way]
        else:
            return train_data, test_data  # [Nc, num_each_way, 2048, 1]


if __name__ == '__main__':
    d = data_generate()
    split = 2
    da1, lb1, da2, lb2 = d.SQ_data_generator3(label=True, way=3, normalize=False, examples=100, split=split)
    # da1, lb1, da2, lb2 = d.SQ_data_generator2(label=True, examples=100, normalize=False, way=7, split=5)
    print(da2.shape)
    print(lb2.shape)
    print(type(da1))
    print(da1[2, 2*split, :10])
    print(lb1[2, 2*split:2*split+2])

    # a = np.array([[[1, 2], [3, 0]], [[3, 5], [6, 0]], [[5, 7], [9, 0]]])
    # # b = a[:2]
    # # b = sample_shuffle(b)
    # np.random.shuffle(a)
    # print(a)
    # print(b)
