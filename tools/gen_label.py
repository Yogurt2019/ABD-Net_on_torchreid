import os
from tools.utils import get_path
import pickle


def main():
    dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train/rock_dataset/train'
    dest_pkl = os.path.join(dir, 'label_dict.pkl')
    dir_list = get_path(dir)
    label_dict = {}
    for index in range(len(dir_list)):
        label_dict[dir_list[index]] = int(index)


if __name__ == '__main__':
    main()
