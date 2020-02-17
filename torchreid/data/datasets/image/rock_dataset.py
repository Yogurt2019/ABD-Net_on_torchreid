from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
from tools.utils import get_path, txt2list
import torchreid
from torchreid.data import ImageDataset
import pickle


class RockDataSet(ImageDataset):
    dataset_dir = 'rock_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        dest_pkl = os.path.join(self.root, self.dataset_dir, 'label_dict.pkl')
        dir_list = get_path(os.path.join(self.root, self.dataset_dir, 'train'))
        label_dict = {}
        for index in range(len(dir_list)):
            label_dict[dir_list[index].split('/')[-1]] = int(index)
        f = open(dest_pkl, 'wb')
        pickle.dump(label_dict, f)
        train, query, gallery = [], [], []
        train_dir = os.path.join(self.dataset_dir, 'train')
        gallery_dir = os.path.join(self.dataset_dir, 'gallery')
        query_dir = os.path.join(self.dataset_dir, 'query')
        for mode in (train_dir, gallery_dir, query_dir):
            for rock_dir in os.listdir(mode):
                pid = int(label_dict[rock_dir])
                if mode != query_dir:
                    camid = 0
                else:
                    camid = 1
                per_rock_dir = os.listdir(os.path.join(mode, rock_dir))
                for per_rock_img in per_rock_dir:
                    img_path = os.path.join(mode, rock_dir, per_rock_img)
                    tuple = (img_path, pid, camid)
                    if mode == train_dir:
                        train.append(tuple)
                    elif mode == gallery_dir:
                        gallery.append(tuple)
                    else:
                        query.append(tuple)
        # TODO:query dataset
        super(RockDataSet, self).__init__(train, query, gallery, **kwargs)
