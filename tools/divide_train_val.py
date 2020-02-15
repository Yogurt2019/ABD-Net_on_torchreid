# Used to divide one-hot dataset into train and val folders
# Each class one folder, divide as the ratio
import os
import shutil


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    dir = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data/Market1501_onehot'
    ratio = 0.9
    dirs = os.listdir(dir)
    for item in dirs:
        files = os.listdir(os.path.join(dir, item))
        for mode in ('train', 'val'):
            t_dir = os.path.join(dir.replace('Market1501_onehot', 'Market1501_train'), mode, item)
            makedir(t_dir)
            divide_point = int(len(files) * ratio)
            if mode == 'train':
                for file in files[:divide_point]:
                    print('src:', os.path.join(dir, item, file))
                    print('dst:', t_dir)
                    shutil.copy(src=os.path.join(dir, item, file), dst=t_dir)
                    print()
            else:
                for file in files[divide_point:]:
                    shutil.copy(src=os.path.join(dir, item, file), dst=t_dir)
                    print()


if __name__ == '__main__':
    main()
