import os
import shutil
from tqdm import tqdm


def query_one_dict(src, dst):
    files = os.listdir(src)
    for file in tqdm(files):
        person_id = file.split('_')[0]
        if file.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp']:
            if int(person_id) in range(1, 1502):
                if not os.path.exists(os.path.join(dst, person_id)):
                    os.mkdir(os.path.join(dst, person_id))
                shutil.copy(os.path.join(src, file), os.path.join(dst, person_id))


def main():
    src = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data/Market1501'
    dst = '/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data/Market1501_onehot'
    for folder in ['bounding_box_test', 'bounding_box_train']:
        target = os.path.join(src, folder)
        query_one_dict(target, dst)


if __name__ == '__main__':
    main()
