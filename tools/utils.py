import os
# 获取最后一级文件夹的路径
def get_path(root):
    """
    :param root: 根目录的路径
    :return: 最后一级文件夹的所有路径
    """
    path_list = []

    def _get_path(root_path):
        if os.path.isfile(root_path) or not os.listdir(root_path):
            if os.path.isfile(root_path):
                path_list.append(os.path.dirname(root_path))
            else:
                path_list.append(root_path)
        else:
            for dir in os.listdir(root_path):
                _get_path(os.path.join(root_path, dir))

    _get_path(root)
    return list(set(path_list))


def txt2list(file_list):
    with open(file_list) as flist:
        full_lines = [line.strip() for line in flist]
    return full_lines
