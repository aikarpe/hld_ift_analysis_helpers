import os
import re
# ================================================================================
#                                                           collection of experimental
#                                                           folders and images 

def collect_dirs(root):
    """
    f-n collects recursively all subfolders for given root

    :return: a list of path
    :rtype: list
    """
    raw = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    dirs = list(filter(lambda x: os.path.isdir(x), raw))

    if len(dirs) == 0:
        return [root]
    else:
        out = [root]
        for d in dirs:
            out = out + collect_dirs(d)
        return out

def match_conc_at_end(pathes):
    """
    f-n returns pathes that ends with `conc_x.xxxxx`, where x is digit
    """
    matches = list(map(lambda x: re.search('conc_[0-9.]{7}$', x), pathes))
    use = list(filter(lambda x: x[0] is not None, zip(matches, pathes)))
    return list(map(lambda x: x[1], use))

def list_images(path):
    """
    f-n returns all jpgs in given folder
    """
    return list(filter(lambda x: re.search(".jpg$", x) is not None, 
                        list(map(lambda x: os.path.join(path, x), os.listdir(path)))))

def collect_images(root):
    """
    f-n collects recursively all jpg images residing in folders ending
    in `conc_x.xxxxx`
    """
    dirs = match_conc_at_end(collect_dirs(root))
    ims = []
    for d in dirs:
        ims = ims + list_images(d)
    return ims


def collect_data_jsons(root):
    """
    fn collects all data.json files under given root
    """
    dirs = collect_dirs(root)
    data_files = []
    for d in dirs:
        data_files = data_files + list(filter(lambda x: re.search("data.json$", x), 
                                                list(map(lambda x: os.path.join(d, x), os.listdir(d)))))
    return data_files

def collect_files(root, regex):
    """
    fn collects all files matching regex under given root
    """
    dirs = collect_dirs(root)
    data_files = []
    for d in dirs:
        data_files = data_files + list(filter(lambda x: re.search(regex, x), 
                                                list(map(lambda x: os.path.join(d, x), os.listdir(d)))))
    return data_files


