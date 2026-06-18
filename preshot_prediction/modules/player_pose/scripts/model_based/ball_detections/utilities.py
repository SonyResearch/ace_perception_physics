import os

def list_files(path, ext):
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(ext):
                file_list.append(os.path.join(root, file))

    return file_list

def expand_files(file_list):
    expanded_files = []
    if file_list is None:
        return expanded_files
    for f in file_list:
        if os.path.isfile(f):
            expanded_files.append(f)
        elif os.path.isdir(f):
            expanded_files.extend(list_files(f, ".h5"))
    return expanded_files
