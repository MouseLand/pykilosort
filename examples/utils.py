import os
import urllib.request


def create_test_directory(test_dir_path):
    try:
        os.makedirs(test_dir_path)
    except FileExistsError:
        pass

    s3_base = 'https://pykilosort.s3.eu-west-2.amazonaws.com/test-recording/'
    for file_ in ['xc.npy', 'yc.npy', 'test.bin']:
        url = s3_base + file_
        print(url)
        filename = test_dir_path + file_
        urllib.request.urlretrieve(url, filename)
