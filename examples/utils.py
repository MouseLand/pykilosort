import os
import urllib.request

S3_BUCKET_BASE_URL = 'https://pykilosort.s3.eu-west-2.amazonaws.com/test-recording/'

def create_test_directory(test_dir_path):
    try:
        os.makedirs(test_dir_path)
    except FileExistsError:
        pass

    print(f"Downloading test files from {S3_BUCKET_BASE_URL} ..."
    for file_ in ['xc.npy', 'yc.npy', 'test.bin']:
        url = S3_BUCKET_BASE_URL + file_
        filename = test_dir_path + file_
        urllib.request.urlretrieve(url, filename)
    print("Done")
