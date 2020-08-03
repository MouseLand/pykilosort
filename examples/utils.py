import os
import urllib.request

S3_BUCKET_BASE_URL = 'https://static.alexmorley.me/pykilosort/test-recording/'


def urlretrieve(url, filename):
    opener = urllib.request.URLopener()
    # Spoof a browser's user agent header because nginx doesn't like urllib
    opener.addheader('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0')
    filename, headers = opener.retrieve(url, filename)

def create_test_directory(test_dir_path):
    try:
        os.makedirs(test_dir_path)
    except FileExistsError:
        pass

    print(f"Downloading test files from {S3_BUCKET_BASE_URL} ...")
    for file_ in ['xc.npy', 'yc.npy', 'test.bin']:
        url = S3_BUCKET_BASE_URL + file_
        filename = test_dir_path + file_
        urlretrieve(url, filename)
    print("Done")
