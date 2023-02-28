import sys
from zipfile import ZipFile

def unzip(zip_file):
    """
    Unzips zip file.
    """
    with ZipFile(zip_file, 'r') as f:
        f.extractall() # extract files from zip file

if __name__ == '__main__':
    zipfile_path = sys.argv[1]
    unzip(zipfile_path)