# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
import requests

class YCBDownloader:
    """
    A class to handle downloading and extracting YCB dataset objects.
    """
    
    def __init__(self, output_directory, objects_to_download, files_to_download, extract):
        """
        Initialize the YCBDownloader.

        Args:
            output_directory (str): The directory where files will be downloaded.
            objects_to_download (str or list): 'all' or a list of object names to download.
            files_to_download (list): List of file types/categories to download.
            extract (bool): Whether to extract the downloaded archives.
        """
        self.output_directory = output_directory
        self.objects_to_download = objects_to_download
        self.files_to_download = files_to_download
        self.extract = extract
        self.base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
        self.objects_url = self.base_url + "objects.json"

    def fetch_objects(self, url):
        """
        Fetch the list of available objects from the given URL.
        
        Args:
            url (str): The URL to the objects.json file.

        Returns:
            list: A list of object names.
        """
        response = requests.get(url)
        objects = response.json()
        return objects["objects"]

    def download_file(self, url, filename):
        """
        Download a file from a URL to a local path.

        Args:
            url (str): The URL of the file to download.
            filename (str): The local path where the file should be saved.
        """
        response = requests.get(url, stream=True)
        file_size = int(response.headers.get("Content-Length", 0))
        print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))

        file_size_dl = 0
        with open(filename, 'wb') as f:
            for buffer in response.iter_content(chunk_size=65536):
                if buffer:
                    file_size_dl += len(buffer)
                    f.write(buffer)
                    status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
                    status = status + chr(8)*(len(status)+1)
                    print(status, end=' ')

    def tgz_url(self, object_name, type_name):
        """
        Generate the URL for a specific object and file type.

        Args:
            object_name (str): The name of the YCB object.
            type_name (str): The type of file to download (e.g., 'berkeley_rgbd').

        Returns:
            str: The full URL to the .tgz file.
        """
        if type_name in ["berkeley_rgbd", "berkeley_rgb_highres"]:
            return self.base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object_name, type=type_name)
        elif type_name in ["berkeley_processed"]:
            return self.base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object_name, type=type_name)
        else:
            return self.base_url + "google/{object}_{type}.tgz".format(object=object_name, type=type_name)

    def extract_tgz(self, filename, dir):
        """
        Extract a .tgz file to a specified directory and delete the archive.

        Args:
            filename (str): The path to the .tgz file.
            dir (str): The directory to extract to.
        """
        tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename, dir=dir)
        os.system(tar_command)
        os.remove(filename)

    def check_url(self, url):
        """
        Check if a URL is reachable.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is reachable, False otherwise.
        """
        try:
            response = requests.head(url)
            return response.status_code == 200
        except Exception as e:
            return False