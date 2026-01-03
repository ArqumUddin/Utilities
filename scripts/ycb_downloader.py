# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse
from vision_utils import YCBDownloader

def comma_separated_list(value):
    """Convert a comma-separated string to a list."""
    return [s.strip() for s in value.split(',')]

if __name__ == "__main__":
    # You can either set this to "all" or a list of the objects that you'd like to
    # download.
    # objects_to_download = "all"
    # objects_to_download = ["002_master_chef_can", "003_cracker_box"]
    
    # You can edit this list to only download certain kinds of files.
    # 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
    # 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
    # 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
    # 'google_16k' contains google meshes with 16k vertices.
    # 'google_64k' contains google meshes with 64k vertices.
    # 'google_512k' contains google meshes with 512k vertices.
    # See the website for more details.
    # files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]

    # Extract all files from the downloaded .tgz, and remove .tgz files.
    # If false, will just download all .tgz files to output_directory
    
    parser = argparse.ArgumentParser(description='Download YCB Object Set')
    parser.add_argument('--output_directory', default="./ycb", type=str,
                        help='Directory to download the files to')
    parser.add_argument('--objects_to_download', default="all", type=str,
                        help='Objects to download')
    parser.add_argument('--files_to_download', default="berkeley_rgb_highres,berkeley_processed", type=comma_separated_list,
                        help='Files to download')
    parser.add_argument('--extract', default=True, type=bool,
                        help='Extract the downloaded files')
    args = parser.parse_args()

    # Create the downloader instance
    downloader = YCBDownloader(
        output_directory=args.output_directory,
        objects_to_download=args.objects_to_download,
        files_to_download=args.files_to_download,
        extract=args.extract
    )

    if not os.path.exists(downloader.output_directory):
        os.makedirs(downloader.output_directory)
    
    # Determine objects to download
    if downloader.objects_to_download == "all":
        objects = downloader.fetch_objects(downloader.objects_url)
    elif ',' in downloader.objects_to_download:
        objects = comma_separated_list(downloader.objects_to_download)
    else:
        objects = [downloader.objects_to_download]

    # Iterate over all objects to download
    for object_name in objects:
        # Iterate over all file types to download for the current object
        # Construct the URL for the specific object and file type
        # Check if the URL exists before attempting download
        # Define the local filename for the downloaded file
        # Download the file
        # Extract the file if requested

        for file_type in downloader.files_to_download:
            url = downloader.tgz_url(object_name, file_type)
            if not downloader.check_url(url):
                continue

            filename = "{path}/{object}_{file_type}.tgz".format(path=downloader.output_directory,
                                                                object=object_name,
                                                                file_type=file_type)
            downloader.download_file(url, filename)
            if downloader.extract:
                downloader.extract_tgz(filename, downloader.output_directory)
