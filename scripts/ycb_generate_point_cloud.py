# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
from vision_utils import YCBPointCloudGenerator

if __name__ == "__main__":
    # Parameters
    # ycb_data_folder = "../data/"			# Folder that contains the ycb data.	
    # target_object = "002_master_chef_can"	# Full name of the target object.
    # viewpoint_camera = "NP3"				# Camera which the viewpoint will be generated.
    # viewpoint_angle = "15"					# Relative angle of the object w.r.t the camera (angle of the turntable).
    
    parser = argparse.ArgumentParser(description='Generate Point Cloud from YCB Data')
    parser.add_argument('--ycb_data_folder', default="../data/", type=str,
                        help='Folder that contains the ycb data')
    parser.add_argument('--target_object', default="002_master_chef_can", type=str,
                        help='Full name of the target object')
    parser.add_argument('--viewpoint_camera', default="NP3", type=str,
                        help='Camera which the viewpoint will be generated')
    parser.add_argument('--viewpoint_angle', default="15", type=str,
                        help='Relative angle of the object w.r.t the camera')
    parser.add_argument('--reference_camera', default="NP5", type=str,
                        help='Reference camera')

    args = parser.parse_args()

    generator = YCBPointCloudGenerator(
        ycb_data_folder=args.ycb_data_folder,
        target_object=args.target_object,
        viewpoint_camera=args.viewpoint_camera,
        viewpoint_angle=args.viewpoint_angle
    )

    generator.generate_point_cloud(referenceCamera=args.reference_camera)
