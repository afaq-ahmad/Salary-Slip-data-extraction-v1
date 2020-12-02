
from xml_json_parser import xml_payslip_json

import sys
import os
import json
import pprint

def initialize():
    # Check if path is provided or exists
    
    
    if len(sys.argv) == 1:
        print('No Images path provided to directory or file')
        exit()
    # Check if path is provided or exists
    if len(sys.argv) == 2:
        print('No XML path provided to directory or file')
        exit()

    images_path = sys.argv[1]
    xml_path = sys.argv[2]
    
    if not os.path.exists(images_path):
        print('Specified Images Path does not exist')
        exit()
    if not os.path.exists(xml_path):
        print('Specified XML Path does not exist')
        exit()
    
    # Check if path is a file or directory and extract all filenames 
    if os.path.isfile(xml_path):
        if not (xml_path.endswith('.xml')):
            print('Specified XML File is not in valid format')
            exit()
        
        xml_path=[xml_path]
        
    # Check if path is a file or directory and extract all filenames 
    if os.path.isfile(images_path):
        if not (images_path.endswith('.jpg') or images_path.endswith('.png')):
            print('Specified File is not in valid format')
            exit()

        images_filenames  = [images_path.split('/')[-1]]
        images_path='/'.join(images_path.split('/')[:-1])
        if images_filenames[0]==images_path:
            images_path=''
    
    else:

        images_filenames = os.listdir(images_path)
        images_filenames = [name.replace('_preprocessed','') for name in images_filenames if name.endswith('.jpg') or name.endswith('.png')]
        
    return images_filenames,images_path,xml_path

if __name__ == "__main__":

    # Get input filenames
    images_filenames,images_path,xml_path = initialize()
    print('Number of Images Inputs : ', len(images_filenames))
    # print(images_filenames,images_path)
    
    for image_path in images_filenames:
        # print(image_path)
        uuid_name=image_path.split('/')[-1].split('.')[0]
        # print(uuid_name)
        output=xml_payslip_json(uuid_name,handling_keys='handling_keys.json',extra_identity_name='',image_folder=images_path,xml_folder=xml_path,jsons_folder='jsons')
        pprint.pprint(output, width=1)