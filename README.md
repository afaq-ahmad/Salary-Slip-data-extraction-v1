# Salary-Slip-data-extraction-v1



## Dictionary Keys update:
Call this function from xml_json_parser.py file.

	def dictionary_update(new_dictionary,handling_keys_path='handling_keys.json')

Input:
 
    new_dictionary: new dictionary consists of new keys or new variable of old keys of dictionary.
    handling_keys_path: (default:handling_keys.json) path of handling keys file.

>Save the updated dictionary as handling_keys.json

## Xml data Extraction:

#### Call this function from xml_json_parser.py file.

	def xml_payslip_json(uuid_name,handling_keys='handling_keys.json',extra_identity_name='',image_folder='preprocessed_images',xml_folder='ocr_output',jsons_folder='jsons')
    Parameters:
     uuid_name (str): specific encrypted uuid_name used to save image,xml.
     Data Read paths:
     image_path=image_folder+'/'+uuid_name+extra_identity_name+'.jpg'
     xml_path=xml_folder+'/'+uuid_name+extra_identity_name+'.xml'
     outputjson_path=jsons_folder+'/'+uuid_name+extra_identity_name+'.json'
    
     Returns:
     dictionary:Returning dictionary of datafound in xml based on keypoints.

#### Testing on single image or directory

  * running on a directory : `python test_code_xml_to_json.py path_to_images_dir xml_dir`
  * running on a file: `python test_code_xml_to_json.py path_to_image xml_file`
  
    This will output the results in a folder named `jsons/file_name`
