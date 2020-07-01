from pathlib import Path
import os

# Custom module imports
from extractor import Extractor
from utils.utils import get_new_filename
from models.event_extractor.wrapper import EventExtractorWrapper

'''
Parameters
'''
input_dir = '/data/users/lybarger/clinicalIE/repo/code/docker/test_input/'
output_dir = '/data/users/lybarger/clinicalIE/repo/code/docker/test_output/'
input_ext = 'txt'

model_dir = '/models/covid/'
xfmr_dir = '/pretrained/biobert_pretrain_output_all_notes_150000/'



'''
Extraction
'''
# Load model
model = Extractor(
        model_class = EventExtractorWrapper,
        model_dir = model_dir,
        xfmr_dir = xfmr_dir)
        

# Recursively find files
files = list(Path(input_dir).rglob("*.{}".format(input_ext)))
print("File count found:{}".format(len(files)))

# Iterate over files
for fn in files:
    
    # Load file text
    print("Processing:{}".format(fn))
    with open(fn,'r') as f:
       text = f.read()
       
    # Get predictions   
    json_str = model.predict(text)
    
    # Save prediction as json
    fn_out = get_new_filename(fn, input_dir, output_dir, ext='json')
    with open(fn_out, 'w') as f:
       f.write(json_str)
    
    


    