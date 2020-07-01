import os


def get_new_filename(fn, dir_in, dir_out, ext='json'):
    
    
    fn_out = os.path.relpath(fn, dir_in)
    fn_out = os.path.splitext(fn_out)[0] + '.json'
    fn_out = os.path.join(dir_out, fn_out)

    return fn_out