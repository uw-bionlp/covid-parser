import os


def get_new_filename(fn, dir_in, dir_out, ext):
    
    
    fn_out = os.path.relpath(fn, dir_in)
    fn_out = '{}.{}'.format(os.path.splitext(fn_out)[0], ext)
    fn_out = os.path.join(dir_out, fn_out)

    return fn_out



def save_json(str_, fn, dir_in, dir_out, ext='json'):

    # Save prediction as json
    fn_out = get_new_filename(fn, dir_in, dir_out, ext=ext)
    
    open(fn_out, 'w').write(str_)
    
        