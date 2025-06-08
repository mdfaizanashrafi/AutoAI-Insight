import os  #imports the OS module to work with fiile paths

def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()  #splits filename and extension
                                            #[1] gets the extension part only

def is_valid_dataset(file_path):
    valid_exts= ['.csv','.parquet','.json']  
    return get_file_extension(file_path) in valid_exts 

