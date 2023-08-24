import os
import subprocess
from itertools import product
import argparse
import json

tuning_config = {
}

def update_config(tuning_config_path : str) -> None:
    """
    replace old keys with new keys
    """
    keys_to_replace = {
        "CUDA_VISIBLE_DEVICES" : "cuda_device",
        "PORT" : "port"
    }
    with open(tuning_config_path, 'r') as f:
        tuning_config = json.load(f)
    for keys in keys_to_replace:
        if keys in tuning_config:
            tuning_config[keys_to_replace[keys]] = tuning_config[keys]
            del tuning_config[keys]
    with open(tuning_config_path, 'w') as f:
        json.dump(tuning_config, f, indent=4)


def generate_config(**modified_kwargs) -> dict:
    """
    modified_kwargs: dict of key, value pairs to be modified from default_configs
    If value is empty string or None, it will not be modified.
    """
    copied_config = default_configs.copy()
    for key, value in modified_kwargs.items():
        if key not in default_configs:
            raise ValueError(f"key {key} not in default_configs")
        if value == "" or value is None:
            continue
        copied_config[key] = value
    return copied_config

def load_default_config(config_path:str):
    """
    config_path: path to json file containing default configs
    Loads default configs from json file, and returns a dict of configs
    """
    default_configs = {
        'project_name_base' : "BASE", 
        'model_file' :'./model.safetensors',
        'optimizer' : 'AdamW8bit',
        'network_dim' : 16,
        'network_alpha' : 8,
        'conv_dim' : 8,
        'conv_alpha' : 1,
        'num_repeats' : 10,
        'epoch_num' : 10,
        'train_batch_size' : 4,
        'unet_lr' : 1e-4,
        'text_encoder_lr' : 2e-5,
        'target_path' : './train',
        'temp_dir' : './tmp',
        'images_folder' : '',
        'cuda_device' : '0',
        'repo_dir' : '.',
        'port' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'seed' : 42,
        'prompt_path' : './prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
        'clip_skip' : 2,
        'max_grad_norm' : 0,
        'up_lr_weight' : '[1,1,1,1,1,1,1,1,1,1,1,1]',
        'down_lr_weight' : '[1,1,1,1,1,1,1,1,1,1,1,1]',
        'mid_lr_weight' : 1,
        'lbw_weights' : '' # [1,]*17 or [1]* 16, modify this if you want
    }
    try:
        with open(config_path, 'r') as f:
            default_configs_loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file at {}, using default configs".format(config_path))
    for keys in default_configs:
        if keys not in default_configs_loaded:
            default_configs_loaded[keys] = default_configs[keys]
    default_configs = default_configs_loaded
    return default_configs

def convert_relative_path_to_absolute_path(dict_config:dict):
    """
    dict_config: dict of configs
    Converts relative path to absolute path
    """
    for key, value in dict_config.items():
        if key in ['target_path', 'temp_dir', 'images_folder', 'model_file', 'prompt_path']:
            dict_config[key] = os.path.abspath(value)
    return dict_config

def generate_tuning_config(config, **modified_kwargs) -> dict:
    """
    modified_kwargs: dict of key, value pairs to be modified from default_configs
    """
    new_config = config.copy()
    for keys in config.keys():
        # remove _list
        if keys.endswith('_list'):
            del new_config[keys]
    new_config.update(modified_kwargs)
    return new_config

def load_tuning_config(config_path:str):
    """
    config_path: path to json file containing default configs
    Loads default configs from json file, and returns a dict of configs
    """
    tuning_config = {
        # example, you can input as _list for iterating
        #'unet_lr_list' : [1e-5, 1e-4, 2e-4, 3e-4],
        #'text_encoder_lr_list' : [1e-5, 2e-5, 3e-5, 4e-5],
        #'network_alpha_list' : [2,4,8],
        #'network_dim_list' : [16],
        #'clip_skip_list' : [1,2],
        #'num_repeats_list' : [10],
        #'seed_list' : [42],
        'cuda_device' : '0',
        'port' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'prompt_path' : './prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
    }
    update_config(config_path)
    try:
        with open(config_path, 'r') as f:
            tuning_config_loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file, using default configs")
    for keys in tuning_config:
        if keys not in tuning_config_loaded:
            # check if list exists instead, then skip
            tuning_config_loaded[keys] = tuning_config[keys]
    tuning_config = tuning_config_loaded
    return tuning_config

# generate_config('unet_lr' : 1e-5) -> returns new config modified with unet lr

if __name__ == '__main__':

    # check if venv is activated
    # if not, activate venv
    import sys
    abs_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(abs_path)) # execute from here
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name_base', type=str, default='BASE')
    parser.add_argument('--default_config_path', type=str, default='default_config.json')
    parser.add_argument('--tuning_config_path', type=str, default='tuning_config.json')
    # train_id_start
    parser.add_argument('--train_id_start', type=int, default=0) #optional
    # images_folder
    parser.add_argument('--images_folder', type=str, default='') #optional
    parser.add_argument('--model_file', type=str, default='') #optional
    parser.add_argument('--port', type=str, default='') #optional
    parser.add_argument('--cuda_device', type=str, default='') #optional
    parser.add_argument('--debug', action='store_true', default=False) #optional
    # venv path
    parser.add_argument('--venv_path', type=str, default='') #optional

    # python automate-train.py --project_name_base BASE --default_config_path default_config.json --tuning_config_path tuning_config.json 
    # --train_id_start 0 --images_folder '' --model_file '' --port '' --cuda_device ''

    args = parser.parse_args()
    debug = args.debug
    project_name_base = args.project_name_base
    model_name = args.model_file
    images_folder = args.images_folder
    cuda_device = args.cuda_device
    venv_path = args.venv_path
    # handling venv
    if venv_path != '':
        execute_path = os.path.join(venv_path, 'bin', 'python')
    else:
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            execute_path = sys.executable # get path of python executable
        else:
            print("venv not activated, activating venv. This uses relative path, so locate this script in the same folder as venv")
            venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv') # expected venv path
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.exists(venv_path):
                raise ValueError("venv not found at {}".format(venv_path))
            # os-specific venv activation, windows -> Scripts, posix -> bin
            if os.name == 'nt': # windows
                execute_path = os.path.join(venv_path, 'Scripts', 'python.exe')
                #print("call " + os.path.abspath(".\\venv\\Scripts\\activate.bat"), shell=True)
                #subprocess.check_call(["call", os.path.abspath(".\\venv\\Scripts\\activate.bat")], shell=True)
            else: # posix
                execute_path = os.path.join(venv_path, 'bin', 'python')
    print(f"using python executable at {execute_path}")
    train_id = args.train_id_start
    default_configs = load_default_config(args.default_config_path)
    tuning_config = load_tuning_config(args.tuning_config_path)

    # warn if custom_dataset is not None
    if tuning_config['custom_dataset'] is not None:
        ignored_options_name = ['images_folder', 'num_repeats','shuffle_caption', 'keep_tokens', 'resolution']
        print(f"custom_dataset is not None, dataset options {ignored_options_name} will be ignored")

    list_arguments_name = {}
    for arguments, values in tuning_config.items():
        if arguments.endswith('_list'):
            list_arguments_name[arguments.replace('_list', '')] = values
    if "PORT" in tuning_config:
        tuning_config['port'] = tuning_config['PORT']
        del tuning_config['PORT']
    keys_to_remove = {'CUDA_VISIBLE_DEVICES', 'PORT'}
    for args in product(*list_arguments_name.values()):
        list_arguments = dict(zip(list_arguments_name.keys(), args))
        temp_tuning_config = generate_tuning_config(tuning_config, **list_arguments)
        # check validity
        if temp_tuning_config['network_alpha'] > temp_tuning_config['network_dim']:
            continue
        if temp_tuning_config['unet_lr'] < temp_tuning_config['text_encoder_lr']:
            continue
        if temp_tuning_config.get('conv_alpha', 1) > temp_tuning_config.get('conv_dim', 8):
            continue
        # this arguments will be used for overriding default configs
    
        config = generate_config(**temp_tuning_config,
                                )
        # override args
        config['project_name_base'] = project_name_base
        if model_name:
            config['model_file'] = model_name
        if images_folder:
            config['images_folder'] = images_folder
        config['cuda_device'] = temp_tuning_config['cuda_device'] if cuda_device == '' else cuda_device
        for keys in keys_to_remove:
            if keys in config:
                del config[keys]
        print(f"running _{train_id}")
        command_inputs = [execute_path, "trainer.py"]
        for arguments, values in config.items():
            if values is None or values == '':
                continue
            command_inputs.append(f"--{arguments}")
            command_inputs.append(str(values))
        command_inputs.append(f"--custom_suffix")
        command_inputs.append(str(train_id))
        if debug:
            print(' '.join(command_inputs) + '\n')
        else:
            subprocess.check_call(command_inputs)
        train_id += 1
