
import os
import subprocess
from itertools import product
import argparse
import json

tuning_config = {
    'unet_lr_list' : [1e-5, 1e-4, 2e-4, 3e-4],
    'text_encoder_lr_list' : [1e-5, 2e-5, 3e-5, 4e-5],
    'network_alpha_list' : [2,4,8],
    'CUDA_VISIBLE_DEVICES' : '0'
}

def generate_config(**modified_kwargs):
    copied_config = default_configs.copy()
    for key, value in modified_kwargs.items():
        if key not in default_configs:
            raise ValueError(f"key {key} not in default_configs")
        if value == "" or value == None:
            continue
        copied_config[key] = value
    return copied_config

def load_default_config(config_path:str):
    default_configs = {
        'project_name_base' : "BASE", # this will be used for creating folders with configs
        'model_file' :'/model.safetensors',
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
        'target_path' : '',
        'temp_dir' : 'Loras/tmp',
        'images_folder' : '',
        'cuda_device' : '0',
        'repo_dir' : 'kohya_ss'
    }
    try:
        with open(config_path, 'r') as f:
            default_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file, using default configs")
    return default_configs

def load_tuning_config(config_path:str):
    tuning_config = {
        'unet_lr_list' : [1e-5, 1e-4, 2e-4, 3e-4],
        'text_encoder_lr_list' : [1e-5, 2e-5, 3e-5, 4e-5],
        'network_alpha_list' : [2,4,8],
        'CUDA_VISIBLE_DEVICES' : '0',
    }
    try:
        with open(config_path, 'r') as f:
            tuning_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file, using default configs")
    return tuning_config

# generate_config('unet_lr' : 1e-5) -> returns new config modified with unet lr

if __name__ == '__main__':
    #print(getsourcefile(lambda:0))
    abs_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(abs_path)) # execute from here
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name_base', type=str, default='BASE')
    parser.add_argument('--default_config_path', type=str, default='default_config.json')
    parser.add_argument('--tuning_config_path', type=str, default='tuning_config.json')
    # train_id_start
    parser.add_argument('--train_id_start', type=int, default=0)
    # images_folder
    parser.add_argument('--images_folder', type=str, default='')
    parser.add_argument('--model_file', type=str, default='')
    # model_file
    args = parser.parse_args()
    project_name_base = args.project_name_base
    train_id = args.train_id_start
    default_configs = load_default_config(args.default_config_path)
    tuning_config = load_tuning_config(args.tuning_config_path)


    unet_lr_list = tuning_config['unet_lr_list'] #[1e-5, 1e-4, 2e-4, 3e-4] #1e-4
    text_encoder_lr_list = tuning_config['text_encoder_lr_list'] #[1e-5, 2e-5, 3e-5, 4e-5] #2e-5
    network_alpha_list = tuning_config['network_alpha_list'] #[2,4,8] #8
    for unet_lr, text_encoder_lr, network_alpha in product(unet_lr_list, text_encoder_lr_list, network_alpha_list):
        config = generate_config(project_name_base=project_name_base,unet_lr=unet_lr, 
                                text_encoder_lr=text_encoder_lr, network_alpha=network_alpha,
                                images_folder=args.images_folder if args.images_folder else "",
                                model_file=args.model_file if args.model_file else "",
                                cuda_device=tuning_config['CUDA_VISIBLE_DEVICES'])
        #print(config)
        print(f"running _{train_id}")
        command_inputs = ["python", "trainer.py"]
        for arguments, values in config.items():
            command_inputs.append(f"--{arguments}")
            command_inputs.append(str(values))
        command_inputs.append(f"--custom_suffix")
        command_inputs.append(str(train_id))
        subprocess.check_call(command_inputs)
        train_id += 1
