import os
import subprocess
from itertools import product
import argparse
import json

tuning_config = {
}

def generate_config(**modified_kwargs):
    """
    modified_kwargs: dict of key, value pairs to be modified from default_configs
    If value is empty string or None, it will not be modified.
    """
    copied_config = default_configs.copy()
    for key, value in modified_kwargs.items():
        if key not in default_configs:
            raise ValueError(f"key {key} not in default_configs")
        if value == "" or value == None:
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
        'model_file' :'model.safetensors',
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
        'target_path' : '/train',
        'temp_dir' : '/tmp',
        'images_folder' : '',
        'cuda_device' : '0',
        'repo_dir' : '.',
        'port' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'seed' : 42,
        'prompt_path' : '/prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
        'clip_skip' : 2
    }
    try:
        with open(config_path, 'r') as f:
            default_configs_loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file, using default configs")
    for keys in default_configs:
        if keys not in default_configs_loaded:
            default_configs_loaded[keys] = default_configs[keys]
    default_configs = default_configs_loaded
    return default_configs

def load_tuning_config(config_path:str):
    """
    config_path: path to json file containing default configs
    Loads default configs from json file, and returns a dict of configs
    """
    tuning_config = {
        'unet_lr_list' : [1e-5, 1e-4, 2e-4, 3e-4],
        'text_encoder_lr_list' : [1e-5, 2e-5, 3e-5, 4e-5],
        'network_alpha_list' : [2,4,8],
        'network_dim_list' : [16],
        'CUDA_VISIBLE_DEVICES' : '0',
        'PORT' : 20060,
        'sample_opt' : 'epoch',
        'sample_num' : 1,
        'seed_list' : [42],
        'prompt_path' : '/prompt/prompt.txt',
        'keep_tokens' : 0,
        'resolution' : 768,
        'lr_scheduler' : 'cosine_with_restarts',
        'lora_type' : 'LoRA',
        'custom_dataset' : None,
        'clip_skip_list' : [1,2]
    }
    try:
        with open(config_path, 'r') as f:
            tuning_config_loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Couldn't load config file, using default configs")
    for keys in tuning_config:
        if keys not in tuning_config_loaded:
            tuning_config_loaded[keys] = tuning_config[keys]
    tuning_config = tuning_config_loaded
    return tuning_config

# generate_config('unet_lr' : 1e-5) -> returns new config modified with unet lr

if __name__ == '__main__':
    import sys
    execute_path = sys.executable # get path of python executable
    #print(getsourcefile(lambda:0))
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

    # python automate-train.py --project_name_base BASE --default_config_path default_config.json --tuning_config_path tuning_config.json 
    # --train_id_start 0 --images_folder '' --model_file '' --port '' --cuda_device ''

    args = parser.parse_args()
    # model_file
    project_name_base = args.project_name_base
    train_id = args.train_id_start
    default_configs = load_default_config(args.default_config_path)
    tuning_config = load_tuning_config(args.tuning_config_path)


    unet_lr_list = tuning_config['unet_lr_list'] #[1e-5, 1e-4, 2e-4, 3e-4] #1e-4
    text_encoder_lr_list = tuning_config['text_encoder_lr_list'] #[1e-5, 2e-5, 3e-5, 4e-5] #2e-5
    network_alpha_list = tuning_config['network_alpha_list'] #[2,4,8] #8
    network_dim_list = tuning_config['network_dim_list'] #[16] #16
    seed_list = tuning_config['seed_list'] if 'seed_list' in tuning_config else [42] #[42]
    clip_skip = tuning_config['clip_skip_list'] if 'clip_skip_list' in tuning_config else [2] #[2]
    if "PORT" in tuning_config:
        tuning_config['port'] = tuning_config['PORT']

    for unet_lr, text_encoder_lr, network_alpha, network_dim, seed, clip_skip in product(unet_lr_list, text_encoder_lr_list, network_alpha_list, network_dim_list, seed_list):
        config = generate_config(project_name_base=project_name_base,unet_lr=unet_lr, 
                                text_encoder_lr=text_encoder_lr, network_alpha=network_alpha,
                                images_folder=args.images_folder if args.images_folder else "",
                                model_file=args.model_file if args.model_file else "",
                                cuda_device=tuning_config['CUDA_VISIBLE_DEVICES'] if args.cuda_device == '' else args.cuda_device,
                                port=tuning_config['port'] if args.port == '' else args.port,
                                sample_opt=tuning_config['sample_opt'],
                                sample_num=tuning_config['sample_num'],
                                seed=seed,
                                prompt_path=tuning_config['prompt_path'],
                                keep_tokens=tuning_config['keep_tokens'],
                                resolution=tuning_config['resolution'],
                                lr_scheduler=tuning_config['lr_scheduler'],
                                lora_type=tuning_config['lora_type'],
                                custom_dataset=tuning_config['custom_dataset'],
                                network_dim = network_dim,
                                clip_skip = clip_skip,
                                )
        #print(config)
        print(f"running _{train_id}")
        command_inputs = [execute_path, "trainer.py"]
        for arguments, values in config.items():
            command_inputs.append(f"--{arguments}")
            command_inputs.append(str(values))
        command_inputs.append(f"--custom_suffix")
        command_inputs.append(str(train_id))
        subprocess.check_call(command_inputs)
        train_id += 1
