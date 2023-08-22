import os
import re
import toml
import subprocess
import argparse
import shutil
import json
import socket
from typing import Tuple

def get_available_port(port: int, max_retries=100) -> int:
  tried_ports = []
  while port < 65535:
    try:
      sock = socket.socket()
      sock.bind(("127.0.0.1", port))
      sock.close()
      return port
    except (OSError, socket.error):
      print(f"Port {port} is already in use.")
      tried_ports.append(port)
      port += 1
    if len(tried_ports) > max_retries:
      break
  raise Exception("No available ports found!")


def validate_dataset():
  if override_dataset_config_file:
    print("Using custom dataset config file ", override_dataset_config_file)
    custom_dataset = override_dataset_config_file
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
  supported_types = (".png", ".jpg", ".jpeg", 'jfif')

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    print("💥 Error: Please choose a valid project name.")
    return

  if custom_dataset:
    try:
      print("📄 Using custom dataset config file "+custom_dataset)
      assert os.path.exists(custom_dataset), "Custom dataset config file not found at "+custom_dataset
      datconf = toml.load(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except Exception as e:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
      print(e)
      raise e
      return
    reg = [d for d in datasets if d.get("is_reg", False)]
    for r in reg:
      print("📁"+r["image_dir"].replace("/content/drive/", "") + " (Regularization)")
    datasets = [d for d in datasets if d not in reg]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
      return
  for f in files:
    if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      return

  if not [txt for txt in files if txt.lower().endswith(".txt")]:
    caption_extension = ""
  if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
    print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
    return

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder.replace("/content/drive/", ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
  if max_train_epochs:
    print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
  else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    return

  if adjust_tags:
    print(f"\n📎 Weighted tags: {'ON' if weighted_captions else 'OFF'}")
    if weighted_captions:
      print(f"📎 Will use {keep_tokens_weight} weight on {keep_tokens} activation tag(s)")
    print("📎 Adjusting tags...")
    adjust_weighted_tags(folders, keep_tokens, keep_tokens_weight, weighted_captions)

  return True

def adjust_weighted_tags(folders, keep_tokens: int, keep_tokens_weight: float, weighted_captions: bool):
  weighted_tag = re.compile(r"\((.+?):[.\d]+\)(,|$)")
  for folder in folders:
    for txt in [f for f in os.listdir(folder) if f.lower().endswith(".txt")]:
      with open(os.path.join(folder, txt), 'r') as f:
        content = f.read()
      # reset previous changes
      content = content.replace('\\', '')
      content = weighted_tag.sub(r'\1\2', content)
      if weighted_captions:
        # re-apply changes
        content = content.replace(r'(', r'\(').replace(r')', r'\)').replace(r':', r'\:')
        if keep_tokens_weight > 1:
          tags = [s.strip() for s in content.split(",")]
          for i in range(min(keep_tokens, len(tags))):
            tags[i] = f'({tags[i]}:{keep_tokens_weight})'
          content = ", ".join(tags)
      with open(os.path.join(folder, txt), 'w') as f:
        f.write(content)

def create_config():
  global dataset_config_file, config_file, model_file, port_num, port_fallback
  from accelerate.utils import write_basic_config
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  if os.path.exists(accelerate_config_file):
    # remove old config file
    os.remove(accelerate_config_file)
  write_basic_config(save_location=accelerate_config_file)
  # load config file and change settings
  if port_fallback:
    # search available port
    new_port_num = get_available_port(port_num)
    if new_port_num != port_num:
      print(f"\n⚠️ Port {port_num} is not available. Using port {new_port_num} instead.")
  else:
    new_port_num = port_num
  with open(accelerate_config_file, 'r') as configfile:
    config_json = json.load(configfile)
    config_json["main_process_port"] = new_port_num
  with open(accelerate_config_file, 'w') as configfile:
    json.dump(config_json, configfile, indent=2)
  if resume:
    resume_points = [f.path for f in os.scandir(output_folder) if f.is_dir()]
    resume_points.sort()
    last_resume_point = resume_points[-1] if resume_points else None
  else:
    last_resume_point = None

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "additional_network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": network_args,
        "network_train_unet_only": True if text_encoder_lr == 0 else None,
        "network_weights": continue_from_lora if continue_from_lora else None
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args if optimizer_args else None,
      },
      "training_arguments": {
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "train_batch_size": train_batch_size,
        "noise_offset": None,
        "clip_skip": 2,
        "min_snr_gamma": min_snr_gamma_value,
        "weighted_captions": weighted_captions,
        "seed": training_seed,
        "max_token_length": 225,
        "xformers": True,
        "lowram": False,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "save_precision": "fp16",
        "mixed_precision": "fp16",
        "output_dir": output_folder,
        "logging_dir": log_folder,
        "output_name": project_name,
        "log_prefix": project_name,
        "save_state": save_state,
        "save_last_n_epochs_state": 1 if save_state else None,
        "resume": last_resume_point,
        "sample_every_n_steps": sample_num if sample_num and sample_opt.lower() == 'steps' else None,
        "sample_every_n_epochs": sample_num if sample_num and sample_opt.lower() == 'epoch' else None,
        "sample_prompts": prompt_path if prompt_path and sample_opt.lower() != 'None' else None
      },
      "model_arguments": {
        "pretrained_model_name_or_path": model_file,
        "v2": custom_model_is_based_on_sd2,
        "v_parameterization": True if custom_model_is_based_on_sd2 else None,
      },
      "saving_arguments": {
        "save_model_as": "safetensors",
      },
      "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
      },
      "dataset_arguments": {
        "cache_latents": True,
      },
      
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
        "flip_aug": flip_aug,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
        "min_bucket_reso": 320 if resolution > 640 else 256,
        "max_bucket_reso": 1280 if resolution > 640 else 1024,
      },
      "datasets": toml.load(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")

def main():
  for dir in (main_dir, repo_dir, log_folder, images_folder, output_folder, config_folder):
    if not dir:
      continue
    os.makedirs(dir, exist_ok=True)
  if not validate_dataset():
    return
  assert os.path.exists(model_file), "Model file not found at "+model_file
  create_config()
  assert os.path.exists(accelerate_config_file), "Config file not found at "+accelerate_config_file
  print("\n⭐ Starting trainer...\n")
  os.chdir(repo_dir)
  subprocess.check_call(["accelerate", "launch", "--config_file="+accelerate_config_file, "--num_cpu_threads_per_process=1", "train_network.py", "--dataset_config="+dataset_config_file, "--config_file="+config_file])
  # move model to output folder

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='LoRA Trainer')

  parser.add_argument('--cuda_device', type=str, default='4',
                      help='CUDA device number (default: 4)')
  parser.add_argument('--project_name_base', type=str, default='WEBTOON_ALL',
                      help='Base name for the project (default: WEBTOON_ALL)')
  parser.add_argument('--model_file', type=str, default='./model.ckpt',
                      help='Path to the model file (default: ./model.ckpt)')
  parser.add_argument('--optimizer', type=str, default='AdamW8bit',
                      help='Optimizer to use (default: AdamW8bit)')
  parser.add_argument('--network_dim', type=int, default=16,
                      help='Dimension of the network (default: 16)')
  parser.add_argument('--network_alpha', type=int, default=8,
                      help='Alpha value for the network (default: 8)')
  parser.add_argument('--conv_dim', type=int, default=8,
                      help='Dimension of the convolutional layers (default: 8)')
  parser.add_argument('--conv_alpha', type=int, default=1,
                      help='Alpha value for the convolutional layers (default: 1)')
  parser.add_argument('--num_repeats', type=int, default=10,
                      help='Number of repeats (default: 10)')
  parser.add_argument('--epoch_num', type=int, default=10,
                      help='Number of epochs (default: 40)')
  parser.add_argument('--train_batch_size', type=int, default=4,
                      help='Batch size for training (default: 4)')
  parser.add_argument('--unet_lr', type=float, default=1e-4,
                      help='Learning rate for the UNet (default: 1e-4)')
  parser.add_argument('--text_encoder_lr', type=float, default=2e-5,
                      help='Learning rate for the text encoder (default: 2e-5)')
  parser.add_argument('--custom_dataset', type=str, default=None,
                      help='Custom dataset config path. (default: None)')
  # add image_folder
  parser.add_argument('--images_folder', type=str, default='')
  #add repo_dir
  parser.add_argument('--repo_dir', type=str, default='.')
  # add custon suffix
  parser.add_argument('--custom_suffix', type=str, default='',
                      help='Custom suffix for the project name (default: "")')
  parser.add_argument('--target_path', type=str, default='./models/Lora',
                      help='Target path for the project (default: ./models/Lora)')
  parser.add_argument('--temp_dir', type=str, default='', help='Temporary directory for the project (default: "")')
  # add port to use for accelerate
  parser.add_argument('--port', type=int, default=20060, help='Port to use for accelerate (default: 20060)')
  # should we use port fallback
  parser.add_argument('--port_fallback', type=bool, default=True, help='Use port fallback (default: False)')
  # prompt
  parser.add_argument('--prompt_path', type=str, default='', help='Prompt file for the project (default: ""), ex : `character is haibara ai,1girl --s 16 --w 512 --h 768 --d 42`')
  # sample_opt ('epoch', 'step')
  parser.add_argument('--sample_opt', type=str, default='epoch', help='Sample option for the project (default: epoch, can be None)')
  # sample_num
  parser.add_argument('--sample_num', type=int, default=1, help='Sample number for the project (default: 1)')
  # seed
  parser.add_argument('--seed', type=int, default=42, help='Seed for the project (default: 42)')
  # keep tokens
  parser.add_argument('--keep_tokens', type=int, default=1, help='Keep tokens for the project (default: 1)')
  # resolution
  parser.add_argument('--resolution', type=int, default=512, help='Resolution for the project (default: 512)')
  # lr_scheduler
  parser.add_argument('--lr_scheduler', type=str, default='cosine_with_restarts', help='LR scheduler for the project (default: cosine_with_restarts)')
  # lora type
  parser.add_argument('--lora_type', type=str, default='LoRA', help='LoRA type for the project (default: LoRA)')
  # clip skip
  parser.add_argument('--clip_skip', type=int, default=2, help='Clip skip for the project (default: 2)')


  args = parser.parse_args()
  # TODO : separate validation
  assert args.sample_opt in ['epoch', 'step', 'None'], "Sample option must be 'epoch' or 'step' or 'None', but given "+args.sample_opt
  assert args.sample_opt == 'None' or args.sample_num > 0, "Sample number must be positive, but given "+str(args.sample_num)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
  project_name_base = args.project_name_base
  model_file = args.model_file
  optimizer = args.optimizer
  network_dim = args.network_dim
  network_alpha = args.network_alpha
  conv_dim = args.conv_dim
  training_seed = args.seed
  conv_alpha = args.conv_alpha
  num_repeats = args.num_repeats
  epoch_num = args.epoch_num
  sample_num = args.sample_num
  sample_opt = args.sample_opt
  prompt_path = args.prompt_path
  assert sample_opt == 'None' or not prompt_path or os.path.exists(prompt_path), "Prompt file not found at "+prompt_path
  train_batch_size = args.train_batch_size
  unet_lr = args.unet_lr
  text_encoder_lr = args.text_encoder_lr
  suffix = args.custom_suffix
  curdir = os.path.dirname(os.path.abspath(__file__)) 
  root_dir = args.temp_dir if args.temp_dir != '' else "./Loras"
  os.chdir(curdir) # change directory to current directory
  repo_dir = args.repo_dir
  port_num = args.port
  port_fallback = args.port_fallback
  images_folder = f"./Loras/{project_name_base}/dataset/" if args.images_folder == '' else args.images_folder
  if args.custom_dataset:
    print("Custom dataset will be loaded from " + args.custom_dataset + " , images_folder will be ignored.")
    images_folder = ''
  project_name = f"{project_name_base}_autotrain" #@param {type:"string"}
  skip_model_test = True
  custom_dataset = None
  override_dataset_config_file = args.custom_dataset 
  assert not override_dataset_config_file or os.path.exists(override_dataset_config_file), "Custom dataset config file not found at "+override_dataset_config_file
  assert args.lr_scheduler in ['constant', 'cosine', 'cosine_with_restarts', 'constant_with_warmup', 'linear', 'polynomial'], "LR scheduler must be 'constant', 'cosine', 'cosine_with_restarts', 'constant_with_warmup', 'linear', or 'polynomial', but given "+args.lr_scheduler
  override_config_file = None
  optimizer_args = None
  continue_from_lora = ""
  weighted_captions = True ## True로 하면 Weighted Caption 적용
  adjust_tags = True
  keep_tokens_weight = 1.0 
  custom_model_is_based_on_sd2 = False #@param {type:"boolean"}
  resolution = args.resolution
  flip_aug = False #@param {type:"boolean"}
  caption_extension = ".txt" #param {type:"string"}
  shuffle_tags = True #@param {type:"boolean"}
  shuffle_caption = shuffle_tags
  keep_tokens = args.keep_tokens #param {type:"slider", min:0, max:10, step:1}
  preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
  max_train_epochs = epoch_num if preferred_unit == "Epochs" else None
  max_train_steps = epoch_num if preferred_unit == "Steps" else None
  save_every_n_epochs = 1 #@param {type:"number"}
  keep_only_last_n_epochs = False
  if not save_every_n_epochs:
    save_every_n_epochs = max_train_epochs
  if not keep_only_last_n_epochs:
    keep_only_last_n_epochs = max_train_epochs

  lr_scheduler = args.lr_scheduler #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
  lr_scheduler_number = 3 #@param {type:"number"}
  lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
  lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0
  lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.5, step:0.01}
  lr_warmup_steps = 0
  min_snr_gamma = True #@param {type:"boolean"}
  min_snr_gamma_value = 5.0 if min_snr_gamma else None
  lora_type = args.lora_type #@param ["LoRA", "LoCon Lycoris", "LoHa Lycoris"]
  conv_compression = False #@param {type:"boolean"}
  network_module = "lycoris.kohya" if "Lycoris" in lora_type else "networks.lora"
  network_args = None if lora_type == "LoRA" else [
    f"conv_dim={conv_dim}",
    f"conv_alpha={conv_alpha}",
  ]
  if "Lycoris" in lora_type:
    network_args.append(f"algo={'loha' if 'LoHa' in lora_type else 'lora'}")
    network_args.append(f"disable_conv_cp={str(not conv_compression)}")
  save_state = False #param {type:"boolean"}
  resume = False #param {type:"boolean"}


  if optimizer == "DAdaptation":
    optimizer_args = ["decouple=True","weight_decay=0.02","betas=[0.9,0.99]"]
    unet_lr = 0.5
    text_encoder_lr = 0.5
    lr_scheduler = "constant_with_warmup"
    network_alpha = network_dim

  #root_dir = os.path.abspath("./Loras")
  #deps_dir = os.path.join(root_dir, "deps")
  
  main_dir      = root_dir
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name + suffix)
  output_folder = os.path.join(main_dir, project_name + suffix, "output")

  config_file = os.path.join(config_folder, "training_config.toml")
  dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
  accelerate_config_file = os.path.join(config_folder, "accelerate_config/config.yaml")

  main()
  #after training, from output_folder, move model file to target_path
  target_path = args.target_path
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  #   # move {project_name}.safetensors to target_path
  #   # rename to {project_name} _suffix.safetensors
    os.rename(os.path.join(output_folder, f"{project_name}.safetensors"), os.path.join(output_folder, f"{project_name}_{suffix}.safetensors"))
    shutil.copy(os.path.join(output_folder, f"{project_name}_{suffix}.safetensors"), target_path)
    # copy config files with suffix
    shutil.copy(config_file, os.path.join(target_path, f"training_config_{suffix}.toml"))
    #shutil.copy(dataset_config_file, os.path.join(target_path, f"dataset_config_{suffix}.toml"))
    #shutil.copy(accelerate_config_file, os.path.join(target_path, f"accelerate_config_{suffix}.yaml"))

  print("Done!")
