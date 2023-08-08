import os
import re
import toml
import subprocess
import argparse
import shutil

# modified from colab py


def validate_dataset():
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
  supported_types = (".png", ".jpg", ".jpeg")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    return

  if custom_dataset:
    try:
      datconf = toml.loads(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
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
  global dataset_config_file, config_file, model_file
  from accelerate.utils import write_basic_config
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  if not os.path.exists(accelerate_config_file):
    write_basic_config(save_location=accelerate_config_file)
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
        "seed": 42,
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
        "resume": last_resume_point
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
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
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
  parser.add_argument('--model_file', type=str, default='/data3/space/space-cowork-sdwebui/sdwebui/stable-diffusion-webui/models/Stable-diffusion/novelai-all.ckpt',
                      help='Path to the model file (default: /data3/space/space-cowork-sdwebui/sdwebui/stable-diffusion-webui/models/Stable-diffusion/novelai-all.ckpt)')
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
  parser.add_argument('--epoch_num', type=int, default=40,
                      help='Number of epochs (default: 40)')
  parser.add_argument('--train_batch_size', type=int, default=4,
                      help='Batch size for training (default: 4)')
  parser.add_argument('--unet_lr', type=float, default=1e-4,
                      help='Learning rate for the UNet (default: 1e-4)')
  parser.add_argument('--text_encoder_lr', type=float, default=2e-5,
                      help='Learning rate for the text encoder (default: 2e-5)')
  # add image_folder
  parser.add_argument('--images_folder', type=str, default='')
  #add repo_dir
  parser.add_argument('--repo_dir', type=str, default='/data3/space/kohya_ss/')
  # add custon suffix
  parser.add_argument('--custom_suffix', type=str, default='',
                      help='Custom suffix for the project name (default: "")')
  # copy target path = /data3/space/research-sdwebui/stable-diffusion-webui/models/Lora
  parser.add_argument('--target_path', type=str, default='/data3/space/research-sdwebui/stable-diffusion-webui/models/Lora',
                      help='Target path for the project (default: /data3/space/research-sdwebui/stable-diffusion-webui/models/Lora)')
  parser.add_argument('--temp_dir', type=str, default='', help='Temporary directory for the project (default: "")')
  args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
  project_name_base = args.project_name_base
  model_file = args.model_file
  optimizer = args.optimizer
  network_dim = args.network_dim
  network_alpha = args.network_alpha
  conv_dim = args.conv_dim
  conv_alpha = args.conv_alpha
  num_repeats = args.num_repeats
  epoch_num = args.epoch_num
  train_batch_size = args.train_batch_size
  unet_lr = args.unet_lr
  text_encoder_lr = args.text_encoder_lr
  suffix = args.custom_suffix
  curdir = os.path.dirname(os.path.abspath(__file__)) 
  root_dir = args.temp_dir if args.temp_dir != '' else "./Loras"
  os.chdir(curdir) # change directory to current directory
  repo_dir = args.repo_dir
  images_folder = f"/data3/space/kohya_ss/colab_ipynb/Loras/{project_name_base}/dataset/" if args.images_folder == '' else args.images_folder
  project_name = f"{project_name_base}_autotrain" #@param {type:"string"}
  skip_model_test = True
  custom_dataset = None
  override_dataset_config_file = None
  override_config_file = None
  optimizer_args = None
  continue_from_lora = ""
  weighted_captions = True ## True로 하면 Weighted Caption 적용
  adjust_tags = True
  keep_tokens_weight = 1.0
  custom_model_is_based_on_sd2 = False #@param {type:"boolean"}
  resolution = 512 #@param {type:"slider", min:512, max:1024, step:128}
  flip_aug = False #@param {type:"boolean"}
  caption_extension = ".txt" #param {type:"string"}
  shuffle_tags = True #@param {type:"boolean"}
  shuffle_caption = shuffle_tags
  activation_tags = "1" #@param [0,1,2,3]
  keep_tokens = int(activation_tags)
  preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
  max_train_epochs = epoch_num if preferred_unit == "Epochs" else None
  max_train_steps = epoch_num if preferred_unit == "Steps" else None
  save_every_n_epochs = 1 #@param {type:"number"}
  keep_only_last_n_epochs = False
  if not save_every_n_epochs:
    save_every_n_epochs = max_train_epochs
  if not keep_only_last_n_epochs:
    keep_only_last_n_epochs = max_train_epochs

  lr_scheduler = "cosine_with_restarts" #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
  lr_scheduler_number = 3 #@param {type:"number"}
  lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
  lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0
  lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.5, step:0.01}
  lr_warmup_steps = 0
  min_snr_gamma = True #@param {type:"boolean"}
  min_snr_gamma_value = 5.0 if min_snr_gamma else None
  lora_type = "LoRA" #@param ["LoRA", "LoCon Lycoris", "LoHa Lycoris"]
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