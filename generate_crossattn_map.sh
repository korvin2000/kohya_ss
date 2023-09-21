python generate_crossattn_map.py \
      --network_module networks.lora \
      --ckpt /data7/sooyeon/LyCORIS/LyCORIS/pretrained/animefull-final-pruned-fp16.safetensors \
      --network_weights ./result/haibara_full_block/haibara-000034.safetensors \
      --seed 42