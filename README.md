## Set Up
mamba create -n anidiffusion python=3.10 -y
conda activate anidiffusion
pip install -r requirements.txt
python tools/download_weights.py
git lfs install
git clone https://huggingface.co/lambdalabs/sd-image-variations-diffusers

Now you should have a folder pretrained_weights with: DWPose, sd-image-variations-diffusers, sd-vae-ft-mse, image_encoder, stable-diffusion-v1-5 subfolders and motion_module.pth, denoising_unet.pth, pose_guider.pth, reference_unet.pth files.

Download the pretrained model from https://drive.google.com/file/d/1_QhnyOlQbhNF6W3Q8D4LmVK5klsHZZFg/view?usp=sharing
Unzip it, and you should have a folder of path results/rndLayerOrder_512/stage1/checkpoint-30000 that contains optimizer.bin, pytorch_model.bin, random_states_0.pkl, scaler.pt, scheduler.bin

## Finetune
Finally, you can finetune the pretrained model on our example data dataset/edina 
python train_stage_1_diverseTopo.py --config configs/sample_finetune/stage1_diverseTopo_rndLayerOrder_json_edina.yaml --mode train --res_512

## Inference
python train_stage_1_diverseTopo.py --config configs/sample_finetune/stage1_diverseTopo_rndLayerOrder_json_edina.yaml --mode val --res_512