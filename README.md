# Defense against Adversarial Attacks on Audio DeepFake Detection Baseline Code

The following repository contains code for our paper called "Defense against Adversarial Attacks on Audio DeepFake Detection".

We base our codebase on [Attack Agnostic Dataset repo](https://github.com/piotrkawa/attack-agnostic-dataset).


### Dependencies
Install required dependencies using: 
```bash
pip install -r requirements.txt
```

Other example configs are available under `configs/training/`


### ASVspoof2021_DF_eval
```bash
CUDA_VISIBLE_DEVICES=3 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/ASVspoof/LA/ASVspoof2021_DF_eval/flac --output_path /datab/Dataset/maibui/random_masking_piotrakawa/ASVSpoof21DFeval/rawnet2/MIFGSM_dc10/ --adv_method MIFGSM_dc10
```
```bash
CUDA_VISIBLE_DEVICES=2 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/maibui/asv21df_spoofonly_ratio25/ --output_path /datab/Dataset/maibui/random_masking_piotrakawa/ASVSpoof21DFeval/rawnet2/adv_training_defense/IFGSM --adv_method1 IFGSM --adv_method2 IFGSM_RM_50
```

### Wakefake
```bash
CUDA_VISIBLE_DEVICES=0 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/generated_audio/ --output_path /datab/Dataset/maibui/random_masking_piotrakawa/Wakefake/aasistssl/IFGSM_RM_10 --adv_method IFGSM_RM_10
CUDA_VISIBLE_DEVICES=3 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/generated_audio/ --output_path /datab/Dataset/maibui/random_masking_piotrakawa/Wakefake/aasistssl/MIFGSM_dc10/ --adv_method MIFGSM_dc10
```

### test denoise
```bash
CUDA_VISIBLE_DEVICES=1 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/generated_audio/ --output_path /home/maibui/audio-deepfake-adversarial-attacks/output/test/adv --adv_method FGSM
CUDA_VISIBLE_DEVICES=1 python gen_adversarial.py --batch_size 1 --input_path /home/maibui/audio-deepfake-adversarial-attacks/output/test/adv --output_path /home/maibui/audio-deepfake-adversarial-attacks/output/test/adv_denoise --adv_method FGSM
```

### ASVSpoof5
```bash
CUDA_VISIBLE_DEVICES=0 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/ASVspoof5/flac_T --output_path /datab/Dataset/maibui/asvspoof5/Conformer_trainRFGSM --file_path /home/maibui/AnalysisAudio/protocols/ASVSpoof5/asvspoof5_train182356.txt --adv_method RFGSM
CUDA_VISIBLE_DEVICES=0 python gen_adversarial.py --batch_size 1 --input_path /datab/Dataset/ASVspoof5/flac_D --output_path /datab/Dataset/maibui/asvspoof5/Conformer_dev2 --file_path /home/maibui/AnalysisAudio/protocols/ASVSpoof5/asvspoof5_dev140949.txt --adv_method RFGSM
```

### copy paste augmentation
```bash
CUDA_VISIBLE_DEVICES=2 python copy_paste_generate.py 
```
