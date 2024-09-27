from copy_paste import CopyPasteAugmentor
from copy_paste_freq import CopyPasteFrequencyAugmentor
import os
import numpy as np
import librosa  
import argparse
from tqdm import tqdm

def generate_samples(input_path, output_path, config):
    """
    This method generates samples using the CopyPasteAugmentor class.
    
    :param folder_path: str, path to the folder containing audio files
    :param config: dict, configuration dictionary
    """
        
    # Get all audio files in the folder
    audio_files = [f for f in os.listdir(input_path) if f.endswith('.flac')]
    
    # Initialize the CopyPasteAugmentor object
    if config["aug_type"] == "CopyPasteFrequencyAugmentor":
        augmentor = CopyPasteFrequencyAugmentor(config)
    if config["aug_type"] == "CopyPasteAugmentor":
        augmentor = CopyPasteAugmentor(config)
    
    # Generate samples for each audio file
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(input_path, audio_file)
        augmentor.load(audio_path)
        augmentor.transform()
        augmented_audio = augmentor.augmented_audio
        # Save the augmented audio
        augmented_file = f"cpa_{audio_file}"
        augmented_path = os.path.join(output_path, augmented_file)
        augmented_audio.export(augmented_path, format="flac")
        # print(f"Augmented audio saved at: {augmented_path}")
        
def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--input_path', type=str, default="/datab/Dataset/maibui/random_masking_piotrakawa/Wakefake/aasistssl/DIFGSM", help='Audio file path')
    
    parser.add_argument('--output_path', type=str, default="/datab/Dataset/maibui/random_masking_piotrakawa/Wakefake/aasistssl_augmented/DIFGSM_ratio1_freq", help='Augmented output path')
    
    parser.add_argument('--shuffle_ratio', type=float, default=0.99, help='shuffle ratio')
    
    parser.add_argument('--frame_size', type=int, default=1025, help='shuffle ratio')
    
    parser.add_argument('--aug_type', type=str, default="CopyPasteAugmentor", help='augmentation type')
    
    parser.add_argument('--out_format', type=str, default="flac", required=False, help='Output format. \n'
                        +'Suported: flac, ogg, mp3, wav. Default: flac. \n'
                        +'Encode by pydub + ffmpeg. Please install ffmpeg first. \n')
    
    # load argument
    args = parser.parse_args()
        
    return args
   
def main():
    args = parse_argument()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        
    generate_samples(args.input_path, args.output_path, {
        "shuffle_ratio": args.shuffle_ratio,
        "frame_size": args.frame_size,
        "aug_type": args.aug_type,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "out_format": args.out_format
    })
        
if __name__ == '__main__':
    main()
    
    