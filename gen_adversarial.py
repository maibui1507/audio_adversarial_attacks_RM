# import required module
import os
from multiprocessing import Pool, set_start_method
import argparse
from tqdm import *
from functools import partial
import logging
import librosa
import soundfile as sf
from random import randrange
import torch
from pydub import AudioSegment
import random
import csv
import pandas as pd
import numpy as np
import threading
from src.models.adversarial import AdversarialNoiseAugmentor

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Set up logging
# logging.basicConfig(filename='running.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')


def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--input_path', type=str, default="",required=True, help='Audio file path')
    
    parser.add_argument('--output_path', type=str, default="",required=True, help='Feature output path')

    parser.add_argument('--adv_method1', type=str, default="",required=True, help='Adversarial attack method 1')
    parser.add_argument('--adv_method2', type=str, default="",required=True, help='Adversarial attack method 2')

    # parser.add_argument('--file_path', type=str, default="",required=True, help='Protocol to get filename')
    
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

    config = {
            "aug_type": "adversarial",
            "output_path": args.output_path,
            "out_format": args.out_format,
            "batch_size": args.batch_size,

            "model_name": "rawnet2",
            "model_pretrained": "/home/maibui/audio-deepfake-adversarial-attacks/pretrained/pre_trained_DF_RawNet2.pth",
            "config_path": "/home/maibui/audio-deepfake-adversarial-attacks/pretrained/Rawnet2_config.yaml",

            # "model_name": "aasistssl",
            # "model_pretrained": "/home/maibui/audio-deepfake-adversarial-attacks/pretrained/Best_LA_model_for_DF.pth",
            # "ssl_model": "/home/maibui/audio_augmentor/pretrained/xlsr2_300m.pth",

            # "model_name": "conformer",
            # "model_pretrained": "/AISRC1/hungdx/Rawformer-implementation-anti-spoofing/pretrained/pretrained/conformer_best.pth",

            "device": "cuda",

            "adv_method1": args.adv_method1,
            "adv_method2": args.adv_method2,
        }

    # file_path = '/home/maibui/AnalysisAudio/protocols/ASVSpoof21_DF_eval/asvspoof21_5000.txt'
    # df = pd.read_csv(file_path, sep=",", header=None)
    # filenames = [os.path.join(args.input_path, f"{filename}.flac") for filename in df[1]] 

    # file_path = '/home/maibui/AnalysisAudio/protocols/WaveFake/wavefake_5000.txt'
    # df = pd.read_csv(file_path, sep=",", header=None)
    # # filenames = [os.path.join(args.input_path, filename) for filename in df[0]] 
    # filenames = [os.path.join(args.input_path, str(filename.split("/")[-1].split(".")[0]) + "_IFGSM_RM_50.flac") for filename in df[0]] 

    # file_path = args.file_path
    # df = pd.read_csv(file_path, sep=",", header=None)
    # filenames = [os.path.join(args.input_path, f"{filename}.flac") for filename in df[1]] 

    # filenames = sorted(filenames)
    # filenames = os.listdir(args.input_path)
    # filenames = [os.path.join(args.input_path, filename) for filename in filenames]
    
    filenames = os.listdir(args.input_path)
    filenames = [os.path.join(args.input_path, filename) for filename in filenames]
 
    ana = AdversarialNoiseAugmentor(config)
    ana.load_batch(filenames)
    ana.transform_batch()


if __name__ == '__main__':
    main()

