from argparse import ArgumentParser
from os.path import basename
import torch
import torchaudio
import time

from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
import sys
sys.path.append('..')
import os, sys
import threading
from utils.audio_helpers import s16le_audio_bytes_to_tensor, tf_resample_audio
import collections
import socket
import yaml, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from web.queue import PCMQueue, ProcPCMQueue, ThreadSafeQueue
import numpy as np
# everything_deterministic()
# torch.manual_seed(0)


class VAPWrapper:

    VAP_NOMINAL_SAMPLE_RATE = 16000  # Nominal sample rate expected by the VAP model

    def __init__(self, 
                model_path: str,
                context_size: int = 2.0, #Context size in seconds
                step_size: int = 0.1, #Step size in seconds
                frame_hz: int = 50,
                device: str = 'cuda:0',
                debug_time: bool = False):

        '''
        Load the VAP model
        '''
        self.device = device

        state_dict_path = os.path.join(model_path)
        print(f"Loading VAP model from state dict {state_dict_path}")

        self.vap_conf = VapConfig(
            sample_rate=VAPWrapper.VAP_NOMINAL_SAMPLE_RATE,
            frame_hz=frame_hz,
        )

        self.model = VapGPT(self.vap_conf)
        sd = torch.load(state_dict_path)
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()


        self.debug_time = debug_time


        '''
        Audio buffer requirements
        '''
        ##These are in seconds
        self.context_size = context_size
        self.step_size = step_size

        ##These are in samples and bytes
        self.context_buffer_sample_cnt = int(self.context_size * VAPWrapper.VAP_NOMINAL_SAMPLE_RATE)
        self.context_buffer_byte_cnt = self.context_buffer_sample_cnt * 2

        self.step_trigger_sample_cnt = int(self.step_size * VAPWrapper.VAP_NOMINAL_SAMPLE_RATE)
        self.triggering_step_buffer_byte_cnt = int(self.step_trigger_sample_cnt * 2)


        '''
        Audio input configs
        '''
        self.context_size = context_size
        self.step_size = step_size  

    def trigger_one_processing_step(self, spkA_tensor_to_commit, spkB_tensor_to_commit):
        ## This call is stateless 

        ## Commit the two parties audio bytes to the VAP model for processing
        if(self.debug_time):
            t1 = time.time()

        # print(f'Triggering VAP model with {len(spkA_tensor_to_commit)} samples of human audio and {len(spkB_tensor_to_commit)} samples of robot audio')
        _, _, res = self.__invoke_vap_model(spkA_tensor_to_commit, spkB_tensor_to_commit)

        ## Interpret the result
        res = batch_to_device(res, "cpu")
        full_probs = res["probs"][0, -1].cpu()  # Shape: [256]
        vad_prob_now = res["vad"][0, -1, 0].cpu()
        next_speaker_prob_now = res["p_now"][0, -1, 0].cpu()
        next_speaker_prob_future = res["p_future"][0, -1, 0].cpu()
        if(self.debug_time):
            t2 = time.time()

            # print(f'VAP model returned in {t2-t1:1.3f}. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')
        else:

            # print(f'VAP model returned. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')

            pass

        vap_result = {
            'vad_prob_now': vad_prob_now,
            'next_speaker_prob_now': next_speaker_prob_now,
            'next_speaker_prob_future': next_speaker_prob_future,
            'full_probs': full_probs,
        }

        return vap_result

    def __invoke_vap_model(self, spk_A_waveform: torch.Tensor, spk_B_waveform: torch.Tensor):

        combined_stereo_waveform = torch.stack((spk_A_waveform, spk_B_waveform), dim=0)

        combined_stereo_waveform = combined_stereo_waveform.unsqueeze(0).to(self.device)

        res = self.model.probs(combined_stereo_waveform)

        return spk_A_waveform, spk_B_waveform, res

