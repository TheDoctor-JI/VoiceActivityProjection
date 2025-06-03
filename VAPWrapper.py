from argparse import ArgumentParser
from os.path import basename
import matplotlib.pyplot as plt
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
from logger.logger import setup_logger
import os, sys
import threading
from utils.audio_helpers import s16le_audio_bytes_to_tensor, tf_resample_audio
import collections
import socket

# everything_deterministic()
# torch.manual_seed(0)


class VAPWrapper:

    def __init__(self, 
                model_path: str,
                context_size: int = 2.0, #Context size in seconds
                step_size: int = 0.1, #Step size in seconds
                frame_hz: int = 50,
                audio_format: str = 's16le',
                vap_nominal_sample_rate: int = 16000,#Has to be the sample rate accepted by the VAP model, which is 16000
                speaker_A_raw_sampling_rate: int = 16000, #Raw sampling rate of the speaker A audio input
                speaker_B_raw_sampling_rate: int = 16000, #Raw sampling rate of the speaker B audio input
                device: str = 'cuda:0',
                debug_time: bool = False,
                parent_logger = None, 
                report_handle = None):

        self.parent_logger = parent_logger

        '''
        Load the VAP model
        '''
        self.device = device
        if parent_logger is None:
            self.logger = setup_logger('VAPWrapper')
        else:
            self.logger = parent_logger.getChild('VAPWrapper')

        state_dict_path = os.path.join(model_path)
        self.logger.info(f"Loading model from state dict {state_dict_path}")
        self.vap_nominal_sample_rate = vap_nominal_sample_rate
        self.vap_conf = VapConfig(
            sample_rate=self.vap_nominal_sample_rate,
            frame_hz=frame_hz,
        )
        self.model = VapGPT(self.vap_conf)
        sd = torch.load(state_dict_path)
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.debug_time = debug_time

        '''
        Audio input configs
        '''
        self.audio_format = audio_format
        if(self.audio_format != 's16le'):
            raise ValueError(f"Unsupported sample format string {self.audio_format}")
        self.context_size = context_size
        self.step_size = step_size  
        self.speaker_A_raw_sampling_rate = speaker_A_raw_sampling_rate
        self.speaker_B_raw_sampling_rate = speaker_B_raw_sampling_rate

        '''
        Audio input buffer
        '''
        self.buffer_lock = threading.Lock()

        self.context_buffer_sample_cnt = int(self.context_size * self.vap_nominal_sample_rate)
        self.context_buffer_byte_cnt = self.context_buffer_sample_cnt * 2

        self.step_trigger_sample_cnt = int(self.step_size * self.vap_nominal_sample_rate)
        self.triggering_step_buffer_byte_cnt = int(self.step_trigger_sample_cnt * 2)

        self.clear_buffers()

        '''
        Miscellaneous
        '''
        self.report_handle = report_handle

    def clear_buffers(self):

        self.speaker_A_context_buffer = torch.zeros(self.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_A_step_buffer = torch.tensor([], dtype=torch.float32)

        self.speaker_B_context_buffer = torch.zeros(self.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_B_step_buffer = torch.tensor([], dtype=torch.float32)

    def recv_audio_chunk(self, aud_chunk: bytes, is_spk_A: bool):
        self.buffer_lock.acquire(blocking=True)

        if is_spk_A:
            # Convert bytes to tensor and resample
            resampled_chunk = tf_resample_audio(
                waveform_tensor=s16le_audio_bytes_to_tensor(aud_chunk), 
                original_sample_rate=self.speaker_A_raw_sampling_rate, 
                target_sample_rate=self.vap_nominal_sample_rate
            )
            # Concatenate tensors instead of adding bytes
            self.speaker_A_step_buffer = torch.cat([self.speaker_A_step_buffer, resampled_chunk], dim=0)
        else:
            # Convert bytes to tensor and resample
            resampled_chunk = tf_resample_audio(
                waveform_tensor=s16le_audio_bytes_to_tensor(aud_chunk),
                original_sample_rate=self.speaker_B_raw_sampling_rate,
                target_sample_rate=self.vap_nominal_sample_rate
            )
            # Concatenate tensors instead of adding bytes
            self.speaker_B_step_buffer = torch.cat([self.speaker_B_step_buffer, resampled_chunk], dim=0)

        self.buffer_lock.release()

    def trigger_one_processing_step(self):


        #For both parties: remove one-step worth of audio from the step buffer to be committed to the VAP model for inference
        self.buffer_lock.acquire(blocking=True)

        spk_A_tensor_step = self.speaker_A_step_buffer[:self.step_trigger_sample_cnt]
        self.speaker_A_step_buffer = self.speaker_A_step_buffer[self.step_trigger_sample_cnt:]

        spk_B_tensor_step = self.speaker_B_step_buffer[:self.step_trigger_sample_cnt]
        self.speaker_B_step_buffer = self.speaker_B_step_buffer[self.step_trigger_sample_cnt:]

        self.buffer_lock.release()

        #Concatenate the step chunk after the existing context chunk to form the full audio bytes to commit to the VAP model
        spkA_tensor_to_commit = torch.cat([self.speaker_A_context_buffer, spk_A_tensor_step], dim=0)
        spkB_tensor_to_commit = torch.cat([self.speaker_B_context_buffer, spk_B_tensor_step], dim=0)

        #Update the context buffer once it reaches the context size -- we have a sliding window of context
        if len(spkA_tensor_to_commit) > self.context_buffer_sample_cnt:
            self.speaker_A_context_buffer = torch.cat([
                self.speaker_A_context_buffer[len(spk_A_tensor_step):], 
                spk_A_tensor_step
            ], dim=0)
        if len(spkB_tensor_to_commit) > self.context_buffer_sample_cnt:
            self.speaker_B_context_buffer = torch.cat([
                self.speaker_B_context_buffer[len(spk_B_tensor_step):], 
                spk_B_tensor_step
            ], dim=0)

        #Commit the two parties audio bytes to the VAP model for processing
        if(self.debug_time):
            t1 = time.time()
        self.logger.debug(f'Triggering VAP model with {len(spkA_tensor_to_commit)} samples of human audio and {len(spkB_tensor_to_commit)} samples of robot audio')
        _, _, res = self.__invoke_vap_model(spkA_tensor_to_commit, spkB_tensor_to_commit)

        #Interpret the result
        res = batch_to_device(res, "cpu")
        vad_prob_now = res["vad"][0, -1, 0].cpu()
        next_speaker_prob_now = res["p_now"][0, -1, 0].cpu()
        next_speaker_prob_future = res["p_future"][0, -1, 0].cpu()
        if(self.debug_time):
            t2 = time.time()
            self.logger.debug(f'VAP model returned in {t2-t1:1.3f}. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')
        else:
            self.logger.debug(f'VAP model returned. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')


        return vad_prob_now, next_speaker_prob_now, next_speaker_prob_future

    def __invoke_vap_model(self, spk_A_waveform: torch.Tensor, spk_B_waveform: torch.Tensor):

        combined_stereo_waveform = torch.stack((spk_A_waveform, spk_B_waveform), dim=0)

        combined_stereo_waveform = combined_stereo_waveform.unsqueeze(0).to(self.device)

        res = self.model.probs(combined_stereo_waveform)

        return spk_A_waveform, spk_B_waveform, res

    def main_thread(self):

        while True:
            # Check if we have enough samples (not bytes) to trigger processing            
            if (
                len(self.speaker_A_step_buffer) >= self.step_trigger_sample_cnt
                and len(self.speaker_B_step_buffer) >= self.step_trigger_sample_cnt
            ):

                vad_prob_now, next_speaker_prob_now, next_speaker_prob_future = self.trigger_one_processing_step()

                if self.report_handle is not None:
                    self.report_handle((vad_prob_now,next_speaker_prob_now,next_speaker_prob_future))

                if(self.plot):
                    #Send the float value to the socket
                    self.plot_socket.sendall(f'{vad_prob_now:.3f},{next_speaker_prob_now:.3f},{next_speaker_prob_future:.3f};'.encode('utf-8'))

    def set_report_handle(self, report_handle):
        self.report_handle = report_handle

    def start_thread(self):
        self.logger.info("Starting VAP model thread")
        self.vap_thread = threading.Thread(target=self.main_thread)
        self.vap_thread.start()



