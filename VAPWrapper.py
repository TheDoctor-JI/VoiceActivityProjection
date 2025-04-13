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

from logger.logger import setup_logger
import os, sys
import threading
from utils.audio_helpers import s16le_audio_bytes_to_tensor, tf_resample_audio
import collections
import socket

# everything_deterministic()
# torch.manual_seed(0)


class VAPWrapper:

    def __init__(self, global_config, audio_config, parent_logger, report_handle):
        self.global_config = global_config
        self.audio_config = audio_config
        self.parent_logger = parent_logger

        '''
        Load the VAP model
        '''
        self.device = self.audio_config['vap_model']['device']
        if parent_logger is None:
            self.logger = setup_logger('VAPWrapper')
        else:
            self.logger = parent_logger.getChild('VAPWrapper')

        state_dict_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.audio_config['vap_model']['state_dict_path']
        )
        self.logger.info(f"Loading model from state dict {state_dict_path}")
        self.vap_conf = VapConfig(
            sample_rate=self.audio_config['vap_model']['sample_rate'],
            frame_hz=self.audio_config['vap_model']['frame_hz'],
        )
        self.model = VapGPT(self.vap_conf)
        sd = torch.load(state_dict_path)
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.debug_time = self.audio_config['vap_model']['debug_time']
        self.plot = self.audio_config['vap_model']['plot_vap']

        '''
        Audio input configs
        '''
        if(self.audio_config['hw_params']['format_str'] != 's16le'):
            raise ValueError(f"Unsupported sample format string {self.audio_config['hw_params']['format_str']}")
        self.context_size = self.audio_config['vap_model']['context_size']
        self.step_size = self.audio_config['vap_model']['step_size']

        '''
        Audio input buffer
        '''
        self.buffer_lock = threading.Lock()

        #We assume the human (mic-stream) is speaker A in the VAP model narration
        self.human_speaker_A_context_buffer_byte_cnt = self.context_size * self.audio_config['hw_params']['microphone']['sampling_rate'] * 2
        self.human_speaker_A_context_buffer = b'\x00' * self.human_speaker_A_context_buffer_byte_cnt#Initialize with silence
        self.human_triggering_step_buffer_byte_cnt = int(self.step_size * self.audio_config['hw_params']['microphone']['sampling_rate'] * 2)
        self.human_speaker_A_step_buffer = b''

        #We assume the robot (speaker-stream) is speaker B in the VAP model narration
        self.robot_speaker_B_context_buffer_byte_cnt = self.context_size * self.audio_config['hw_params']['speaker']['sampling_rate'] * 2
        self.robot_speaker_B_context_buffer = b'\x00' * self.robot_speaker_B_context_buffer_byte_cnt#Initialize with silence
        self.robot_triggering_step_buffer_byte_cnt = int(self.step_size * self.audio_config['hw_params']['speaker']['sampling_rate'] * 2)
        self.robot_speaker_B_step_buffer = b''

        '''
        Miscellaneous
        '''
        self.report_handle = report_handle
        if(self.plot):
            #Prepare a socket to local host 8080, to which we will send data
            self.plot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.plot_socket.connect(('localhost', self.audio_config['vap_model']['plotting_port']))

    def recv_audio_chunk(self, aud_chunk: bytes, is_human: bool):
        self.buffer_lock.acquire(blocking=True)
        if is_human:
            self.human_speaker_A_step_buffer += aud_chunk
        else:
            self.robot_speaker_B_step_buffer += aud_chunk
        self.buffer_lock.release()

    def invoke_vap_model(self, human_bytes: bytes, robot_bytes: bytes):

        human_waveform = tf_resample_audio(
                            waveform_tensor=s16le_audio_bytes_to_tensor(human_bytes), 
                            original_sample_rate=self.audio_config['hw_params']['microphone']['sampling_rate'], 
                            target_sample_rate=self.vap_conf.sample_rate
                        )
        robot_waveform = tf_resample_audio(
                            waveform_tensor=s16le_audio_bytes_to_tensor(robot_bytes), 
                            original_sample_rate=self.audio_config['hw_params']['speaker']['sampling_rate'], 
                            target_sample_rate=self.vap_conf.sample_rate
                        )
        combined_stereo_waveform = torch.stack((human_waveform, robot_waveform), dim=0)
        combined_stereo_waveform = combined_stereo_waveform.unsqueeze(0).to(self.device)
        res = self.model.probs(combined_stereo_waveform)
        return human_waveform, robot_waveform, res


    def main_thread(self):

        while True:

            if (
                len(self.human_speaker_A_step_buffer) >= self.human_triggering_step_buffer_byte_cnt
                and len(self.robot_speaker_B_step_buffer) >= self.robot_triggering_step_buffer_byte_cnt
            ):
                self.buffer_lock.acquire(blocking=True)

                #For both parties: keep the new step chunk, remove one chunk from the step buffer
                human_bytes_step = self.human_speaker_A_step_buffer[:self.human_triggering_step_buffer_byte_cnt]
                self.human_speaker_A_step_buffer = self.human_speaker_A_step_buffer[self.human_triggering_step_buffer_byte_cnt:]

                robot_bytes_step = self.robot_speaker_B_step_buffer[:self.robot_triggering_step_buffer_byte_cnt]
                self.robot_speaker_B_step_buffer = self.robot_speaker_B_step_buffer[self.robot_triggering_step_buffer_byte_cnt:]

                self.buffer_lock.release()

                #Concatenate the step chunk after the context chunk
                human_bytes_to_commit = self.human_speaker_A_context_buffer + human_bytes_step
                robot_bytes_to_commit = self.robot_speaker_B_context_buffer + robot_bytes_step

                #Update the context buffer
                self.human_speaker_A_context_buffer = self.human_speaker_A_context_buffer[len(human_bytes_step):] + human_bytes_step
                self.robot_speaker_B_context_buffer = self.robot_speaker_B_context_buffer[len(robot_bytes_step):] + robot_bytes_step

                #Commit the two parties audio bytes to the VAP model for processing
                if(self.debug_time):
                    t1 = time.time()
                self.logger.debug(f'Triggering VAP model with {len(human_bytes_to_commit)} bytes of human audio and {len(robot_bytes_to_commit)} bytes of robot audio')
                human_waveform, robot_waveform, res = self.invoke_vap_model(human_bytes_to_commit, robot_bytes_to_commit)
                res = batch_to_device(res, "cpu")
                vad_prob_now = res["vad"][0, -1, 0].cpu()
                next_speaker_prob_now = res["p_now"][0, -1, 0].cpu()
                next_speaker_prob_future = res["p_future"][0, -1, 0].cpu()
                if(self.debug_time):
                    t2 = time.time()
                    self.logger.debug(f'VAP model returned in {t2-t1:1.3f}. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')
                else:
                    self.logger.debug(f'VAP model returned. P_now: {next_speaker_prob_now} P_future: {next_speaker_prob_future}')

                self.report_handle((vad_prob_now,next_speaker_prob_now,next_speaker_prob_future))

                if(self.plot):
                    #Send the float value to the socket
                    self.plot_socket.sendall(f'{vad_prob_now:.3f},{next_speaker_prob_now:.3f},{next_speaker_prob_future:.3f};'.encode('utf-8'))

    def set_report_handle(self, report_handle):
        self.report_handle = report_handle

    def start(self):
        self.logger.info("Starting VAP model thread")

        self.vap_thread = threading.Thread(target=self.main_thread)
        self.vap_thread.start()



