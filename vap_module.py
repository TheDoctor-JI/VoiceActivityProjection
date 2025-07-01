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

    def __init__(self, 
                model_path: str,
                context_size: int = 2.0, #Context size in seconds
                step_size: int = 0.1, #Step size in seconds
                frame_hz: int = 50,
                vap_nominal_sample_rate: int = 16000,#Has to be the sample rate accepted by the VAP model, which is 16000
                device: str = 'cuda:0',
                debug_time: bool = False):

        '''
        Load the VAP model
        '''
        self.device = device

        state_dict_path = os.path.join(model_path)
        print(f"Loading VAP model from state dict {state_dict_path}")

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
        Audio buffer requirements
        '''
        self.context_buffer_sample_cnt = int(self.context_size * self.vap_nominal_sample_rate)
        self.context_buffer_byte_cnt = self.context_buffer_sample_cnt * 2

        self.step_trigger_sample_cnt = int(self.step_size * self.vap_nominal_sample_rate)
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
            'next_speaker_prob_future': next_speaker_prob_future
        }

        return vap_result



    def __invoke_vap_model(self, spk_A_waveform: torch.Tensor, spk_B_waveform: torch.Tensor):

        combined_stereo_waveform = torch.stack((spk_A_waveform, spk_B_waveform), dim=0)

        combined_stereo_waveform = combined_stereo_waveform.unsqueeze(0).to(self.device)

        res = self.model.probs(combined_stereo_waveform)

        return spk_A_waveform, spk_B_waveform, res



def get_args():

    vap_config_path = '/home/eeyifanshen/e2e_audio_LLM/dialog_turntaking_new/VoiceActivityProjection/configs.yaml'

    # Load config from YAML file
    with open(vap_config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:", json.dumps(config, indent=2))
    return config


class VAPParams:

    VAP_CONFIGS = get_args()
    VAP_CONTEXT_SEC = VAP_CONFIGS['context_size_sec'] #Context size in seconds
    VAP_STEP_SIZE =  VAP_CONFIGS['step_size_sec'] #Step size in seconds
    VAP_AUD_FRAME_HZ = VAP_CONFIGS['frame_hz']
    EXPECTED_ENCODING = 's16le'
    EXPECTED_SAMPLING_RATE = 16000#Has to be the sample rate accepted by the VAP model, which is 16000
    VAP_PIPELINE = VAPWrapper(
        model_path=VAP_CONFIGS['model_path'],
        context_size=VAP_CONTEXT_SEC,
        step_size=VAP_STEP_SIZE,
        frame_hz=VAP_AUD_FRAME_HZ,
        vap_nominal_sample_rate=EXPECTED_SAMPLING_RATE,
        device=VAP_CONFIGS['device'],
        debug_time=False,
    )


    def __init__(self, sid, socketio):
        try:
            self.sid = sid

            ## Config for dialog state prediction
            self.vap_configs = VAPParams.VAP_CONFIGS

            ## Main VAP instance
            self.vap_wrapper = VAPParams.VAP_PIPELINE

            # Control flags
            self.stop_all_threads = False

            '''
            Audio input buffer -- these are what we send to the VAP model for processing
            '''
            self.buffer_lock = threading.Lock()

            self.clear_buffers()


        except Exception as e:
            print(f"Error initializing vap params: {e}")
            raise


    def clear_buffers(self):

        self.buffer_lock.acquire(blocking=True)

        self.speaker_A_context_buffer = torch.zeros(self.vap_wrapper.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_A_step_buffer = torch.tensor([], dtype=torch.float32)

        self.speaker_B_context_buffer = torch.zeros(self.vap_wrapper.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_B_step_buffer = torch.tensor([], dtype=torch.float32)

    def reset_context(self):
        """Reset the conversation context"""
        try:

            # Clear the context buffers in vap
            self.clear_buffers()

            # Input queue
            self.audio_data_input_queue = ProcPCMQueue()


        except Exception as e:
            print(f"Error resetting context: {e}")
            raise
    
    def start_all_threads(self):
        """Start all necessary threads for dialog state prediction"""
        try:
            # Start the data input thread to receive audio chunks
            self.data_input_thread = threading.Thread(
                target=self.receive_raw_audio_chunk,
                name="DataInput_Thread"
            )
            self.data_input_thread.start()

            # Start the VAP main thread to process audio chunks
            self.vap_thread = threading.Thread(
                target=self.vap_main_thread,
                name="VAP_Main_Thread"
            )
            self.vap_thread.start()

        except Exception as e:
            print(f"Error starting threads: {e}")
            raise

    def release(self):
        """Release resources"""
        try:
            self.stop_all_threads = True

            if hasattr(self, 'data_input_thread'):
                self.data_input_thread.join(timeout=2)
            if hasattr(self, 'vap_thread'):
                self.vap_thread.join(timeout=2)

        except Exception as e:
            print(f"Error releasing resources: {e}")

    def receive_raw_audio_chunk(self):

        '''
        Expect audio data of the form
        {
            'audio': <bytes>,  # Raw audio data in bytes
            'sr': <int>,      # Sampling rate, e.g., 16000
            'enc': <str>      # Encoding, e.g., 's16le'
            'timestamp': <float> # timestamp for the audio chunk
        } 
        as well as an identity string which can be either 'user' or 'system'.
        '''

        while not self.stop_all_threads:

            # print(f"Sid: {self.sid} Received raw audio chunk for '{identity}' with size: {len(audio_dat_dict['audio'])}")

            ## Get the audio data from the input queue
            time.sleep(0.005)

            data_item = self.audio_data_input_queue.get()

            if data_item is None:
                continue

            audio_dat_dict, identity = data_item

            ## Check encoding
            if(audio_dat_dict['enc'] != 's16le'):
                raise ValueError(f"Expected audio encoding '{VAPParams.EXPECTED_ENCODING}', but got {audio_dat_dict['enc']}")

            ## Check sampling rate, do resampling if necessary
            audio_chunk = audio_dat_dict['audio']
            if(audio_dat_dict['sr'] != VAPParams.EXPECTED_SAMPLING_RATE):

                audio_chunk = tf_resample_audio(
                    waveform_tensor=s16le_audio_bytes_to_tensor(audio_chunk), 
                    original_sample_rate=audio_dat_dict['sr'], 
                    target_sample_rate=VAPParams.EXPECTED_SAMPLING_RATE
                )
            else:
                audio_chunk = s16le_audio_bytes_to_tensor(audio_chunk)

            ## Replace the audio bytes with a tensor
            audio_dat_dict['audio'] = audio_chunk

            ## Enqueue the audio chunk to the correct buffer for the VAP model to process
            self.send_to_step_buffer(
                aud_chunk=audio_dat_dict['audio'],##Pass the audio chunk tensor directly, drop others
                is_spk_A=(identity == 'user')##map user to speaker A and system to speaker B
            )


    def send_to_step_buffer(self, aud_chunk: torch.tensor, is_spk_A: bool):
        '''
        See the code below, here we expect the audio chunk to be a tensor of shape (N, ) where N is the number of samples. The expected sample rate is 16000 Hz, and the audio format is 's16le' (we would have already converted the audio bytes to a tensor before passing it here).
        '''

        self.buffer_lock.acquire(blocking=True)

        buffer_to_concat = self.speaker_A_step_buffer if is_spk_A else self.speaker_B_step_buffer

        buffer_to_concat = torch.cat([buffer_to_concat, aud_chunk], dim=0)
        
        self.buffer_lock.release()


    def vap_main_thread(self):
        while not self.stop_all_threads:
            time.sleep(0.005)

            if (
                len(self.speaker_A_step_buffer) >= self.vap_wrapper.step_trigger_sample_cnt
                and len(self.speaker_B_step_buffer) >= self.vap_wrapper.step_trigger_sample_cnt
            ):

                self.buffer_lock.acquire(blocking=True)

                ## Consume one step worth of audio from the step buffers
                spk_A_tensor_step = self.speaker_A_step_buffer[:self.step_trigger_sample_cnt]
                self.speaker_A_step_buffer = self.speaker_A_step_buffer[self.step_trigger_sample_cnt:]

                spk_B_tensor_step = self.speaker_B_step_buffer[:self.step_trigger_sample_cnt]
                self.speaker_B_step_buffer = self.speaker_B_step_buffer[self.step_trigger_sample_cnt:]

                self.buffer_lock.release()
                
                
                ## Concatenate the step chunk after the existing context chunk to form the full audio bytes to commit to the VAP model
                spkA_tensor_to_commit = torch.cat([self.speaker_A_context_buffer, spk_A_tensor_step], dim=0)
                spkB_tensor_to_commit = torch.cat([self.speaker_B_context_buffer, spk_B_tensor_step], dim=0)


                ## Update the context buffer once it reaches the context size -- we have a sliding window of context
                ## This means we dump one step worth of audio from the head of the context buffer and add the new step chunk to the tail of the context buffer
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


                ## Run inference on the VAP model with the two parties' audio chunks
                vap_result = self.vap_wrapper.trigger_one_processing_step()

                ## Emit the VAD state
                self.emit_vad_state(vap_result)

            else:##Not enough new audio data to process, skip this step
                continue

    def emit_vad_state(self, vad_state):
        pass
