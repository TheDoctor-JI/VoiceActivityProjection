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
from vap_pool import VAPPooledObject, VAPObjectPool
from VAPWrapper import VAPWrapper
from vap_helper import get_va_states_by_speaker_bin_mask
from logger.logger import setup_logger
from FloorState.FloorStateEvent import FloorStateDef, FloorEvent, FloorEventType
from FloorState.floor_state_emission import *

def get_args():

    vap_config_path = '/home/eeyifanshen/e2e_audio_LLM/dialog_turntaking_new/VoiceActivityProjection/configs.yaml'

    # Load config from YAML file
    with open(vap_config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("VAP configuration loaded:", json.dumps(config, indent=2))
    return config


class VAPParams:

    VAP_CONFIGS = get_args()
    VAP_POOL = VAPObjectPool(configs=VAP_CONFIGS)
    EXPECTED_ENCODING = 's16le'  # Expected audio encoding
    USER_SPK_ID = 0 # We assume the user is always speaker A (id 0 in vap), and the system is speaker B
    USER_BIN_MASK = VAP_CONFIGS['interested_user_bin_pattern']  # User's bin mask
    SLEEP_INTERVAL = VAP_CONFIGS['thread_sleep_interval']

    def __init__(self, sid, socketio, event_outlet, parent_logger=None):
        try:
            self.sid = sid
            self.socketio = socketio
            self.event_outlet = event_outlet

            if parent_logger is not None:
                self.logger = parent_logger.getChild(f"VAPParams")
            else:
                self.logger = setup_logger(f"{self.sid}_VAPModule", file_log_level="DEBUG", terminal_log_level="INFO")

            ## Config for dialog state prediction
            self.vap_configs = VAPParams.VAP_CONFIGS

            self.debug_time = self.vap_configs.get('debug_time', False)

            ## Acquire VAP instance from pool
            self.vap_pool = VAPParams.VAP_POOL
            self.vap_wrapper = self.vap_pool.acquire()
            if self.vap_wrapper is None:
                raise Exception("Failed to get VAP instance from pool")
            else:
                self.vap_wrapper.set_logger(parent_logger=self.logger)  # Set the logger for the VAP wrapper
                self.logger.debug(f"Acquired VAP instance {self.vap_wrapper.id} from pool")

            self.VAP_STATE_CORRESPONDING_TO_USER_BIN_MASK = get_va_states_by_speaker_bin_mask(
                vap_wrapper=self.vap_wrapper,
                speaker_idx =VAPParams.USER_SPK_ID,
                bin_mask=VAPParams.USER_BIN_MASK,
            )

            # Control flags
            self.stop_all_threads = False

            ## Floor state machine
            self.last_user_floor_state = False  
            self.current_user_floor_state = False
            self.last_user_occupying_floor_timestamp = time.time() 
            self.latching_timeout = self.vap_configs.get('user_floor_latching_sec', 0.0)  # Default to no latching
            self.prediction_threshold = self.vap_configs.get('occupying_floor_threshold', 0.5)  # Default threshold for occupying floor state

            '''
            Audio input buffer -- these are what we send to the VAP model for processing
            '''
            self.buffer_lock = threading.Lock()

            self.clear_buffers()


        except Exception as e:
            self.logger.error(f"Error initializing VAP params: {e}")
            self.release()
            raise

    def clear_buffers(self):

        self.buffer_lock.acquire(blocking=True)

        self.speaker_A_context_buffer = torch.zeros(self.vap_wrapper.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_A_step_buffer = torch.tensor([], dtype=torch.float32)

        self.speaker_B_context_buffer = torch.zeros(self.vap_wrapper.context_buffer_sample_cnt, dtype=torch.float32)
        self.speaker_B_step_buffer = torch.tensor([], dtype=torch.float32)

        self.buffer_lock.release()

    def reset_context(self):
        """Reset the conversation context"""
        try:

            # Clear the context buffers in vap
            self.clear_buffers()

            # Input queue
            self.audio_data_input_queue = ProcPCMQueue()


        except Exception as e:
            self.logger.error(f"Error resetting context: {e}")
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
            self.logger.error(f"Error starting threads: {e}")
            self.release()
            raise

    def release(self):
        """Release resources"""
        try:
            self.stop_all_threads = True

            # Wait for threads to finish
            if hasattr(self, 'data_input_thread'):
                self.data_input_thread.join(timeout=2)
            if hasattr(self, 'vap_thread'):
                self.vap_thread.join(timeout=2)

            # Release VAP instance back to pool
            if self.vap_wrapper:
                self.vap_pool.release(self.vap_wrapper)

        except Exception as e:
            self.logger.error(f"Error releasing VAP resources: {e}")

    def enqueue_audio_data(self, identity, audio_data_dict):
        """
        Enqueue audio data for processing.
        audio_data_dict should be a dictionary with the following keys:
                {
                    'audio': audio_chunk,  # Raw audio data in bytes
                    'sr': sampling_rate,   # Sampling rate, e.g., 16000
                    'enc': encoding,       # Encoding, e.g., 's16le'
                    'time_stamp': timstamp # Timestamp for the audio chunk
                },
        """
        self.audio_data_input_queue.put(
            (
                audio_data_dict,
                identity
            )
        )
  
    def receive_raw_audio_chunk(self):

        '''
        Expect audio data of the form
        {
            'audio': <bytes>,  # Raw audio data in bytes
            'sr': <int>,      # Sampling rate, e.g., 16000
            'enc': <str>      # Encoding, e.g., 's16le'
            'time_stamp': <float> # timestamp for the audio chunk
        } 
        as well as an identity string which can be either 'user' or 'system'.
        '''
        try:
           
            while not self.stop_all_threads:

                # self.logger.debug(f"Sid: {self.sid} Received raw audio chunk for '{identity}' with size: {len(audio_dat_dict['audio'])}")

                ## Get the audio data from the input queue
                time.sleep(VAPParams.SLEEP_INTERVAL)

                data_item = self.audio_data_input_queue.get()

                if data_item is None:
                    continue

                audio_dat_dict, identity = data_item

                ## Check encoding
                if(audio_dat_dict['enc'] != 's16le'):
                    raise ValueError(f"Expected audio encoding '{VAPParams.EXPECTED_ENCODING}', but got {audio_dat_dict['enc']}")

                ## Check sampling rate, do resampling if necessary

                audio_chunk = np.frombuffer(bytes(audio_dat_dict['audio']), dtype=np.int16)
                audio_chunk = audio_chunk.astype(np.float32) / 32767.0
                audio_chunk_tensor = torch.from_numpy(audio_chunk)
                if(audio_dat_dict['sr'] != VAPWrapper.VAP_NOMINAL_SAMPLE_RATE):
                    try:
                        audio_chunk = tf_resample_audio(
                            waveform_tensor=audio_chunk_tensor, 
                            original_sample_rate=audio_dat_dict['sr'], 
                            target_sample_rate=VAPParams.VAP_NOMINAL_SAMPLE_RATE
                        )
                    except Exception as e:
                        self.logger.error(f"Error resampling audio chunk: {e}")


                ## Enqueue the audio chunk to the correct buffer for the VAP model to process
                self.send_to_step_buffer(
                    aud_chunk=audio_chunk_tensor,##Pass the audio chunk tensor directly, drop others
                    is_spk_A=(identity == 'user')##map user to speaker A and system to speaker B
                )
        
        except Exception as e:
            self.logger.error(f"Error initializing VAP params: {e}")
            self.release()
            raise

    def send_to_step_buffer(self, aud_chunk: torch.tensor, is_spk_A: bool):
        '''
        See the code below, here we expect the audio chunk to be a tensor of shape (N, ) where N is the number of samples. The expected sample rate is 16000 Hz, and the audio format is 's16le' (we would have already converted the audio bytes to a tensor before passing it here).
        '''

        self.buffer_lock.acquire(blocking=True)


        if is_spk_A:

            self.speaker_A_step_buffer = torch.cat([self.speaker_A_step_buffer, aud_chunk], dim=0)

        else:

            self.speaker_B_step_buffer = torch.cat([self.speaker_B_step_buffer, aud_chunk], dim=0)

        
        self.buffer_lock.release()

    def vap_main_thread(self):
        
        try:
            while not self.stop_all_threads:
                time.sleep(VAPParams.SLEEP_INTERVAL)

                if (
                    len(self.speaker_A_step_buffer) >= self.vap_wrapper.step_trigger_sample_cnt
                    and len(self.speaker_B_step_buffer) >= self.vap_wrapper.step_trigger_sample_cnt
                ):

                    # self.logger.debug(f"Sid: {self.sid} Triggering VAP model with step size {self.vap_wrapper.step_trigger_sample_cnt}.")

                    self.buffer_lock.acquire(blocking=True)

                    ## Consume one step worth of audio from the step buffers
                    spk_A_tensor_step = self.speaker_A_step_buffer[:self.vap_wrapper.step_trigger_sample_cnt]
                    self.speaker_A_step_buffer = self.speaker_A_step_buffer[self.vap_wrapper.step_trigger_sample_cnt:]

                    spk_B_tensor_step = self.speaker_B_step_buffer[:self.vap_wrapper.step_trigger_sample_cnt]
                    self.speaker_B_step_buffer = self.speaker_B_step_buffer[self.vap_wrapper.step_trigger_sample_cnt:]

                    self.buffer_lock.release()
                    
                    
                    ## Concatenate the step chunk after the existing context chunk to form the full audio bytes to commit to the VAP model
                    spkA_tensor_to_commit = torch.cat([self.speaker_A_context_buffer, spk_A_tensor_step], dim=0)
                    spkB_tensor_to_commit = torch.cat([self.speaker_B_context_buffer, spk_B_tensor_step], dim=0)


                    ## Update the context buffer once it reaches the context size -- we have a sliding window of context
                    ## This means we dump one step worth of audio from the head of the context buffer and add the new step chunk to the tail of the context buffer
                    if len(spkA_tensor_to_commit) > self.vap_wrapper.context_buffer_sample_cnt:
                        self.speaker_A_context_buffer = torch.cat([
                            self.speaker_A_context_buffer[len(spk_A_tensor_step):], 
                            spk_A_tensor_step
                        ], dim=0)
                    if len(spkB_tensor_to_commit) > self.vap_wrapper.context_buffer_sample_cnt:
                        self.speaker_B_context_buffer = torch.cat([
                            self.speaker_B_context_buffer[len(spk_B_tensor_step):], 
                            spk_B_tensor_step
                        ], dim=0)


                    ## Run inference on the VAP model with the two parties' audio chunks
                    if self.debug_time:
                        self.logger.debug(f"Triggering VAP model: step size {len(spk_B_tensor_step)}, context size {len(spkA_tensor_to_commit) - len(spk_A_tensor_step)}")


                    vap_result = self.vap_wrapper.trigger_one_processing_step(
                        spkA_tensor_to_commit = spkA_tensor_to_commit,
                        spkB_tensor_to_commit = spkB_tensor_to_commit
                    )

                    if self.debug_time:
                        self.logger.debug(f"VAP inference done.")

                    ## Marginalize the VAP state for the user based on the user's bin mask
                    user_speak_prob = vap_result['full_probs'][..., self.VAP_STATE_CORRESPONDING_TO_USER_BIN_MASK].sum(dim=-1)
                    user_speak_prob_value = float(user_speak_prob.item())
                    is_occupying_floor = user_speak_prob_value >= self.prediction_threshold

                    ## Update the state machine
                    res_timestamp = time.time()
                    self.last_user_floor_state = self.current_user_floor_state
                    if is_occupying_floor:#Positive flag, we stays in / goes into the occupying floor state immediately, and update the timestamp
                        self.last_user_occupying_floor_timestamp = res_timestamp
                        self.current_user_floor_state = True
                    else:
                        if self.current_user_floor_state:
                            if res_timestamp - self.last_user_occupying_floor_timestamp >= self.vap_configs['user_floor_latching_sec']:## Timeout occurred, drop out of the occupying floor state
                                self.current_user_floor_state = False
                            else:## Before the timeout, we stay in the occupying floor state
                                pass


                    vap_event = {
                        'user_speak_prob': user_speak_prob_value,
                        'is_occupying_floor': self.current_user_floor_state,
                        'last_time_occupying_floor': self.last_user_floor_state,
                        'timestamp': res_timestamp
                    }

                    ## Emit the VAP state to the gui for visualization
                    emit_vap_state_update(
                        socketio=self.socketio,
                        sid=self.sid, 
                        **vap_event
                    )

                    ## Also emit the VAP state to the event outlet for further processing
                    self.event_outlet(
                        FloorEvent(
                            event_data=vap_event,
                            event_type=FloorEventType.OCCUPYING_STATE_REPORT
                        )
                    )



                else:##Not enough new audio data to process, skip this step
                    continue
        
        except Exception as e:
            self.logger.error(f"Error initializing VAP params: {e}")
            self.release()
            raise

    def warmup_compiled_methods(self):
        ## Push a few audio samples to feature gating queue of both human and system
        num_of_chunks = 5
        for i in range(num_of_chunks):
            for identity in ['user', 'system']:
                self.enqueue_audio_data(
                    identity=identity,
                    audio_data_dict= {
                        'audio': b'\x00' * 2 * self.vap_wrapper.step_trigger_sample_cnt,  # 2 bytes per sample for 's16le', push exactly the amount of data for one step
                        'sr': VAPWrapper.VAP_NOMINAL_SAMPLE_RATE,
                        'enc': VAPParams.EXPECTED_ENCODING,
                        'time_stamp': time.time()
                    }
                )
            time.sleep(0.1)
            
        time.sleep(5)  # Give some time for the audio chunks to be processed

        ## Wait till the VAP model processes all the audio chunks
        while (len(self.speaker_A_step_buffer) > 0 or len(self.speaker_B_step_buffer) > 0):
            time.sleep(VAPParams.SLEEP_INTERVAL)

        ## Wait a bit longer to make sure the processing of the last chunk is done
        time.sleep(2)

        self.logger.debug(f"Warmed up compiled methods for user {self.sid} with {num_of_chunks} audio chunks.")