"""
Visualization of stimulus use in each condition

External_BS/Internal_BS: samples where both subjects are speaking
External: samples where interlocutor is speaking
Internal: samples where the locutor is speaking
"""
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils.general_functions import load_pickle
from utils.processing import shifted_matrix
import config


stimulus_name = "Turn"

sessions_dic = {s: {0: None, 1: None} for s in config.sessions}
stimuli = {
    'External_BS': sessions_dic,
    'External': sessions_dic,
    'Internal': sessions_dic
}
for condition in stimuli:
    for session in tqdm(sessions_dic, desc=f"Loading stimuli for {condition}", total=len(sessions_dic)):
        for ch in [0, 1]:
            # Get relevant indexes for each subject
            samples_info = load_pickle(
                path=f"saves/preprocessed_data/External/tmin-0.2_tmax0.6/samples_info/samples_info_{session}.pkl"
            )
            trial_lengths = samples_info[f'trial_lengths{ch+1}']
            keep_indexes = samples_info[f'keep_indexes{ch+1}']

            # Load whole stimulus take average across multiple dimension
            stimulus = load_pickle(
                path=f"saves/preprocessed_data/All/tmin-0.2_tmax0.6/{stimulus_name}/Sesion{session}.pkl"
            )[ch].mean(axis=1)
            original_indexes = np.arange(stimulus.shape[0])
            mask_keep = np.zeros_like(original_indexes)
            mask_keep[keep_indexes] = 1

            # Sum 1 windows of delays = [-26, ...,  0, ..., 77] surrounding indexes to keep
            for d in config.delays:
                shifted_indexes = np.array(keep_indexes) + d
                # Only keep indexes with full windows
                if shifted_indexes.min() < 0 or shifted_indexes.max() >= stimulus.shape[0]:
                    continue
                else:
                    mask_keep[shifted_indexes] = 1
            # Shift the stimulus in order to get stimulus filter in situations
            shift_mat = shifted_matrix(
                features=stimulus,
                delays=config.delays,
                indices_to_keep=keep_indexes,
                use_gpu=True
            )
            masked_stimulus = shift_mat[:, np.where(config.delays == 0)[0]]

            trial_original_indexes = []
            trial_mask_keep = []
            trial_stimulus = []
            
            # Segment data according to trial lengths
            for l, length in enumerate(trial_lengths):
                start, end = sum(trial_lengths[:l]), sum(trial_lengths[:l+1])
                if (start == end):
                    continue
                trial_original_indexes.append(original_indexes[start:end])
                trial_mask_keep.append(mask_keep[start:end])
                trial_stimulus.append(stimulus[start:end])

            stimuli[condition][session][ch] = {
                "original_indexes": original_indexes,
                "mask_keep": mask_keep,
                "stimulus": stimulus,
                "trial_original_indexes": trial_original_indexes,
                "trial_mask_keep": trial_mask_keep,
                "trial_stimulus": trial_stimulus
            }
            
            
########################            
# Make animation
from matplotlib.widgets import Button
from scipy.io import wavfile
import sounddevice as sd
import threading
import time
class AudioPlotPlayer:
    def __init__(self, stimuli, wav_data, config, condition, session, ch, trial, sr, x_speed=1.0):
        self.stimuli = stimuli
        self.wav_data = wav_data
        self.config = config
        self.condition = condition
        self.session = session
        self.trial = trial-1
        self.sr = sr
        self.ch = ch
        self.x_speed = x_speed  # New parameter for speed
        
        self.fig, self.ax = plt.subplots(figsize=(8, 5))  
        self.cursor_line = self.ax.axvline(0, color='green', linestyle='--')
        self.playing = False
        self.stop_requested = False  # Flag to stop playback

        # Plot stimulus
        self.fig.suptitle(f"Condition: {condition}, Session: {session}, Channel: {ch+1}, Trial: {self.trial}\n\n Speed: {self.x_speed}x")
        time_axis = self.stimuli[self.condition][self.session][self.ch]["trial_original_indexes"][self.trial].astype(float)
        time_axis -= time_axis[0]
        time_axis /= config.sr
        self.ax.plot(
            time_axis,
            self.stimuli[self.condition][self.session][self.ch]["trial_stimulus"][self.trial].flatten(),
            color='blue'
        )
        stimulus = self.stimuli[self.condition][self.session][self.ch]["trial_stimulus"][self.trial].flatten()
        mask = self.stimuli[self.condition][self.session][self.ch]["trial_mask_keep"][self.trial].astype(bool).flatten()
        min_val = np.min(stimulus)
        max_val = np.max(stimulus)
        self.ax.fill_between(
            time_axis,
            np.ones_like(time_axis) * min_val,
            np.ones_like(time_axis) * max_val,
            where=mask,
            color='red',
            alpha=0.3
        )
        self.ax.grid(visible=True)
        self.ax.set_xlabel("Time (s)")
        # Dense x-axis labels
        xticks = np.arange(time_axis[0], time_axis[-1], 5)  # every 5 seconds
        self.ax.set_xticks(xticks)
        self.ax.set_xlim(time_axis[0], time_axis[-1])

        # Add play button
        ax_play = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.play_audio)

        # Add stop button
        ax_stop = plt.axes([0.68, 0.01, 0.1, 0.05])
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_stop.on_clicked(self.stop_audio)

    def play_audio(self, event):
        if not self.playing:
            self.playing = True
            self.stop_requested = False
            threading.Thread(target=self._play_and_animate, daemon=True).start()

    def stop_audio(self, event):
        self.stop_requested = True
        sd.stop()  # Stop audio playback immediately

    def _play_and_animate(self):
        try:
            try:
                # Play audio at x_speed
                sd.play(self.wav_data, int(self.sr * self.x_speed))
            except Exception as e:
                print(f"Audio playback error with sr={self.sr}: {e}")
                from scipy.signal import resample_poly
                wav_data_resampled = resample_poly(self.wav_data, 44100, self.sr)
                sd.play(wav_data_resampled, int(44100 * self.x_speed))
                self.sr = 44100
            duration = len(self.wav_data) / (self.sr * self.x_speed)
            start_time = time.time()
            while True:
                if self.stop_requested:
                    break
                elapsed = (time.time() - start_time) * self.x_speed  # Use x_speed for green line
                if elapsed > duration * self.x_speed:
                    break
                self._update_cursor(elapsed)
                self.fig.canvas.flush_events()
                time.sleep(0.0025)
            self._update_cursor(duration * self.x_speed)
            self.fig.canvas.flush_events()
            sd.wait()
        except Exception as e:
            print(f"Audio playback error: {e}")
        self.playing = False
        self.stop_requested = False

    def _update_cursor(self, current_time):
        self.cursor_line.set_xdata([current_time])
        self.fig.canvas.draw_idle()
        
if __name__ == "__main__":        

    condition, session, ch = 'External', 30, 1
    trial = 21

    wav_ch1 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel1.wav")
    wav_ch2 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel2.wav")
    if wav_ch1.exists() is False or wav_ch2.exists() is False:
        raise FileNotFoundError(f"Missing WAV files for session {session}, trial {trial}")
    # Sum wavs
    sr, wav_ch1_data = wavfile.read(wav_ch1)
    sr, wav_ch2_data = wavfile.read(wav_ch2)
    wav_data = wav_ch1_data + wav_ch2_data

    player = AudioPlotPlayer(stimuli, wav_data, config, condition, session, ch, trial, sr, x_speed=1)
    plt.show()