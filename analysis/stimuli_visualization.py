# -*- coding: utf-8 -*-
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
import subprocess

from utils.general_functions import load_pickle
from utils.processing import shifted_matrix
import config

from matplotlib.animation import FuncAnimation, FFMpegWriter

stimulus_name = "Envelope"
stimulus_name = "Hearing-Turn"

stimuli = {
    'External_BS': {s: {0: None, 1: None} for s in config.sessions},
    'External': {s: {0: None, 1: None} for s in config.sessions},
    'Internal': {s: {0: None, 1: None} for s in config.sessions}
}
for condition in stimuli:
    for session in tqdm(config.sessions, desc=f"Loading stimuli for {condition}", total=len(config.sessions)):
        for ch in [0, 1]:
            # condition ='External'
            # session = 27
            # ch = 1
            # Get relevant indexes for each subject
            samples_info = load_pickle(
                path=f"saves/preprocessed_data/tmin-0.2_tmax0.6/samples_info/{condition}/samples_info_{session}.pkl"
            )
            trial_lengths = samples_info[f'trial_lengths{ch+1}'] # has length of trials + 1 (0 at start)
            keep_indexes = samples_info[f'keep_indexes{ch+1}']

            # Load whole stimulus take average across multiple dimension
            stimulus = load_pickle(
                    path=f"saves/preprocessed_data/tmin-0.2_tmax0.6/{stimulus_name}/Sesion{session}.pkl"
                )
            if condition.startswith("External") or stimulus_name == "Hearing-Turn":
                chan = 1 if ch == 0 else 0
                stimulus = stimulus[chan].mean(axis=1)
            else:
                stimulus = stimulus[ch].mean(axis=1)

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
                if l==0 or (start == end):
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

# =======================================================
# Get statistics of hearing times per session and channel            
hearing_times = []
for session in config.sessions:
    for ch in [0, 1]:
        hearing_time_s = sum(stimuli['External'][session][ch]['mask_keep'])/config.sr
        hearing_times.append(hearing_time_s)
median_time, percentil_0, percentil_100 = np.median(hearing_times), np.percentile(hearing_times, 0), np.percentile(hearing_times, 100)
print(f"\n\t\tMedian hearing time across all sessions and channels: {median_time//60:.0f} m {median_time%60:.0f} s ({percentil_0/60:.0f}-{percentil_100/60:.0f}) m, range\n")

speaking_times = []
for session in config.sessions:
    for ch in [0, 1]:
        speaking_time_s = sum(stimuli['External_BS'][session][ch]['mask_keep'])/config.sr
        speaking_times.append(speaking_time_s)
median_time, percentil_0, percentil_100 = np.median(speaking_times), np.percentile(speaking_times, 0), np.percentile(speaking_times, 100)
print(f"\n\t\tMedian both speaking time across all sessions and channels: {median_time//60:.0f} m {median_time%60:.0f} s ({percentil_0:.0f}-{percentil_100:.0f}) s, range\n")

# # Plot both distributions
# plt.figure(figsize=(8, 5))
# plt.hist(np.array(hearing_times)/60, bins=20, alpha=0.7, label='Hearing Time (External)', color='blue')
# plt.hist(np.array(speaking_times)/60, bins=20, alpha=0.7, label='Speaking Time (External)', color='orange')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Frequency')
# plt.title('Distribution of Hearing and Speaking Times')
# plt.legend()
# plt.show()

# ============================
# Make video of dialogue turns
from scipy.io import wavfile
class AudioPlotPlayer:
    def __init__(self, stimuli, stimulus_name, wav_data, config, session, ch, trial, sr, x_speed=1.0, n_xticks=10):
        self.stimuli = stimuli
        self.wav_data = wav_data
        self.config = config
        self.session = session
        self.trial = trial-1
        self.x_speed = x_speed
        self.sr = sr
        self.ch = ch

        # Use smaller figure size for lower resolution
        self.fig, self.ax = plt.subplots(figsize=(8, 5))  # Reduced from (8, 5)
        self.fig.suptitle(
            f"Session: {session}, Channel: {ch+1}, Trial: {self.trial+1}\n\n Speed: {self.x_speed}x"
        )

        self.cursor_line = self.ax.axvline(0, color='green', linestyle='--')
        self.stop_requested = False
        self.playing = False

        self.video_filename = f"turns_media/videos/{stimulus_name}_session{session}_ch{ch+1}_trial{self.trial+1}.mp4"

        # Plot both conditions
        time_axis = self.stimuli['External'][self.session][self.ch]["trial_original_indexes"][self.trial].astype(float)
        time_axis -= time_axis[0]
        time_axis /= config.sr
        
        stimulus_ext = self.stimuli['External'][self.session][self.ch]["trial_stimulus"][self.trial].flatten()
        if stimulus_name == "Turn":
            stimulus_int = np.zeros_like(stimulus_ext)
            stimulus_ext_bs = np.zeros_like(stimulus_ext)
        else:
            stimulus_int = self.stimuli['Internal'][self.session][self.ch]["trial_stimulus"][self.trial].flatten()
            stimulus_ext_bs = self.stimuli['External_BS'][self.session][self.ch]["trial_stimulus"][self.trial].flatten()
        stimulus = stimulus_ext_bs + stimulus_ext + stimulus_int
        stimulus = stimulus_ext + stimulus_int

        mask_ext_bs = self.stimuli['External_BS'][self.session][self.ch]["trial_mask_keep"][self.trial].astype(bool).flatten()
        mask_ext = self.stimuli['External'][self.session][self.ch]["trial_mask_keep"][self.trial].astype(bool).flatten()
        mask_int = self.stimuli['Internal'][self.session][self.ch]["trial_mask_keep"][self.trial].astype(bool).flatten()
        min_val = np.min(stimulus)
        max_val = np.max(stimulus)
        if min_val == max_val:
            min_val -= 1
            max_val += 1

        self.ax.fill_between(
            time_axis,
            np.ones_like(time_axis) * min_val,
            np.ones_like(time_axis) * max_val,
            where=mask_ext_bs,
            color='green',
            alpha=0.3,
            label='Both Speaking'
        )
        self.ax.fill_between(
            time_axis,
            np.ones_like(time_axis) * min_val,
            np.ones_like(time_axis) * max_val,
            where=mask_ext,
            color='red',
            alpha=0.3,
            label='External'
        )
        self.ax.fill_between(
            time_axis,
            np.ones_like(time_axis) * min_val,
            np.ones_like(time_axis) * max_val,
            where=mask_int,
            color='blue',
            alpha=0.3,
            label='Internal'
        )
        
        self.ax.plot(
            time_axis,
            stimulus,
            color='black'
        )
            
        self.ax.grid(visible=True)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_xlim(time_axis[0], time_axis[-1])
        xticks = np.arange(time_axis[0], time_axis[-1], (time_axis[-1]-time_axis[0])/n_xticks)
        self.ax.set_xticks(xticks)
        self.ax.legend()

        # Chequeo de coincidencia de índices de sombreado
        mask_external = self.stimuli['External'][self.session][self.ch]["trial_mask_keep"][self.trial]
        mask_internal = self.stimuli['Internal'][self.session][self.ch]["trial_mask_keep"][self.trial]
        if np.array_equal(mask_external, mask_internal):
            raise ValueError("ATENCIÓN: Los índices de sombreado (mask_keep) de 'External' e 'Internal' son IGUALES para session={}, ch={}, trial={}".format(self.session, self.ch, self.trial+1))

    def make_video(self):
        duration = len(self.wav_data) / (self.sr * self.x_speed)
        fps = 30
        n_frames = int(duration * fps)
        time_axis = self.stimuli['External'][self.session][self.ch]["trial_original_indexes"][self.trial].astype(float)
        time_axis -= time_axis[0]
        time_axis /= self.config.sr

        def update(frame):
            current_time = frame / fps
            self.cursor_line.set_xdata([current_time])
            return self.cursor_line,

        writer = FFMpegWriter(fps=fps)  # Remove dpi argument here
        print(f"Saving video to {self.video_filename} ...")
        anim = FuncAnimation(self.fig, update, frames=n_frames, blit=True)
        anim.save(self.video_filename, writer=writer, dpi=120)
        print(f"Video saved. Play {self.video_filename} and the corresponding WAV file together for perfect sync.")

        # --- Add audio to video using ffmpeg ---
        audio_path = f"turns_media/{self.session}_{self.trial+1}_audio.wav"
        # Save the audio as a WAV file
        from scipy.io.wavfile import write as wav_write
        wav_write(audio_path, self.sr, self.wav_data.astype(np.int16))
        output_path = self.video_filename.replace('.mp4', '_with_audio.mp4')
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', self.video_filename,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            output_path
        ]
        print(f"Combining video and audio to {output_path} ...")
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Final video with audio saved to {output_path}")
        except Exception as e:
            print(f"ffmpeg failed: {e}")

        self.playing = False
        self.stop_requested = False
        
        # Delete temporal files
        Path(audio_path).unlink(missing_ok=True)
        Path(self.video_filename).unlink(missing_ok=True)

    def _update_cursor(self, current_time):
        self.cursor_line.set_xdata([current_time])
        self.fig.canvas.draw_idle()
        
if __name__ == "__main__":        
    
    # # Desde la perspectiva del participante con canal 0. Es decir, se ve externo los momentos de habla de 1 y el estimulo de 1. Viceversa
    # session, ch = 21, 0
    # trial = 1 #TRIAL 4, 6(dudoso), 10(dudoso), 11(dudoso), 18(dudoso), 19(dudoso), 
    # # 22(habla uno), 26(habla uno y dudoso pq no), 28 (estsa maso) SESSION 27 ESTA MAL ANOTADO Y SON MUY COORTOS. EN 2 no habla un canal
    
    # wav_ch1 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel1.wav")
    # wav_ch2 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel2.wav")
    # if wav_ch1.exists() is False or wav_ch2.exists() is False:
    #     raise FileNotFoundError(f"Missing WAV files for session {session}, trial {trial}")
    # # Sum wavs
    # sr, wav_ch1_data = wavfile.read(wav_ch1)
    # sr, wav_ch2_data = wavfile.read(wav_ch2)
    # wav_data = wav_ch1_data + wav_ch2_data

    # player = AudioPlotPlayer(stimuli, 'Envelope', wav_data, config, session, ch, trial, sr, x_speed=1)
    # player.make_video()
    from utils.load_utils import get_trials
    for session in tqdm(config.sessions, total=len(config.sessions)):
        for ch in [0, 1]:
            for trial in get_trials(session):
    
                wav_ch1 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel1.wav")
                wav_ch2 = Path(rf"data\wavs\S{session}\s{session}.objects.{trial:02}.channel2.wav")
                if wav_ch1.exists() is False or wav_ch2.exists() is False:
                    raise FileNotFoundError(f"Missing WAV files for session {session}, trial {trial}")
                # Sum wavs
                sr, wav_ch1_data = wavfile.read(wav_ch1)
                sr, wav_ch2_data = wavfile.read(wav_ch2)
                wav_data = wav_ch1_data + wav_ch2_data

                player = AudioPlotPlayer(stimuli, 'Envelope', wav_data, config, session, ch, trial, sr, x_speed=1)
                player.fig.savefig(f"turns_media/figs/{stimulus_name}_session{session}_ch{ch+1}_trial{trial}.png", dpi=150)
                plt.close('all')