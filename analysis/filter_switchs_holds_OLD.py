"""
This script extracts IPU's from interlocutor turns in order to use as encoding
features under the External condition (predicting EEG from interlocutor voice)

# TODO: habría que identificar holds que podrían haber sido switches. Es decir, 
momentos en los cuales el interlocutor da señales de terminar de hablar, pero 
sigue hablando. Un indicador que cuantifique este efecto debería ser el ground
truth.
Encontrar correlatos neuronales enmarañados en cambios de turnos confusos va a 
ser dificil sino (si a nivel comportamental no se evidencian, por qué esperarí
amos un correlato a nivel cerebro).
"""
import pandas as pd
import config
import json

# Read data
turn_table_path = config.base_path / "turns/turn_table.csv"
turn_table = pd.read_csv(
    turn_table_path, header=0, delimiter=','
)
total_sessions = config.sessions + config.held_out_sessions

# Get switches
turn_table_s = turn_table[turn_table['tt_label']=='S']
switch_statistics = {s:{} for s in total_sessions}
for session in total_sessions:
    regex_str = rf's{session}'# regex_str = rf's{session}+\.objects\.\d+'
    ses_turn_table = turn_table_s[turn_table_s['trial_id'].str.contains(regex_str, regex=True)]
    trials = [trial[-2:] for trial in ses_turn_table['trial_id'].unique().tolist()]
    session_data = []
    for trial in trials:
        trial_turn_table = ses_turn_table[ses_turn_table['trial_id'].str.endswith(trial)]
        
        # Hearing switch corresponds to a switch from the interlocutor
        trial_turn_table_ch1 = trial_turn_table[trial_turn_table['speaker1'].str.endswith('1')]
        trial_turn_table_ch2 = trial_turn_table[trial_turn_table['speaker1'].str.endswith('2')]

        trial_data = {1:[], 2:[]}
        for k, trial_turn_table in enumerate([trial_turn_table_ch1, trial_turn_table_ch2], start=1):
            for switch_index in trial_turn_table.index:
                switch = trial_turn_table.loc[switch_index]
                
                # Discard transitions with superposition
                if (switch["ipu2_start_time"]-switch["ipu1_end_time"])<0:
                    continue
                # Keep turns over 400 ms (to have enough voice to make the prediction)
                if (switch["ipu1_end_time"]-switch["ipu1_start_time"])<.4:
                    continue
                # Filter switches that are preceded by ipu shorter than 500 ms (prevent backchannels)
                elif (switch["ipu2_end_time"]-switch["ipu2_start_time"])<.5:
                    continue

                # Check we are analyzing a switch of the interlocutor to hearer
                speaker = int(switch["speaker1"][-1])
                interlocutor = int(switch["speaker2"][-1])
                # if (speaker!=(-k+3)) and (interlocutor!=(k)):
                #     raise ValueError("Unexpected speaker and/or interlocutor")

                json_data = {
                    "speaker": speaker,
                    "interlocutor": interlocutor,
                    "ipu1_start_time": float(switch["ipu1_start_time"]),
                    "ipu1_end_time": float(switch["ipu1_end_time"])
                }
                trial_data[k].append(json_data)

            # The filename change the channel number in order to use cues from the interlocutor to predict
            filename = turn_table_path.parent / "switches_external" / f"sess_{session}_trial_{trial}_ch_{k}.json"
            filename.parents[0].mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as json_file:
                json.dump(trial_data[k], json_file, indent=2)
            session_data.append(trial_data)
    
    # Store data for switch_statistics
    switch_statistics[session] = session_data

# Now save holds
turn_table_h = turn_table[turn_table['tt_label']=='H']
hold_statistics = {s:{} for s in total_sessions}
for session in total_sessions:
    regex_str = rf's{session}'# regex_str = rf's{session}+\.objects\.\d+'
    ses_turn_table = turn_table_h[turn_table_h['trial_id'].str.contains(regex_str, regex=True)]
    trials = [trial[-2:] for trial in ses_turn_table['trial_id'].unique().tolist()]
    
    session_data = []
    for trial in trials:
        trial_turn_table = ses_turn_table[ses_turn_table['trial_id'].str.endswith(trial)]

        # Hearing a hold corresponds to still being engaged with the interlocutor
        trial_turn_table_ch1 = trial_turn_table[trial_turn_table['speaker1'].str.endswith('2')]
        trial_turn_table_ch2 = trial_turn_table[trial_turn_table['speaker1'].str.endswith('1')]

        trial_data = {1:[], 2:[]}
        for k, trial_turn_table in enumerate([trial_turn_table_ch1, trial_turn_table_ch2], start=1):
            for hold_index in trial_turn_table.index:
                hold = trial_turn_table.loc[hold_index]
                
                # Discard transitions with superposition
                if (hold["ipu2_start_time"]-hold["ipu1_end_time"])<0:
                    continue
                # Keep turns over 400 ms (to have enough voice to make the prediction)
                if (hold["ipu1_end_time"]-hold["ipu1_start_time"])<.4:
                    continue
                # It's not necessary to prevent backchannels in holds (but, why not)
                elif (hold["ipu2_end_time"]-hold["ipu2_start_time"])<.5:
                    continue

                # Check we are analyzing a hold of the interlocutor to hearer
                speaker = int(hold["speaker1"][-1])
                interlocutor = int(hold["speaker2"][-1])
                if (speaker!=(-k+3)) and (interlocutor!=(k)):
                    raise ValueError("Unexpected speaker and/or interlocutor")

                json_data = {
                    "speaker": speaker,
                    "interlocutor": interlocutor,
                    "ipu1_start_time": float(hold["ipu1_start_time"]),
                    "ipu1_end_time": float(hold["ipu1_end_time"])
                }
                trial_data[k].append(json_data)

            # The filename change the channel number in order to use cues from the interlocutor to predict
            filename = turn_table_path.parent / "holds_external" / f"sess_{session}_trial_{trial}_ch_{k}.json"
            filename.parents[0].mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as json_file:
                json.dump(trial_data[k], json_file, indent=2)
            session_data.append(trial_data)
            
    # Store data for hold_statistics
    hold_statistics[session] = session_data

# ======================================
# Visualize switches and hold statistics
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for visualization
switch_counts = []
hold_counts = []

for session in total_sessions:
    for ch in [1, 2]:
        subject = f"s{session}c{ch}"
        switch_count = sum(len(trial_data[ch]) for trial_data in switch_statistics[session])
        hold_count = sum(len(trial_data[ch]) for trial_data in hold_statistics[session])
        switch_counts.append({
            "Subject": subject,
            "Count": switch_count,
            "Type": "Switch"
        })
        hold_counts.append({
            "Subject": subject,
            "Count": hold_count,
            "Type": "Hold"
        })

stats_df = pd.DataFrame(switch_counts + hold_counts)

plt.figure(figsize=(10, 6))
plt.grid(visible=True)
plt.yticks(range(0, max(stats_df["Count"]) + 1, 50))

# Get consistent colors for each class
palette = sns.color_palette("tab10", n_colors=2)
event_types = stats_df["Type"].unique()
color_map = {etype: palette[i] for i, etype in enumerate(event_types)}

ax = sns.barplot(
    data=stats_df,
    x="Subject",
    y="Count",
    hue="Type",
    palette=color_map,
    errorbar=None
)
plt.title("Number of Switches and Holds per Subject (Hearing)")
plt.xlabel("Subject")
plt.ylabel("Count")

# Add mean and std lines for each class with matching color
for event_type in event_types:
    class_counts = stats_df[stats_df["Type"] == event_type]["Count"]
    mean = class_counts.mean()
    std = class_counts.std()/(len(class_counts)**(1/2))
    color = color_map[event_type]
    plt.axhline(mean, color=color, linestyle='--', linewidth=2, label=f"{event_type} Mean")
    plt.fill_between(
        x=[-0.5, len(stats_df["Subject"].unique())-0.5],
        y1=mean-std, y2=mean+std,
        color=color, alpha=0.2, label=f"{event_type} ±1 SD of Mean"
    )
plt.legend(title="Event Type")
plt.tight_layout()
fig_path = config.figures_path / "cues_statistics" / "hearing" / "switch_hold_statistics.png"
fig_path.parents[0].mkdir(parents=True, exist_ok=True)
plt.savefig(
    fig_path,
    dpi=400
    )
# plt.show()

# Print summary statistics for switches and holds per subject
print("Switch and Hold Counts per Subject:")
for subject in stats_df["Subject"].unique():
    sub_df = stats_df[stats_df["Subject"] == subject]
    switch_count = sub_df[sub_df["Type"] == "Switch"]["Count"].values[0]
    hold_count = sub_df[sub_df["Type"] == "Hold"]["Count"].values[0]
    print(f"{subject}: Switches = {switch_count}, Holds = {hold_count}")

for event_type in event_types:
    class_counts = stats_df[stats_df["Type"] == event_type]["Count"]
    mean = class_counts.mean()
    std = class_counts.std()/(len(class_counts)**(1/2))
    print(f"\n{event_type}: Mean = {mean:.2f}, SEM = {std:.2f}")