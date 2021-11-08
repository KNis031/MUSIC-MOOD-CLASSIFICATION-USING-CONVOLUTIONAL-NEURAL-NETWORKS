import numpy as np
import matplotlib.pyplot as plt
import commons
import librosa
import librosa.display

def example_spec_plots():
    input_file = '/Users/karlsimu/Desktop/SCHOOL/MUSICINFORMATICS/PROJECTMUINF/mtg-jamendo-dataset/data/autotagging_moodtheme.tsv'
    tracks, tags, extra = commons.read_file(input_file)

    print(tracks[7400])

    audio = np.load('/Users/karlsimu/Desktop/SCHOOL/MUSICINFORMATICS/PROJECTMUINF/data/npy/00/7400.npy')

    slice_len=1366
    middle_window = int(audio.shape[1]/2)
    audio_ms = audio[:,middle_window-(int(slice_len/2)):middle_window+(int(slice_len/2))]

    print(audio.shape)

    fig, ax = plt.subplots(figsize=(22,5))
    #plt.figure(figsize=(22,5))
    x = np.linspace(0,tracks[7400]['duration'],audio.shape[1])
    y = np.linspace(0,6000,96)
    ax.pcolormesh(x,y,audio)
    #img = librosa.display.specshow(audio, x_axis='time', y_axis='mel', cmap='viridis', fmax=6000, sr=12000, hop_length=256)
    plt.vlines(x[middle_window-(int(slice_len/2))],y[0],y[95],colors='r')
    plt.vlines(x[middle_window+(int(slice_len/2))],y[0],y[95], colors='r')
    plt.vlines(x[middle_window-1],y[43],y[46],colors='r')
    plt.hlines(y[43],x[middle_window-1],x[middle_window+2],colors='r')
    plt.vlines(x[middle_window+2],y[43],y[46], colors='r')
    plt.hlines(y[46],x[middle_window-1],x[middle_window+2], colors='r')
    ax.set(title=tracks[7400]['mood/theme'])
    ax.set_ylabel('mel')
    ax.set_xlabel('time (s)')
    plt.show()

    fig, ax = plt.subplots(figsize=(22,5))
    #plt.figure(figsize=(22,5))
    x_ms = np.linspace(0,29.1,audio_ms.shape[1])
    y = np.linspace(0,6000,96)
    ax.pcolormesh(x_ms,y,audio_ms)
    plt.vlines(x_ms[0],y[0],y[95],colors='r')
    plt.vlines(x_ms[1365],y[0],y[95], colors='r')
    plt.vlines(x_ms[int(1365/2)-1],y[43],y[46],colors='r')
    plt.vlines(x_ms[int(1365/2)+2],y[43],y[46], colors='r')
    plt.hlines(y[43],x_ms[int(1365/2)-1],x_ms[int(1365/2)+2],colors='r')
    plt.hlines(y[46],x_ms[int(1365/2)-1],x_ms[int(1365/2)+2], colors='r')
    ax.set(title=tracks[7400]['mood/theme'])
    ax.set_ylabel('mel')
    ax.set_xlabel('time (s)')
    plt.show()

    return

def main():
    example_spec_plots()

if __name__ == "__main__":
    main()