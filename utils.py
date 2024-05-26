import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def get_final_circle_freqs(embeddings):
    embedding = embeddings[-1]
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    sorted_freq = np.argsort(signal)[::-1]
    threshold = np.mean(signal) * 2 
    num_circles = (signal > threshold).sum() // 2
    cur_freqs = [min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) for i in range(num_circles)]
    return list(zip(cur_freqs, signal[cur_freqs]))


def plot_ode_traj(real, predicted): 
    for i in range(29):
        plt.figure(figsize=(10, 5))
        plt.plot(real[:, i], label='Original Signal')
        plt.plot(predicted[:, i], label='Predicted Signal')
        plt.title('Original vs. Predicted Signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.title(f'Frequency {i+1}')
        plt.legend()
        plt.show()
        # plt.savefig(f'figs/evolution/version0/accumulated_signal_freq_{i}.png')
        
def rollout_ode_traj(reg, real):
    cur_traj = real[0:1, :29].copy()
    trajs = real[0:1, :29].copy()
    cur_x = real[0:1].copy()
    for i in range(real.shape[0]-1):
        predictions = reg.predict(cur_x)
        cur_traj += predictions
        trajs = np.concatenate(
            [trajs, cur_traj], axis=0
        )
        cur_x = cur_traj
    return trajs
    
def format_subplot(ax, grid_x=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)