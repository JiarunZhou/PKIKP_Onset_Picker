import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
from tensorflow.keras.models import load_model
from obspy import read, Stream
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list", default="", type=str, help="Input txt file of data address")
    parser.add_argument("--len_input", default=50, type=int, help="Length of processed waveforms input into the picker (s)")
    parser.add_argument("--freq_min", default=0.5, type=float, help="Min filtering frequency (Hz)")
    parser.add_argument("--freq_max", default=2, type=float, help="Max filtering frequency (Hz)")
    parser.add_argument("--return_optimal", action="store_false", help="If only return the optimal pick and its quality")
    parser.add_argument("--save_pick", default=None, type=str, help="File outputting auto picks and qualities")
    parser.add_argument("--save_plot", default=None, type=str, help="Directory plotting waveforms labelled with auto picks")
    
    # Don't modify if no specific need
    parser.add_argument("--model_dir", default="Trained_model.keras", type=str, help="Imported Model")
    parser.add_argument("--sampling_rate", default=40, type=int, help="Sampling rate (Hz)")
    parser.add_argument("--prediction", default=60, type=float, help="Onset location predicted by the Earth model, e.g., ak135 (s)")
    parser.add_argument("--t_shift", default=0.1, type=float, help="Step of the sliding window (s)")
    
    args = parser.parse_args()

    return args


def pre_process(data, sampling_rate = 40, len_input = 50, 
                freq_min = 0.5, freq_max= 2, prediction = 60, detrend = True):
    
    # Pre-process
    data.resample(sampling_rate)
    if detrend == True: data.detrend(type="linear")
    data.filter(type="bandpass", freqmin=freq_min, freqmax=freq_max)
    
    #Cut
    st = []
    window_start = int(prediction-len_input*0.5)
    if window_start < 0:
        raise ValueError("The input waveform is too short (Default: 150 s) or the cut window is too long (Default: 50 s).")
    for tr in data:
        tr = tr[window_start*sampling_rate:(window_start+len_input)*sampling_rate]
        tr = tr/np.max(abs(tr))
        st.append(tr)
        
    return np.array(st)  


def sliding_window_picking(tr, model, sampling_rate = 40, len_window = 20, t_shift = 0.1, only_valid = True):
    n_shift = t_shift * sampling_rate
    n_length = len_window*sampling_rate
    
    tt = (np.arange(0, len(tr), n_shift)) /sampling_rate # start of each window
    
    shape = [int(np.floor(len(tr)/n_shift - n_length/n_shift + 1)),n_length]
    strides = [int(tr.strides[0]*n_shift),tr.strides[0]]
    tr_win = np.lib.stride_tricks.as_strided(tr, shape=shape, strides=strides)
    
    tr_win = normalize(tr_win,norm = "max")
    tt = tt[:len(tr_win)]
    tr_win = np.reshape(tr_win,(len(tr_win),len_window*sampling_rate,1))
    ts = model.predict(tr_win, verbose = False, batch_size=32) # pick in each window
    
    if only_valid == True:
        # Only keep the picks in range of picking windows
        ts_valid = []
        for j in range(len(ts)):
            if np.abs(ts[j]) <= len_window and np.abs(ts[j]) >= 0:
                ts_valid.append(ts[j]+tt[j]) 
            else:
                ts_valid.append([0]) # Out-of-range picks will be reset to the end
        return tt, ts_valid
    else:
        return tt,ts
    

def cluster_preds(predictions, eps=0.1, min_neighbors=5):
    dbscan = DBSCAN(eps, min_samples=min_neighbors) ## Perform DBSCAN cluster
    dbscan.fit(predictions.reshape(-1,1))
    clusters, counts = np.unique(dbscan.labels_, return_counts=True)
    dbscan_labels = dbscan.labels_
    if -1 in clusters:
        clusters = clusters[1:]
        counts = counts[1:]
    picks = np.zeros(len(clusters))
    for c in clusters:
        picks[c] = np.mean(predictions[dbscan.labels_ == c])
    
    if len(picks) == 0:
        picks == np.zeros(1)
        counts == np.zeros(1)
            
    return picks, dbscan_labels, counts    
    
    
def picker(data, model, sampling_rate = 40, t_shift = 0.1, eps = 0.1, return_optimal = True):
    len_window = 20 # model input length = sliding window length
    
    picks_highest_quality = []
    qualities_highest = []
    picks_all = []
    qualities_all = []
    
    if type(model) == str:
        model = load_model(model)
    
    for tr in data:  
        # Predict
        tt, ts_valid = sliding_window_picking(tr, model, 
                                              sampling_rate, len_window, t_shift, only_valid = True)
        
        # Cluster picks
        picks,dbscan_labels,counts = cluster_preds(np.array(ts_valid), eps=eps, min_neighbors=5)
        if len(picks) == 0:
            raise ValueError("No auto pick is returned.")
            
        # Quality of picks
        qualities = counts/(len_window/t_shift)

        picks_highest_quality.append(picks[np.argmax(qualities)])
        qualities_highest.append(np.max(qualities))
        picks_all.append(picks)
        qualities_all.append(qualities)

    if return_optimal == True:
        return picks_highest_quality, qualities_highest
    else:
        return picks_all, qualities_all   

    
def auto_pick_plot(tr, model, save_name = "Plot.jpg"):
    
    auto_picks, qualities = picker(tr, model, return_optimal = False)
    optimal_pick = auto_picks[0][np.argmax(qualities)]
    
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 1,height_ratios=(3, 2),hspace=0.1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(tr[0],c = "k",lw = 1.2)
    ax0.axvline(optimal_pick*40,c= "g",lw = 1.5,label = "Automatic pick: %.2f s"%optimal_pick)
#     ax0.axvline(onset_in_window*40,c = "r",lw = 1.5,label = "Manual pick: %.2f s"%onset_in_window)
    ax0.set_xticks(ticks = np.arange(0,2001,200), labels = [])
    ax0.set_yticks([-1,0,1])
    ax0.set_yticklabels([-1,0,1],fontsize = 14)
    ax0.set_ylabel("Amplitude",fontsize = 15) 
    ax0.set(xlim=(0,2000))
    ax0.legend(fontsize = 13,loc = "upper left")

    ax1 = fig.add_subplot(gs[1])
    ax1.axvline(optimal_pick*40,c= "g",lw = 1.5,ls = "dotted")
    for ia,a in enumerate(auto_picks[0]):
        l1 = ax1.axvline(a*40,ymin = 0,ymax = qualities[0][ia],color = "green",lw  = 1.5)
    ax1.grid(True, axis = "both",linestyle = "--")
    ax1.set_xticks(np.arange(0,2001,200))
    ax1.set_xticklabels(np.arange(-25,26,5),fontsize = 14)
    ax1.set_xlabel("Time w.r.t. ak135 prediction (s)",fontsize = 15)
    ax1.set_ylabel("Quality",fontsize = 15) 
    ax1.legend([l1],["Sliding-window \npicks"],fontsize = 13,loc = "upper right")
    plt.savefig(save_name,dpi=300,bbox_inches = "tight")  


def picking_animation(tr, model, len_input = 50, sampling_rate = 40, t_shift = 0.1, save_name = "Animation.mp4"):  
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    
    len_window = 20
    
    ## Get instantaneous picks
    tt, ts = sliding_window_picking(tr[0], model, only_valid = False)
    ts = [ts[j] + tt[j] for j in range(len(ts))]
    picks,dbscan_labels,counts = cluster_preds(np.array(ts))

    ## plot
    fig,ax = plt.subplots(2,figsize = (10,8),sharex = True)
    ax[0].set_ylim(-1.05,1.05)
    ax[0].plot(np.arange(0,len_input,1/sampling_rate),tr[0])
#     ax[0].axvline(onset_in_window, c= "r",lw = 2, ls = "--", label = "Human picked arrival")
    ln1, = ax[0].plot([],[], c= "g",lw = 2, label = "Instantaneous CNN pick")
    ln2, = ax[0].plot([], [], 'tab:grey',label = "Sliding window")
    ln3, = ax[0].plot([], [], 'tab:grey')
    ax[0].legend(loc = "upper right")
    ax[1].set_ylim(0,1)
    ax[1].set_xlabel("Time (s)",fontsize = 15)
    ax[1].set_ylabel("Quality",fontsize = 15)
    ax[1].set_xticks(np.arange(0,51,5))
    ln4, = ax[1].plot([],[],"go",ms=1.5)
    collection = ax[0].fill_between([0,len_window],-1.05,1.05,facecolor = "lightgrey",alpha = 0.5)
    xdata3,ydata3 = [],[]
    def update(frame):
        xdata = ts[frame]
        ydata = np.linspace(-1.05,1.05,100)
        xdata2 = tt[frame]
        xdata3.append(picks[dbscan_labels[frame]])
        dbscan_label_f = len([d for d in dbscan_labels[0:frame+1] if d == dbscan_labels[frame]])

        if picks[dbscan_labels[frame]] != 0 and dbscan_labels[frame] != -1:
            try:
                len_ydata3 = dbscan_label_f/(len_window/t_shift)
            except ZeroDivisionError:
                len_ydata3 = 0
        else:
            len_ydata3 = 0
        ydata3.append(len_ydata3)

        dummy = ax[0].fill_between([xdata2,xdata2+len_window], -1.05, 1.05, alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        #update the vertices of the PolyCollection
        collection.set_paths([dp.vertices])

        ln1.set_data(xdata,ydata)
        ln2.set_data([xdata2],ydata)
        ln3.set_data([xdata2+len_window],ydata)
        ln4.set_data(xdata3,ydata3)

        return ln1,ln2,ln3,ln4

    ani = FuncAnimation(fig, func = update, frames=len(ts), blit=True, repeat = False)
    writer = FFMpegWriter(fps = 15)
    ani.save(save_name,writer = writer,dpi = 300)
    
    
if __name__ == "__main__":
    args = read_args()
    if args.len_input <= 20:
        raise ValueError('The waveform input into the auto picker should be > 20 s to execute the sliding-window picking process.')
    
    ## Load data
    files = np.loadtxt(args.data_list,"str")
    st  = Stream()
    for f in files:
        st += read(f)
    
    ## Load model
    model = load_model(args.model_dir)
    
    if args.len_input != 50:
        warnings.warn("The non-default input waveform length may result in imprecise or unreliable results.")
    
    ## Pre-process data
    st_processed = pre_process(st, args.sampling_rate, args.len_input, 
                               args.freq_min, args.freq_max, args.prediction)
    
    ## Auto picking
    optimal_pick, quality = picker(st_processed, model, args.sampling_rate, args.t_shift, return_optimal = args.return_optimal)
    pick_quality = np.array([optimal_pick,quality]).T
    
    ## Output results
    for i,f in enumerate(files):
        print(f,"\n","Auto picked PKIKP onset: %.2f s; Picking quality: %.2f"%\
              (pick_quality[i,0], pick_quality[i,1]))
        print("---------------")
        
        if args.save_plot != None:
            save_name = "/".join([args.save_plot, f.split("/")[-1]+".jpg"])
            auto_pick_plot([st_processed[i]], model, save_name)
        
    if args.save_pick != None:
        np.savetxt(args.save_pick, pick_quality, fmt = "%.2f")

        
    
        
        
        
    
    