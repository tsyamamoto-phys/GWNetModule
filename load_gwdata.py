
# coding: utf-8

import numpy as np
from scipy.signal import butter, filtfilt, tukey
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import os 
import glob


def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    # len(freqs) = Nt / 2 + 1
    
    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.

    #ホワイトニング：周波数領域に変換し、asdで割った後、正規化が正しく行われるように気をつける
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def predictpsd_whiten(strain, dt, fs=4096):
    Pxx, freqs = mlab.psd(strain, Fs = fs, NFFT = 4*fs)
    psd = interp1d(freqs, Pxx)
    strain_wh = whiten(strain, psd, dt)
    return strain_wh


def bandpass(strain, fmin, fmax, fs=4096):
    # strain.shape = (# of data, length of data)
    nyq = 0.5*fs
    bb, ab = butter(6, [fmin/nyq, fmax/nyq], btype='band')
    strain_bp = filtfilt(bb, ab, strain)
    return strain_bp


def normalize(array):
    # input: (datas, length, channels)形のnumpy.array
    # output: (datas, length, channels)形のnumpy.arrayで、
    # 正規化したもの。ついでに型もnp.float32にする。
    
    tentative_max = np.amax(abs(array), axis=1, keepdims=True)
    true_max = np.amax(abs(tentative_max), axis=2, keepdims=True)
    array_norm = array / true_max
    return array_norm.astype(np.float32)




def load_traingwdata(datadir, datanum=None, mydataset=None, mylabels=None, WHITEN=False, BANDPASS=None, CUT=None, NORM=False):
    '''
    WHITEN: True or False
    BANDPASS: None or taple(fmin, fmax)
    koffset: None or taple(length, offset)
    '''
    
    filelist = glob.glob(datadir+'mockdata_*.dat')
    if datanum!=None:
        filelist = filelist[:datanum]
    all_num = len(filelist)

    meta_file = datadir + 'metadata.dat'
    fr, fi = np.genfromtxt(meta_file, usecols=(1,2), max_rows=datanum).transpose()
    mylabels = np.array((fr, fi)*1).transpose().reshape(all_num*1, 2).astype(np.float32)
    
    if mydataset==None:
        mydataset = []
    else:
        pass
    
    # loading data and whitening(optional)
    if WHITEN:
        print('loading data and whitening')
        for i in range(all_num):
            mock_file = datadir + 'mockdata_%05d.dat'%i
            hp, hc = np.loadtxt(mock_file)[1:3]
            hp_whiten = predictpsd_whiten(hp, dt=1.0/4096)
            hc_whiten = predictpsd_whiten(hc, dt=1.0/4096)
            h = np.vstack((hp_whiten, hc_whiten))
            mydataset.append(h.transpose())
    else:
        print('loading data')
        for i in range(all_num):
            mock_file = datadir + 'mockdata_%05d.dat'%i
            h = np.loadtxt(mock_file)[1:3]
            mydataset.append(h.transpose())
    
    # bandpass filter(optional)
    if BANDPASS==None:
        pass
    else:
        print('bandpass')
        fmin=BANDPASS[0]
        fmax=BANDPASS[1]
        mydataset = bandpass(mydataset, fmin, fmax)
    
    # signal cutting(optional)
    if CUT==None:
        pass
    else:
        print('cutting the data')
        mydataset_cut = []
        length = CUT[0]
        koffset = CUT[1]
        for i in range(all_num):
            hoge = np.loadtxt(filelist[i])[3]
            kcoa = np.argmax(hoge)
            kstart = kcoa + koffset# + np.random.randint(-16,16)
            h = mydataset[i,kstart:kstart+length,:]
            mydataset_cut.append(h)
        mydataset = np.array(mydataset_cut)
    
    
    
    if NORM:
        mydataset = normalize(mydataset)
    else:
        pass
    
    return np.array(mydataset), np.array(mylabels)    



def load_testgwdata(snr, mydataset=None, mylabels=None, WHITEN=False, BANDPASS=None, CUT=None, NORM=False):
    # 波形を読み込んで、合体時刻を時間の原点に合わせる。
    # これは田中さんのデータ
    snrlist = {60: 0, 30: 1, 20: 2}
    test_num = snrlist[snr]
    test_dir = 'practice20171215/'
    test_data = np.empty((0, 8192, 2), dtype=np.float64)
    t_list = np.empty((0, 8192), dtype=np.float64)

    omega_file = test_dir + "ωlist.dat"
    test_labels = np.genfromtxt(omega_file, usecols=(1,2), skip_header=5*test_num, skip_footer=10-5*test_num).transpose()
    test_labels = (test_labels / (2*np.pi)).transpose().astype(np.float32)

    for file_num in (np.arange(1,6)+5*test_num):
        mockdata_file = test_dir+"mockdata%d.dat" % file_num
        metadata_file = test_dir+"metadata%d.dat" % file_num
        t, hp, hc = np.genfromtxt(mockdata_file).transpose()
        tentative = np.array([hp[-8192:], hc[-8192:]]).transpose()
        test_data = np.append(test_data, tentative.reshape(1, 8192, 2), axis=0)
    
        ld = open(metadata_file)
        lines = ld.readlines()
        ld.close()
        for line in lines:
            if line.startswith("Approximate mereger time"):
                    t0 = float(line[:-1].replace("Approximate mereger time:", "").replace("sec",""))
                    t = t -t0
                    t = t[-8192:]
                    t_list = np.append(t_list, t.reshape(1,8192), axis=0)

                
    # これは中野さんのデータ

    test_dir = 'mockdata_nakano_20171216/'
    omega_file = test_dir + "answer_for_20171216.txt"
    t0_file = test_dir+"metadata_for_20171216.txt"
    linelist = []
    ld = open(t0_file)
    lines = ld.readlines()
    ld.close()
    for line in lines:
        if line.startswith("Approximate mereger time"):
            linelist.append(line)

    test_labels_n = np.genfromtxt(omega_file, usecols=(1,2), skip_header=5*test_num, skip_footer=10-5*test_num).transpose()
    test_labels_n = (test_labels_n).transpose().astype(np.float32)
    test_labels = np.append(test_labels, test_labels_n).reshape(10,2)
    del test_labels_n


    for file_num in (np.arange(1,6)+10+5*test_num):
        mockdata_file = test_dir+"mockdata_%03d.dat" % file_num
        metadata_file = test_dir+"metadata_%03d.dat" % file_num
        t, hp, hc = np.genfromtxt(mockdata_file).transpose()
        tentative = np.array([hp[-8192:], hc[-8192:]]).transpose()
        test_data = np.append(test_data, tentative.reshape(1, 8192, 2), axis=0)
    
        t0 = float(linelist[file_num - 11].replace("Approximate mereger time (sec):", ""))
        t = t - t0
        t = t[-8192:]
        t_list = np.append(t_list, t.reshape(1,8192), axis=0)

        
    # loading data and whitening(optional)
    if WHITEN:
        print('whitening')
        testdata_wh = []
        for i in range(10):
            hp = test_data[i,:,0]
            hc = test_data[i,:,1]
            hp_whiten = predictpsd_whiten(hp, dt=1.0/4096)
            hc_whiten = predictpsd_whiten(hc, dt=1.0/4096)
            h = np.vstack((hp_whiten, hc_whiten))
            testdata_wh.append(h.transpose())
        test_data = np.array(testdata_wh)
        
    # bandpass filter(optional)
    if BANDPASS==None:
        pass
    else:
        print('bandpass')
        fmin=BANDPASS[0]
        fmax=BANDPASS[1]
        test_data = bandpass(test_data, fmin, fmax)
        
        
    if CUT==None:
        pass
    else:
        test_data_cut = []
        length = CUT[0]
        koffset = CUT[1]
        for i in range(10):
            k = np.argmin(abs(t_list[i]))
            kstart = k+koffset#+np.random.randint(-8,8)
            test_data_cut.append(test_data[i,kstart:kstart+length,:])
        test_data = np.array(test_data_cut)
        
        
    if NORM:
        test_data = normalize(test_data)
    
    return test_data, test_labels
