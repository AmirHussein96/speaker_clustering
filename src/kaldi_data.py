import os
import numpy as np
import subprocess

def read_data_list(data,utt2spk=False,utt2lang=False,segments=False):
    fileid = open(data+'/wav.scp','r')
    temp = fileid.readlines()
    fileid.close()

    filelist = []
    utt_label = []
    for iter,line in enumerate(temp):
        if len(line.split(' ')) ==2:
            filelist.extend([line.rstrip().split(' ')[-1]])
            utt_label.extend([line.rstrip().split(' ')[0]])
        else:
            if len(line.split('.sph'))==2:
                filelist.extend([line.rstrip().split('.sph')[0].split(' ')[-1] + '.sph'])
                utt_label.extend([line.rstrip().split(' ')[0]])
            elif len(line.split('.flac'))==2:
                filelist.extend([line.rstrip().split('.flac')[0].split(' ')[-1] + '.flac'])
                utt_label.extend([line.rstrip().split(' ')[0]])
    filelist = np.array(filelist)
    utt_label = np.array(utt_label)


    if utt2spk!=False:
        lines = open(data+'/utt2spk','r').readlines()
        spk_label = []
        for line in lines:
            spk_label.extend([line.rstrip().split()[-1]])
        spk_label = np.array(spk_label)
    if utt2lang!=False:
        lines = open(data+'/utt2lang','r').readlines()
        lang_label = []
        for line in lines:
            lang_label.extend([line.rstrip().split()[-1]])
        lang_label = np.array(lang_label)

    if segments!=False:

        utt2wav={}
        for iter in range(len(filelist)):
            print(utt_label[iter])
            print(filelist[iter])
            utt2wav[utt_label[iter]]=filelist[iter]

        lines=open(data+'/segments','r').readlines()
        seg_segid = []
        seg_uttid = []
        seg_windows = []
        seg_filelist = []
        for iter in range(len(lines)):
            seg_filelist.append( utt2wav[lines[iter].rstrip().split()[1]] )
            seg_segid.append( lines[iter].rstrip().split()[0] )
            seg_uttid.append( lines[iter].rstrip().split()[1] )
            seg_windows.append([ np.float(lines[iter].rstrip().split()[2]), np.float(lines[iter].rstrip().split()[3])])


    if segments:
        if (utt2spk!=False) and (utt2lang!=False):
            return filelist, utt_label, spk_label, lang_label, seg_filelist, seg_segid, seg_uttid, seg_windows
        elif (utt2spk!=False) and (utt2lang==False):
            return filelist, utt_label, spk_label, seg_filelist, seg_segid, seg_uttid, seg_windows
        elif (utt2spk==False) and (utt2lang!=False):
            return filelist, utt_label, lang_label, seg_filelist, seg_segid, seg_uttid, seg_windows
        else:
            return filelist, utt_label, seg_filelist, seg_segid, seg_uttid, seg_windows
    else:
        if (utt2spk!=False) and (utt2lang!=False):
            return filelist, utt_label, spk_label, lang_label
        elif (utt2spk!=False) and (utt2lang==False):
            return filelist, utt_label, spk_label
        elif (utt2spk==False) and (utt2lang!=False):
            return filelist, utt_label,lang_label
        else:
            return filelist, utt_label


def label2num(label,original_label):
    fid = open(original_label)
    lines = fid.readlines()
    fid.close()
    spks=[]
    for iter, line in enumerate(lines):
        spks.append(line.rstrip().split()[-1])
    spks = np.unique(spks)
    spk_dict=dict()
    for iter in range(len(spks)):
        spk_dict[spks[iter]]=iter
    spk_label_num = []
    for iter in range(len(label)):
        spk_label_num.append(spk_dict[label[iter]])
    spk_label_num = np.array(spk_label_num)
    return spk_label_num

def split_data(name,filelist,utt_label,spk_label=[],lang_label=[],total_split=1):
    split_len = len(utt_label)/total_split
    overflow = len(utt_label)%total_split
    print ("Total splits = "+str(total_split)+", average length per split = "+str(split_len))
#     os.mkdir(name+'/split'+str(total_split))
    subprocess.call(['mkdir','-p', name+'/split'+str(total_split)])
    start=0
    end_=0

    for split in range(1,total_split+1):
        os.mkdir(name+'/split'+str(total_split)+'/'+str(split))
        filename_wav = name+'/split'+str(total_split)+'/'+str(split)+'/wav.scp'
        filename_utt2spk = name+'/split'+str(total_split)+'/'+str(split)+'/utt2spk'
        filename_utt2lang = name+'/split'+str(total_split)+'/'+str(split)+'/utt2lang'
        start = end_
        end_ = start+split_len
        if overflow>0:
            overflow-=1
            end_+=1
        if split==total_split:
            end_ = len(utt_label)
        with open(filename_wav,'w') as file:
            for iter in range(start,end_):
                file.write('%s %s\n' % (utt_label[iter], filelist[iter]) )
        if len(spk_label)!=0:
            with open(filename_utt2spk,'w') as file:
                for iter in range(start,end_):
                    file.write('%s %s\n' % (utt_label[iter], spk_label[iter]) )
        if len(lang_label)!=0:
            with open(filename_utt2lang,'w') as file:
                for iter in range(start,end_):
                    file.write('%s %s\n' % (utt_label[iter], lang_label[iter]) )





def split_segments(name,segments,total_split):

    #generate uttid to wavfilename dictionary
    wavlist,utt_label = read_data_list(name)
    # print('here', wavlist)
    # print(utt_label)
    utt2wav={}
    for iter,line in enumerate(wavlist):
        utt2wav[utt_label[iter]]=line

    split_len = int(len(segments)/total_split)
    overflow = len(segments)%total_split
    print (split_len,overflow)
    start=0
    end_=0

    for split in range(1,total_split+1):
        subprocess.call(['mkdir','-p', name+'/split'+str(total_split)+'/'+str(split)])
        filename_segments = name+'/split'+str(total_split)+'/'+str(split)+'/segments'
        filename_wavlist = name+'/split'+str(total_split)+'/'+str(split)+'/wav.scp'
        start = end_
        end_ = start+split_len
        # print(start, end_)
        if overflow>0:
            overflow-=1
            end_+=1
        if split==total_split:
            end_ = len(segments)

        split_wavlist = []
        split_uttid = []
        with open(filename_segments,'w') as file:
            for iter in range(start,end_):
                file.write('%s\n' % (segments[iter].rstrip()) )
                uttid = segments[iter].rstrip().split()[1]
                # print(uttid)
                split_uttid.append(uttid )
                split_wavlist.append(utt2wav[uttid])


        split_uttid = np.unique(split_uttid)
        split_wavlist = np.unique(split_wavlist)
        with open(filename_wavlist,'w') as file:
            for iter in range(len(split_wavlist)):
                file.write('%s %s\n' %( split_uttid[iter], split_wavlist[iter] ) )




def write_data(name,filelist,utt_label,spk_label=[],lang_label=[]):
#     os.mkdir(name)
    subprocess.call(['mkdir','-p', name])

    filename = name+'/wav.scp'
    with open(filename,'w') as file:
        for iter in range(len(utt_label)):
            file.write('%s %s\n' % (utt_label[iter], filelist[iter]) )

    # Utterance label using number
    if len(spk_label)!=0:
        filename = name+'/utt2spk'
        with open(filename,'w') as file:
            for iter in range(len(utt_label)):
                file.write('%s %s\n' % (utt_label[iter], spk_label[iter]) )
    # if lang_label!=-1:
    if len(lang_label)!=0:
        filename = name+'/utt2lang'
        with open(filename,'w') as file:
            for iter in range(len(utt_label)):
                file.write('%s %s\n' % (utt_label[iter], lang_label[iter]) )


