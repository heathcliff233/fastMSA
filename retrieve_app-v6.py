import pickle
import os
import sys
import re
import math
import numpy as np
import esm
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from model import MyEncoder
from data import QueryDataset, EbdDataset, SingleConverter
from myutils import get_filename
import faiss
import streamlit as st
import sys 
sys.path.append("/share/hongliang") 
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5,7"
import pandas as pd
import phylopandas.phylopandas as ph

BATCHSZ=1
search_batch = 10
#gtmsadir = "/ssdcache/wangsheng/train_test_data/CAMEO_RawData/cameo_msa/"
gtmsadir = "/ssdcache/wangsheng/train_test_data/CASP_RawData/allDM_msa/"
#gtmsadir = "./c1000_msa/"
#msadir = "./c1000_msa/" 
#fasta_path = "/ssdcache/zhengliangzhen/sequence_databases/uniref90_2019_07.fasta"
#fasta_path = "/ssdcache/zhengliangzhen/sequence_databases/uniref90_2020_03.fasta"
#fasta_path = "/share/hongliang/res-database.fasta"
#dm_path = "/share/hongliang/seq_db.fasta" 
#dm_path = "/share/hongliang/res-database.fasta"
#dm_path = "/ssdcache/wangsheng/databases/uniref90/uniref90.fasta"
dm_path = "/ssdcache/zhengliangzhen/hongliang/ur90_201803.fasta"
#fasta_path = "/ssdcache/wangsheng/databases/uniref90/uniref90.fasta"
#ctx_dir = "./random_ebd/"
ctx_dir = "./fseq_ebd_v2/"
tmp_path = "./v6-tmp/tmp_retrieve/"
download_path = "./v6-tmp/download_it/"
upload_path = "./v6-tmp/upload_it/"
expand_seq = "./v6-tmp/expand_seq/"
qjackhmmer = "/share/hongliang/qjackhmmer"

def qencode(model, loader, device="cuda:0"):
    model.eval()
    res = []
    for i, (ids, toks) in enumerate(loader):
        toks = toks.to(device)
        with torch.no_grad():
            out = model.forward_once(toks)
        out = out.cpu().numpy()
        res.extend(
            [out[i] for i in range(out.shape[0])]
        )
    return np.stack(res, axis=0)


@st.cache
def get_model():
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = MyEncoder(encoder, 0)
    prev = torch.load('./model_from_dgx/v2/99.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    batch_converter = SingleConverter(alphabet)

    return model, batch_converter

#@st.cache(allow_output_mutation=True)
#def get_raw_seq_database():
#    df = ph.read_fasta(fasta_path, use_uids=False)
#    #df = ph.read_fasta_dev(fasta_path)
#    return df


@st.cache(allow_output_mutation=True)
def gen_ctx_ebd():
    ctx_list = os.listdir(ctx_dir)
    ctx_list.sort()
    df = ph.read_fasta_dev(dm_path, use_uids=False)
    #df.loc[:,'sequence'] = df['sequence'].map(lambda x: re.sub('[(\-)]', '', x))
    print("Finish reading fasta")
    buffer = []
    index = faiss.IndexFlatIP(768)
    cnt = 0
    for i in ctx_list:
        with open(ctx_dir+i, "rb") as reader:
            whole_doc = pickle.load(reader)
            for name, tok in whole_doc:
                buffer.append(tok)
                cnt += 1
            index.add(np.stack(buffer, axis=0))
            buffer = []
            print('\r indexed %-7d/%d'%(cnt, len(df)), end="")
    print("ctx seq num: ", len(df))
    print("ebd seq num: ", cnt)

    return index, df


import base64
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {os.path.basename(bin_file)}</a>'
    return href


def my_aligner():
    files = os.listdir(tmp_path)
    files.sort()
    #src_seq = os.listdir(expand_seq)
    finish_list = []
    cnt = 0
    total_files = len(files)
    align_bar = st.progress(0)
    for fp in files:
        pref = fp[:-6]
        args = " -B "+ download_path+ "%s.a3m -E 0.001 --cpu 8 -N 3 "%pref+expand_seq+pref+".fasta"+" "+tmp_path+"%s.fasta | grep -E \'New targets included:|Target sequences:\'"%pref
        cmd = qjackhmmer+args
        os.system(cmd)
        finish_list.append(download_path+"%s.a3m"%pref)
        cnt += 1
        align_bar.progress(cnt/total_files)
    return finish_list

def gen_query(upload_file_path):
    df = ph.read_fasta(upload_file_path, use_uids=False)
    tot_num = len(df)
    for i in range(tot_num):
        seq_slice = df.iloc[i]
        filename = seq_slice.id
        seq_slice.phylo.to_fasta(expand_seq+filename+'.fasta', id_col='id')


st.title("Retriever-demo-v6")
st.markdown(f'Please upload one sequence in one fasta file end with .fasta/.seq')
tar_num = st.selectbox(
    "Target num: ",
    [128, 2048, 20000, 100000, 200000, 400000, 500000, 600000, 800000, 1000000, 2000000, 5000000]
)

for f in os.listdir(tmp_path):
    os.remove(os.path.join(tmp_path, f))
for f in os.listdir(download_path):
    os.remove(os.path.join(download_path, f))
for f in os.listdir(upload_path):
    os.remove(os.path.join(upload_path, f))
for f in os.listdir(expand_seq):
    os.remove(os.path.join(expand_seq, f))

model, batch_converter = get_model()
device = torch.device("cuda:0")
model = model.to(device)
index, df = gen_ctx_ebd()
#seq_database = get_raw_seq_database()
#=== remove duplicates in ctx
#ori_idx = df['id'].map(lambda x: x.split('/')[0])
#ori_idx = ori_idx.drop_duplicates(keep='first')
#oseq = seq_database[seq_database['id'].isin(ori_idx)]
#oseq.phylo.to_fasta_dev('./res-database.fasta')
#st.markdown("ctx has %d / %d / %d and ur90 has %d"%(oseq.shape[0], ori_idx.shape[0], df.shape[0], seq_database.shape[0]))
#===
uploaded = st.file_uploader("Upload", ['fasta', 'seq'])
if uploaded is not None:
    fpath = os.path.join(upload_path,uploaded.name)
    with open(fpath, "wb") as f:
        f.write(uploaded.getbuffer())
    gen_query(fpath)
    qs = os.listdir(expand_seq)
    qs.sort()
    qr = [expand_seq+rc for rc in qs]
    dataset = QueryDataset(qr)
    #gen_query(fpath)
    #dataset = EbdDataset(fpath)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    tot_tar = encoded.shape[0]
    st.markdown(f'Finish encoding, searching indexes...')
    my_bar = st.progress(0)
    tot_recall_rate = 0
    for i in range(math.ceil(tot_tar/search_batch)):
        scores, idxes = index.search(encoded[i*search_batch:(i+1)*search_batch], tar_num)
        idx_batch = len(idxes)
        for j in range(idx_batch):
            tar_idx = idxes[j]
            sp = df.iloc[tar_idx]
            #remove duplicates
            #sp = sp.drop_duplicates(subset=['sequence'], keep='first')
            #sp = ph.read_fasta(gtmsadir+qs[i*search_batch+j][:-6]+'.a3m')
            #sp = sp.drop_duplicates(subset=['id'], keep='first')
            #st.markdown("contain seq %d"%sp.shape[0])
            #########################################
            # Retrieve raw sequence from the UR90 database
            #sel_ids = sp['id'].map(lambda x: x.split('/')[0])
            #sel_ids = sel_ids.drop_duplicates()
            #sel_ids = sp['id'].drop_duplicates()
            #raw_seq = seq_database[seq_database['id'].isin(sel_ids)]
            #st.markdown("raw seq num %d"%raw_seq.shape[0])
            #########################################
            #====recall rate calculation for casp
            #gt = ph.read_fasta(gtmsadir+qs[i*search_batch+j][:-6]+'.a3m')
            #gt['id'] = gt['id'].map(lambda x: x.split('/')[0])
            #==
            #gt = gt[gt['id'].isin(ori_idx)]
            #==
            #gt = gt.drop_duplicates(subset=['id'], keep='first')
            #num_rt = raw_seq.shape[0]
            #num_gt = gt.shape[0]
            #num_cb = pd.concat([gt, raw_seq], axis=0).drop_duplicates(subset=['id'], keep='first').shape[0]
            #if num_gt==0:
            #    num_gt+=1
            #rc_rate = (num_rt + num_gt - num_cb) / num_gt
            #tot_recall_rate += rc_rate
            #st.markdown("rc %d / %d"%((num_rt+num_gt-num_cb), num_gt))
            
            #====get gt, calculate recall rate
            #gt = ph.read_fasta(gtmsadir+qs[i*search_batch+j][:-6]+'.a3m')
            #gt['id'] = gt['id'].map(lambda x: x.split('/')[0])
            #gt_seq = df[df['id'].isin(gt['id'])]
            #gt_seq = gt_seq.drop_duplicates(subset=['id'], keep='first')
            #num_rt = sp.shape[0]
            #num_gt = gt_seq.shape[0]
            #num_cb = pd.concat([gt_seq, sp], axis=0).drop_duplicates(subset=['id'], keep='first').shape[0]
            #rc_rate = 1 if num_gt==0 else (num_rt + num_gt - num_cb) / num_gt
            #tot_recall_rate += rc_rate
            #st.markdown("rc %d / %d"%((num_rt+num_gt-num_cb), num_gt))
            #########################################
            #raw_seq.phylo.to_fasta_dev(tmp_path+dataset.records[i*search_batch+j].id+".fasta")
            #raw_seq.phylo.to_fasta(tmp_path+dataset.records[i*search_batch+j].id+".fasta", id_col='id')
            #sp.phylo.to_fasta_dev(tmp_path+dataset.records[i*search_batch+j].id+".fasta")
            my_bar.progress((i*search_batch+j+1)/tot_tar)
    #out_str = "Recall rate = %.2f%%" % (tot_recall_rate/tot_tar*100)
    #st.markdown(out_str)
    #st.markdown(f'Start alignment')
    #download_list = my_aligner()
    st.markdown(f'Finished')
    #os.system("cp -r ./v6-tmp/tmp_retrieve /share/hongliang/casp14-40-ur9018-db")
    #os.system("cp -r ./v6-tmp/download_it /share/hongliang/casp14-10-ur9018-res")
    
    #st.markdown(get_binary_file_downloader_html(download_path+'1a04A01.a3m'), unsafe_allow_html=True)
    #for i in range(len(download_list)):
    #    st.markdown(get_binary_file_downloader_html(download_list[i]), unsafe_allow_html=True)

