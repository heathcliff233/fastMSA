import pickle
import os
import sys
import re
import math
import numpy as np
import esm
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import sys 
sys.path.append("/share/hongliang") 
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from model import MyEncoder
from data import QueryDataset, EbdDataset, SingleConverter
from myutils import get_filename
import faiss
import streamlit as st

import pandas as pd
import phylopandas.phylopandas as ph

BATCHSZ=1
search_batch = 10
#fasta_path = "/ssdcache/wangsheng/databases/uniref90/uniref90.fasta"
fasta_path = "./test.a3m"
ctx_dir = "./ur90_ebd/"
tmp_path = "./tmp_retrieve/"
download_path = "./download_it/"
upload_path = "./upload_it/"
expand_seq = "./expand_seq/"
qjackhmmer = "/share/wangsheng/GitBucket/alphafold2_sheng/alphafold2/util/qjackhmmer"

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
    prev = torch.load('../split_train/39.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    batch_converter = SingleConverter(alphabet)

    return model, batch_converter

@st.cache(allow_output_mutation=True)
def gen_ctx_ebd():
    ctx_list = os.listdir(ctx_dir)
    ctx_list.sort()
    df = ph.read_fasta(fasta_path, use_uids=False)
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
            print('\r indexed %-5d/%d'%(cnt, len(df)), end="")
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
    for fp in files:
        pref = fp[:-6]
        args = " -B "+ download_path+ "%s.a3m -E 0.001 --cpu 8 -N 1 "%pref+expand_seq+pref+".fasta"+" "+tmp_path+"%s.fasta | grep -E \'New targets included:|Target sequences:\'"%pref
        cmd = qjackhmmer+args
        os.system(cmd)
        finish_list.append(download_path+"%s.a3m"%pref)
        cnt += 1
    return finish_list

def gen_query(upload_file_path):
    df = ph.read_fasta(upload_file_path, use_uids=False)
    tot_num = len(df)
    for i in range(tot_num):
        seq_slice = df.iloc[i]
        filename = seq_slice.id
        seq_slice.phylo.to_fasta(expand_seq+filename+'.fasta', id_col='id')


st.title("Retriever-demo-v2")
st.markdown(f'Please upload one sequence in one fasta file end with .fasta/.seq')
tar_num = st.selectbox(
    "Target num: ",
    [32, 128, 250, 2048, 20000, 100000, 1000000]
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
uploaded = st.file_uploader("Upload", ['fasta', 'seq'])
if uploaded is not None:
    fpath = os.path.join(upload_path,uploaded.name)
    with open(fpath, "wb") as f:
        f.write(uploaded.getbuffer())
    #dataset = EbdDataset(seq_path)
    qs = os.listdir(expand_seq)
    qs.sort()
    dataset = QueryDataset(qs)
    gen_query(fpath)
    dataset = EbdDataset(fpath)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    tot_tar = encoded.shape[0]
    for i in range(math.ceil(tot_tar/search_batch)):
        scores, idxes = index.search(encoded[i*search_batch:(i+1)*search_batch].reshape((1,-1)), tar_num)
        idx_batch = len(idxes)
        for j in range(idx_batch):
            tar_idx = idxes[j]
            sp = df.iloc[tar_idx]
            sp.loc[:,'sequence'] = sp['sequence'].map(lambda x: re.sub('[(\-)]', '', x))
            sp.phylo.to_fasta(tmp_path+dataset.records[i*search_batch+j].id+".fasta", id_col='id')
    st.markdown(f'Start alignment')
    download_list = my_aligner()
    st.markdown(f'Finished')
    #st.markdown(get_binary_file_downloader_html(download_path+'1a04A01.a3m'), unsafe_allow_html=True)
    for i in range(len(download_list)):
        st.markdown(get_binary_file_downloader_html(download_list[i]), unsafe_allow_html=True)

