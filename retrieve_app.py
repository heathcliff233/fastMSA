import pickle
import os
import sys
import re
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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pandas as pd
import phylopandas.phylopandas as ph

BATCHSZ=1
#tar_num = 2048
path = "./split/split_dataset_test.txt"
qdir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_seq/"
msadir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
ctx_dir = "./split_ebd/"
#save_path = "./pred-202108232011-2048/"
tmp_path = "./tmp_retrieve/"
download_path = "./download_it/"
upload_path = "./upload_it/"
seq_path = "./1a04A01.seq"
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


def get_idx(idx, lst):
    cnt = 0
    tot = len(lst)
    while idx >= lst[cnt] :
        cnt += 1
        if cnt == tot :
            break
    cnt -= 1
    return cnt, idx-lst[cnt]


@st.cache
def get_model():
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = MyEncoder(encoder, 0)
    prev = torch.load('./split_train/39.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    batch_converter = SingleConverter(alphabet)

    return model, batch_converter

@st.cache(allow_output_mutation=True)
def gen_ctx_ebd():
    path_list = np.genfromtxt(path, dtype='str').T[0]
    path_list.sort()

    ctx_list = os.listdir('./split_ebd')
    ctx_list.sort()
    ttpath = './split/test.txt'
    file_list, lines = get_filename(ttpath)
    sorted_id = sorted(range(len(file_list)), key=lambda k: file_list[k])
    file_list.sort()
    src_list = [msadir+file for file in file_list]
    # Add phylopandas func
    df_list = [ph.read_fasta(src, use_uids=False) for src in src_list]
    df = pd.concat(df_list)
    lines = [lines[i] for i in sorted_id]
    pctg = [0]*len(lines)
    for i in range(1, len(lines)):
        pctg[i] = pctg[i-1] + lines[i-1]//2
    buffer = []
    index = faiss.IndexFlatIP(768)
    cnt = 0
    #tot_ctx = len(ctx_list)
    for i in ctx_list:
        with open('./split_ebd/'+i, "rb") as reader:
            whole_doc = pickle.load(reader)
            for name, tok in whole_doc:
                buffer.append(tok)
            index.add(np.stack(buffer, axis=0))
            #print("tot len", len(buffer))
            buffer = []
            cnt += 1
            #print('\r ', cnt, '/'+str(tot_ctx), file=sys.stderr, end="")
    #print("loaded ctx", file=sys.stderr)
    return index, df, pctg, lines


import base64
def get_binary_file_downloader_html(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {os.path.basename(bin_file)}</a>'
    return href


def my_aligner():
    files = os.listdir(tmp_path)
    files.sort()
    src_seq = os.listdir(upload_path)
    finish_list = []
    cnt = 0
    for fp in files:
        pref = fp.split('.')[0]
        args = " -B "+ download_path+ "%s.a3m -E 0.001 --cpu 8 -N 3 "%pref+upload_path+src_seq[cnt]+" "+tmp_path+"%s.fasta | grep -E \'New targets included:|Target sequences:\'"%pref
        cmd = qjackhmmer+args
        os.system(cmd)
        finish_list.append(download_path+"%s.a3m"%pref)
        cnt += 1
    return finish_list

st.title("Retriever-demo")
st.markdown(f'Please upload one sequence in one fasta file end with .fasta/.seq')
tar_num = st.selectbox(
    "Target num: ",
    [128, 2048, 20000, 100000, 200000]
)
for f in os.listdir(tmp_path):
    os.remove(os.path.join(tmp_path, f))
for f in os.listdir(download_path):
    os.remove(os.path.join(download_path, f))
for f in os.listdir(upload_path):
    os.remove(os.path.join(upload_path, f))
model, batch_converter = get_model()
device = torch.device("cuda:0")
model = model.to(device)
index, df, pctg, lines = gen_ctx_ebd()
uploaded = st.file_uploader("Upload", ['fasta', 'seq'])
if uploaded is not None:
    fpath = os.path.join(upload_path,uploaded.name)
    with open(fpath, "wb") as f:
        f.write(uploaded.getbuffer())
    #dataset = EbdDataset(seq_path)
    dataset = EbdDataset(fpath)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    tot_tar = encoded.shape[0]
    for i in range(tot_tar):
        scores, idxes = index.search(encoded[i].reshape((1,-1)), tar_num)
        tar_idx = idxes[0]
        sp = df.iloc[tar_idx]
        sp.loc[:,'sequence'] = sp['sequence'].map(lambda x: re.sub('[(\-)]', '', x))
        sp.phylo.to_fasta(tmp_path+dataset.records[i].id+".fasta", id_col='id')
    download_list = my_aligner()
    st.markdown(f'Finished')
    #st.markdown(get_binary_file_downloader_html(download_path+'1a04A01.a3m'), unsafe_allow_html=True)
    for i in range(len(download_list)):
        st.markdown(get_binary_file_downloader_html(download_list[i]), unsafe_allow_html=True)

