import sys 
sys.path.append("/share/hongliang") 
import numpy as np
import pandas as pd
import phylopandas.phylopandas as ph
import streamlit as st

@st.cache
def load_data():
    df = ph.read_fasta('/ssdcache/wangsheng/databases/uniref90/uniref90.fasta', use_uids=False)
    #df = ph.read_fasta('./hmmalign256-v1/1a04A01.a3m', use_uids=False)
    return df

st.title("my first app")
df = load_data()
st.table(df.head())

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)



