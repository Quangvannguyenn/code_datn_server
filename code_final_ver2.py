import streamlit as st
import googleapiclient.discovery
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import plotly.graph_objects as go

tokenizer = AutoTokenizer.from_pretrained("quanqnv19/hsd-model")
model = AutoModelForSequenceClassification.from_pretrained("quanqnv19/hsd-model")
device = 'cpu'




def encoder_generator(sentences):
    input_ids = []
    attention_masks =[]
    encoded_dict = tokenizer.encode_plus(sentences,
                                             add_special_tokens=True,
                                             max_length=20,
                                             pad_to_max_length=True,
                                             truncation = True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)
    return input_ids,attention_masks

def get_data_comment(input_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyDbt-xdAOjDhJghQGVMxfbsSiSyCFJr1Jw"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=input_id,
        maxResults=1000
    )
    comments = []
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
            public
        ])
    while (1 == 1):
        try:
            nextPageToken = response['nextPageToken']
        except KeyError:
            break
        nextPageToken = response['nextPageToken']
        nextRequest = youtube.commentThreads().list(part="snippet", videoId="-GJgqIJsTME", maxResults=1000, pageToken=nextPageToken)
        response = nextRequest.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                public
            ])
    df = pd.DataFrame(comments, columns=['Author', 'Updated at', 'Like count', 'Comment','public'])
    return df.drop('public', axis=1)

def process(data, list_clean, list_offensive, list_hate):
    for i in range(len(data)):
        input_ids, attention_masks = encoder_generator(data.iloc[i]['Comment'])
        in_data = TensorDataset(input_ids, attention_masks)
        in_data_loader = DataLoader(in_data, sampler=RandomSampler(in_data))
        with torch.no_grad():
            for batch in tqdm(in_data_loader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                embedded = model(b_input_ids,b_input_mask)[0]
                predictions = np.argmax(embedded[0]).flatten()[0].item()
                state = predictions
                if state == 0:
                    list_clean.append(data.iloc[i])
                elif state == 1:
                    list_offensive.append(data.iloc[i])
                else:
                    list_hate.append(data.iloc[i])


# process(get_data_comment('b2-NB7UUBcU'))

def display(data):
    list_clean = []
    list_offensive = []
    list_hate = []
    labels = ['CLEAN - Sạch','OFFENSIVE - Tiêu cực', 'HATE - Xúc phạm']
    process(data, list_clean, list_offensive, list_hate)
    values = [len(list_clean), len(list_offensive), len(list_hate)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', insidetextorientation='radial')])
    fig.update_layout(height=400, width=400)
    container = st.container(border=True)
    lef_col, right_col = container.columns(2)
    with lef_col:
        st.subheader("Kết quả phân loại bình luận:")
        st.write("CLEAN: "+str(len(list_clean)))
        st.download_button(
            "Tệp bình luận CLEAN",
            pd.DataFrame(list_clean).to_csv(index=False).encode('utf-8'),
            "file.csv",
            "text/csv",
            key="1"
        )
        st.write("OFFENSIVE: "+str(len(list_offensive)))
        st.download_button(
            "Tệp bình luận OFFENSIVE",
            pd.DataFrame(list_offensive).to_csv(index=False).encode('utf-8'),
            "file.csv",
            "text/csv",
            key="2"
        )
        st.write("HATE: "+str(len(list_hate)))
        st.download_button(
            "Tệp bình luận HATE",
            pd.DataFrame(list_hate).to_csv(index=False).encode('utf-8'),
            "file.csv",
            "text/csv",
            key="3"
        )
    with right_col:
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    st.markdown("<h2 style='text-align: center; color: #8C3819;'>Nhận dạng nội dung tiêu cực và xúc phạm PhoBERT-CNN</h2>", unsafe_allow_html=True)
    input_id = st.text_input("Nhập vào URL Video YouTube")
    if len(input_id)>2:
        id_video = input_id.split("v=")[1]
        data = get_data_comment(id_video)
        display(data)
        st.dataframe(data)
