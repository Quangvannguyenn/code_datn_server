import requests
import pandas as pd
import json
from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go

device = 'cpu'
MAX_LEN = 125

st.cache_data.clear()
st.cache_resource.clear()

tokenizer = AutoTokenizer.from_pretrained("quanqnv19/VN-Sentiment-Classification")
model = RobertaForSequenceClassification.from_pretrained("quanqnv19/VN-Sentiment-Classification")

def get_comment(comment):
        return {
        # 'uid': comment['from']['id'],
        # 'name': comment['from']['name'],
        'time': comment['created_time'],
        'message': comment['message'], 
        'cmt_id': comment['id']
    }

def get_post(data):
    return {
        'id_post': data['id'],
        'time': data['created_time'],
        'message': data['message'], 
    }

def get_comments(post_id, long_term_accessToken):
    url = f'https://graph.facebook.com/v19.0/{post_id}/comments?access_token={long_term_accessToken}'
    response = requests.request("GET", url)
    data = json.loads(response.text)
    excel_data = list(map(get_comment, data['data']))
    df = pd.DataFrame(excel_data)
    # df.to_excel('comments.xlsx', index=False)
    return excel_data

def get_feed(long_term_accessToken):
    url = f'https://graph.facebook.com/v19.0/me/feed?access_token={long_term_accessToken}'
    response = requests.request("GET", url)
    data = json.loads(response.text)
    feed_data = list(map(get_post, data['data']))
    return feed_data

def get_comments_in_post(data_post):
    comments = []
    for item in data_post:
        comments.append(item['message'])
    return comments

def display_data(data_post, post):
    labels = ['Tích cực','Tiêu cực', 'Trung lập']
    comments = get_comments_in_post(data_post)
    pos, nev, neu = process_data(comments)
    print(pos+nev+neu)
    values = [pos, nev, neu]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', insidetextorientation='radial')])
    fig.update_layout(height=400, width=400)
    container = st.container(border=True)
    lef_col, right_col = container.columns(2)
    with lef_col:
        st.subheader("Id bài viết: ")
        st.write(post['id_post'])
        st.markdown("---")
        st.subheader("Nội dung:")
        st.write(post['message'])
        st.markdown("---")
        st.subheader("Kết quả phân loại bình luận:")
        st.write("Tiêu cực: "+str(nev))
        st.write("Tích cực: "+str(pos))
        st.write("Trung lập: "+str(neu))
    with right_col:
        st.plotly_chart(fig, use_container_width=True)
    print("ok")




def process_data(comments):
    neg, pos, neu = 0, 0, 0
    for sentence in comments:
        print(tokenizer.encode(sentence))
        input_ids = torch.tensor([tokenizer.encode(sentence)])
        with torch.no_grad():
            out = model(input_ids)
            print(out.logits.softmax(dim=-1).tolist())
            res = np.argmax(out[0], axis=1).flatten()[0].item()
            if res == 0:
                neg = neg + 1
            elif neg == 1:
                pos = pos + 1
            else:
                neu = neu + 1
    return pos, neg, neu


if __name__ == '__main__':
    file = st.file_uploader("Upload Page access token file: ")
    if file:
        for token in file:
            long_term_accessToken = token.decode("utf-8")
        feeds = get_feed(long_term_accessToken)
        for post in feeds:
            data_comments = get_comments(post['id_post'], long_term_accessToken)
            display_data(data_comments, post)

