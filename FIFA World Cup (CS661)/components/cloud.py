from wordcloud import WordCloud
import os
import pandas as pd

def word_cloud(string,i):
    wordcloud_pic = f"./assets/images/wordcloud_{i}.png"
    wordcloud = WordCloud(collocations = False, background_color = 'white').generate(a)
    wordcloud.to_file(f'{wordcloud_pic}')     
    return wordcloud_pic

df = pd.read_csv("data/tournaments.csv")
a = " ".join(winner for winner in df["winner"])

word_cloud(a,1)