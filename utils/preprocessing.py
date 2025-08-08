import re

def clean_url(url):
    return re.sub(r'https?://|www\.', '', url).strip().lower()

def prepare_text_features(df):
    return (df['url'].fillna('') + ' ' + df['title'].fillna('')).apply(clean_url)
