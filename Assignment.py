import requests
from bs4 import BeautifulSoup
import nltk
from nltk import sent_tokenize, word_tokenize
import syllables
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


nltk.download('punkt')

def load_words_from_file(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as file:
        words = [line.strip().lower() for line in file]
    return set(words)

def load_stopwords():
    stopwords = set()
    stopwords_files = [
        'StopWords_Auditor.txt',
        'StopWords_Currencies.txt',
        'StopWords_DatesandNumbers.txt',
        'StopWords_Generic.txt',
        'StopWords_GenericLong.txt',
        'StopWords_Geographic.txt',
        'StopWords_Names.txt'
    ]

    for file in stopwords_files:
        stopwords.update(load_words_from_file(file, encoding='ISO-8859-1'))

    return stopwords

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = ""

        for paragraph in soup.find_all('p'):
            article_text += paragraph.get_text() + "\n"

        return article_text
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def calculate_text_metrics(text, positive_words, negative_words, stopwords):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word.lower() not in stopwords]

    complex_word_count = sum(1 for word in cleaned_words if len(word) > 2 and syllables.estimate(word) > 2)
    syllable_count = sum(syllables.estimate(word) for word in cleaned_words)

    positive_score = sum(1 for word in cleaned_words if word.lower() in positive_words)
    negative_score = sum(1 for word in cleaned_words if word.lower() in negative_words)

    avg_sentence_length = len(cleaned_words) / len(sentences)
    percentage_complex_words = (complex_word_count / len(cleaned_words)) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(cleaned_words) / len(sentences)

    personal_pronouns = sum(1 for word in cleaned_words if re.match(r'\b(?:i|we|my|ours|us)\b', word, flags=re.IGNORECASE))
    avg_word_length = sum(len(word) for word in cleaned_words) / len(cleaned_words)

    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001),
        'SUBJECTIVITY SCORE': (positive_score + negative_score) / (len(cleaned_words) + 0.000001),
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': len(cleaned_words),
        'SYLLABLE PER WORD': syllable_count / len(cleaned_words),
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

input_data = pd.read_excel('Input.xlsx')

positive_words = load_words_from_file('positive-words.txt', encoding='utf-8')
negative_words = load_words_from_file('negative-words.txt', encoding='ISO-8859-1')

stopwords = load_stopwords()

output_data = []
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    article_text = get_text_from_url(url)

    if article_text:
        text_metrics = calculate_text_metrics(article_text, positive_words, negative_words, stopwords)

        output_data.append({
            'URL_ID': url_id,
            'URL': url,
            **text_metrics
        })

output_df = pd.DataFrame(output_data)

output_df.to_excel('Output.xlsx', index=False)
