import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from lxml import etree


def process_news(tag):
    head, text = tag[0].text, tag[1].text
    tokens = get_tokens(text)
    dictionary[head] = Counter(sorted(tokens, reverse=True))


def get_tokens(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in list(string.punctuation) + stopwords.words('english')]
    return tokens


lemmatizer = WordNetLemmatizer()
dictionary = {}
xml_path = 'news.xml'

tree = etree.parse(xml_path)
root = tree.getroot()
corpus = root[0]

for n in range(len(corpus)):
    process_news(corpus[n])

for key in dictionary:
    print(f'{key}:')
    print(*[n[0] for n in dictionary[key].most_common(5)])
    print()
