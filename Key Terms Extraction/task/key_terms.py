import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer


def process_news(tag):
    head, text = tag[0].text, tag[1].text
    tokens = get_tokens(text)
    dictionary[head] = {'tokens': tokens}


def get_tokens(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = get_lemmas(tokens)
    tokens = eliminate_meaningless(tokens)
    tokens = get_nouns(tokens)
    return tokens


def get_lemmas(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def eliminate_meaningless(tokens):
    return [token for token in tokens if token not in list(string.punctuation) + stopwords.words('english')]  # + ['ha', 'wa', 'u', 'a']]


def get_nouns(tokens):
    nouns = []
    for token in tokens:
        if nltk.pos_tag([token])[0][1] == 'NN':
            nouns.append(token)
    return nouns


lemmatizer = WordNetLemmatizer()
dictionary = {}
xml_path = 'news.xml'

tree = etree.parse(xml_path)
root = tree.getroot()
corpus = root[0]

for n in range(len(corpus)):
    process_news(corpus[n])

news_collection = [' '.join(dictionary[key]['tokens']) for key in dictionary]

vectorizer = TfidfVectorizer(tokenizer=get_tokens)
tfidf_matrix = vectorizer.fit_transform(news_collection)
terms = vectorizer.get_feature_names()

for n, key in enumerate(dictionary):
    dictionary[key]['weights'] = {}
    for word in dictionary[key]['tokens']:
        dictionary[key]['weights'][word] = tfidf_matrix[(n, terms.index(word))]

for key in dictionary:
    print(f'{key}:')
    res = [val[0] for val in sorted(dictionary[key]['weights'].items(), key=lambda x: (x[1], x[0]), reverse=True)]
    print(*res[:5])
    print()
