import nltk
from collections import Counter
from lxml import etree

dictionary = {}
xml_path = 'news.xml'

tree = etree.parse(xml_path)
root = tree.getroot()

for n in range(len(root[0])):
    head, text = root[0][n][0].text, root[0][n][1].text
    tokens = nltk.tokenize.word_tokenize(text.lower())
    dictionary[head] = Counter(sorted(tokens, reverse=True))

for key in dictionary:
    print(f'{key}:')
    print(*[n[0] for n in dictionary[key].most_common(5)])
    print()
