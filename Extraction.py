### This module should read the input to the model and perform entity extraction to determine a category ###
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

### One time Downloads ####
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        return: cleaned string
        warning: removing all nltk stop words is almost certainly a bad idea in this use case.
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
# df['questionText'] = df['questionText'].apply(clean_text)
# print_plot(10)


def extraction(text):
    """
        text: a string
        return: the original string tokenized  --with part of speech tags--
    """
    tokens = nltk.word_tokenize(text)
#     posTags = nltk.pos_tag(tokens)
    return tokens


### Unit Test ####
text = "I am feeling overworked and depressed"
print(extraction(text))

# CC coordinating conjunction
# CD cardinal digit
# DT determiner
# EX existential there (like: “there is” … think of it like “there exists”)
# FW foreign word
# IN preposition/subordinating conjunction
# JJ adjective ‘big’
# JJR adjective, comparative ‘bigger’
# JJS adjective, superlative ‘biggest’
# LS list marker 1)
# MD modal could, will
# NN noun, singular ‘desk’
# NNS noun plural ‘desks’
# NNP proper noun, singular ‘Harrison’
# NNPS proper noun, plural ‘Americans’
# PDT predeterminer ‘all the kids’
# POS possessive ending parent’s
# PRP personal pronoun I, he, she
# PRP$ possessive pronoun my, his, hers
# RB adverb very, silently,
# RBR adverb, comparative better
# RBS adverb, superlative best
# RP particle give up
# TO, to go ‘to’ the store.
# UH interjection, errrrrrrrm
# VB verb, base form take
# VBD verb, past tense, took
# VBG verb, gerund/present participle taking
# VBN verb, past participle is taken
# VBP verb, sing. present, known-3d take
# VBZ verb, 3rd person sing. present takes
# WDT wh-determiner which
# WP wh-pronoun who, what
# WP$ possessive wh-pronoun whose
# WRB wh-adverb where, when