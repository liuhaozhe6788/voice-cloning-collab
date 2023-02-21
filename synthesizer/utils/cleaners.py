"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You"ll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""
import re
from unidecode import unidecode
from synthesizer.utils.numbers import normalize_numbers

# http://www.speech.cs.cmu.edu/cgi-bin/cmudict
_alphabet2pronunciation = {
    'A': 'eiiy',
    'B': 'bee',
    'b': 'bee',
    'C': 'see',
    'c': 'see',  
    'D': 'dee',
    'd': 'dee',   
    'E': 'eee',
    'e': 'eee',
    'F': 'efph',
    'f': 'efph', 
    'G': 'jee',
    'g': 'jee',        
    'H': 'eiich',
    'h': 'eiich',   
    'J': 'jay',
    'j': 'jay', 
    'K': 'kay',
    'k': 'kay',  
    'L': 'el',
    'l': 'el', 
    'M': 'em',
    'm': 'em', 
    'N': 'en',
    'n': 'en', 
    'O': 'ow',
    'o': 'ow',
    'P': 'pee',
    'p': 'pee',
    'Q': 'kyuw',
    'q': 'kyuw',    
    'R': 'arr',
    'r': 'arr',   
    'S': 'es',
    's': 'es', 
    'T': 'tee',
    't': 'tee', 
    'U': 'you',
    'u': 'you', 
    'V': 'vee',
    'v': 'vee', 
    'W': 'dablyu',
    'w': 'dablyu', 
    'X': 'eks',
    'x': 'eks', 
    'Y': 'why',
    'y': 'why', 
    'Z': 'zee',
    'z': 'zee'
}

_abbreviations_lowercase = ["lol", "pov", "tbh", "omg"]

# Regular expression matching whitespace:
_whitespace_regex = re.compile(r"\s+")

# Regular expression
_abbreviations_lowercase_regex = re.compile(rf"\b({'|'.join(_abbreviations_lowercase)})\b")

_abbreviations_capital_regex = re.compile(r"\b([A-Z]{1,22})\b")

# List of (regular expression, replacement) pairs for abbreviations with ending '.':
_abbreviations_alphabet_regex = [(re.compile(rf"\b{x[0]}\b"), x[1]) for x in _alphabet2pronunciation.items()]

# List of (regular expression, replacement) pairs for abbreviations with ending '.':
_abbreviations_dot_tail_regex = [(re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1]) for x in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

# List of (regular expression, replacement) pairs for special char abbreviation:
_abbreviations_special_char_regex = [(re.compile(r"%s" % x[0], re.IGNORECASE), x[1]) for x in [
    ("#(\w+)", r'\1.'),  # split the hashtag word
    ("@", " at ")]]

def replace_special_char(text):
    # replace special characters
    for regex, replacement in _abbreviations_special_char_regex:
        text = re.sub(regex, replacement, text)
    return text

def letter2pronunciation(text):
    # uppercase some abbreviations that may not be uppercase
    text = re.sub(_abbreviations_lowercase_regex, lambda match: match.group(1).upper(), text)

    # split abbreviations consisting of <=22 capital letters to individual letters
    text = re.sub(_abbreviations_capital_regex, lambda match: ' '.join(match.group(1)), text)

    # convert alphabets to corresponding pronunciation
    for regex, replacement in _abbreviations_alphabet_regex:
        text = re.sub(regex, replacement, text)

    return text

def expand_abbreviations(text):
    # expand abbreviations ending with dot
    for regex, replacement in _abbreviations_dot_tail_regex:
        text = re.sub(regex, replacement, text)
    # expand other abbreviations 

    return text

def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    """lowercase input tokens."""
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_regex, " ", text)


def convert_to_ascii(text):
    return unidecode(text)

def split_conj(text):
    wordtable=['at','on','in','during','for','before','after','since','until',
        'between','under','above','below','by','beside','near','next to','outside','inside',
        'behind','with','through']
    a='\\b('+"|".join([' ' + i for i in wordtable])+')\\b'
    b=re.sub(a,r". \1",text)

    return b

def add_breaks(text):
    def convert(match_obj):
        if match_obj.group() is not None:
            return match_obj.group().replace(',', '')
    text = re.sub(r"[0-9]+[\,][0-9]+", convert, text)
    text = text.replace('-', ' ')
    text = text.replace(',', '.')
    text = text.replace(';', '.')
    text = text.replace('~', ' to ')
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = replace_special_char(text)
    text = letter2pronunciation(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = add_breaks(text)
    # text = split_conj(text) 
    text = collapse_whitespace(text)
    return text
