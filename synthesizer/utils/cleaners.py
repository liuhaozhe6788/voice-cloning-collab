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


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regular expression

# List of (regular expression, replacement) pairs for abbreviations with ending '.':
_abbreviations_dot_tail = [(re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1]) for x in [
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
_abbreviations_char = [(re.compile(r"%s" % x[0], re.IGNORECASE), x[1]) for x in [
    ("(#\w+)", r'\1.'),  # split the hashtag word
    ("@", " at ")]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations_dot_tail:
        text = re.sub(regex, replacement, text)
    for regex, replacement in _abbreviations_char:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    """lowercase input tokens."""
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


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
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = add_breaks(text)
    text = split_conj(text) 
    text = collapse_whitespace(text)
    return text
