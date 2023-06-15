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
    'I': 'eye',
    'i': 'eye',
    'J': 'jay',
    'j': 'jay', 
    'K': 'kay',
    'k': 'kay',  
    'L': 'ell',
    'l': 'ell', 
    'M': 'emm',
    'm': 'emm', 
    'N': 'enn',
    'n': 'enn', 
    'O': 'oww',
    'o': 'oww',
    'P': 'pee',
    'p': 'pee',
    'Q': 'kyuw',
    'q': 'kyuw',    
    'R': 'arr',
    'r': 'arr',   
    'S': 'ess',
    's': 'ess', 
    'T': 'tee',
    't': 'tee', 
    'U': 'yyou',
    'u': 'yyou', 
    'V': 'wee',
    'v': 'wee', 
    'W': 'dablyu',
    'w': 'dablyu', 
    'X': 'ecks',
    'x': 'ecks', 
    'Y': 'why',
    'y': 'why', 
    'Z': 'zee',
    'z': 'zee'
}

_abbreviations_lowercase = ["lol", "pov", "tbh", "omg"]

# Regular expression matching whitespace:
_whitespace_regex = re.compile(r"\s+")

# Regular expression
_abbreviations_lowercase_regex = re.compile(rf"\b(?!')({'|'.join(_abbreviations_lowercase)})\b(?!')")

_abbreviations_capital_regex = re.compile(r"\b(?!')([A-Z0-9]*[A-Z][A-Z0-9]*)(?!')\b")  

_abbreviations_capital_plural_regex = re.compile(r"\b(?!')([A-Z0-9]*[A-Z][A-Z0-9]*s)(?!')\b")  

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
    ("@", " at "),
    ('~', ' to '),
    ('&', ' and '),
    ('%', ' percent '),
    ('\+', ' plus '),
    ('-', ' ')]]

# convert words that do not pronounce properly
_words_convert_regex = [(re.compile(rf"\b{x[0]}\b", flags=re.IGNORECASE), x[1]) for x in [
    ("etc", "et cetera"),
    ("guy", "guuy"),
    ("guys", "gize")
]]

def replace_special_char(text):
    # replace special characters
    for regex, replacement in _abbreviations_special_char_regex:
        text = re.sub(regex, replacement, text)
    return text

def letter2pronunciation(text):

    # uppercase some abbreviations that may not be uppercase
    text = re.sub(_abbreviations_lowercase_regex, lambda match: match.group(1).upper() + '.', text)

    def convert(match):
        char_list = [*match]
        if char_list[-1] is 's' and len(char_list) < 5:
            for idx in range(len(char_list)):
                if idx < len(char_list) - 1:
                    char_list[idx] = _alphabet2pronunciation.get(char_list[idx], char_list[idx])
                else:
                    char_list[idx - 1] += char_list[idx]
            return " ".join(char_list[:idx])
        elif len(char_list) < 4:
            char_list = map(lambda char: _alphabet2pronunciation.get(char, char), char_list)
            return " ".join(char_list) 
        else: return "".join(char_list)
    # split abbreviations consisting of one or more capital letters and zero or more numbers in single form to individual letters
    # and convert the letters to pronunciation
    text = re.sub(_abbreviations_capital_regex, lambda match: convert(match.group(1)), text)

    # split abbreviations consisting of one or more capital letters and zero or more numbers in plural form to individual letters
    # and convert the letters to pronunciation
    text = re.sub(_abbreviations_capital_plural_regex, lambda match: convert(match.group(1)), text)

    return text

def expand_abbreviations(text):
    # expand abbreviations ending with dot
    for regex, replacement in _abbreviations_dot_tail_regex:
        text = re.sub(regex, replacement, text)
    # expand other abbreviations 
    for regex, replacement in _words_convert_regex:
        text = re.sub(regex, replacement, text)
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
    text = re.sub(r"(\d{1,3}(,\d{3})+)\.?(\d+)?", lambda x: x.group(1).replace(",", "") + (("." + x.group(3)) if x.group(3) else ""), text)  # remove comma in numbers
    text = text.replace('-', ' ')
    text = text.replace(',', '. ')
    text = text.replace(';', '. ')
    text = text.replace(':', '. ')
    text = text.replace('!', '. ')
    text = text.replace('?', '. ')
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


def english_cleaners_predict(text):
    """Pipeline for English text, including number and abbreviation expansion for prediction."""
    text = convert_to_ascii(text)
    text = replace_special_char(text)
    text = expand_abbreviations(text)
    text = letter2pronunciation(text)
    text = lowercase(text)
    text = expand_numbers(text)
    # text = split_conj(text) 
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion for training preprocessing."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
