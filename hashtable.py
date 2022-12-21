import re
def splitword(text):
    wordtable=['at','on','in','during','for','before','after','since','until',
        'between','under','over','above','below','by','beside','near','next to','outside','inside',
        'behind','with','through','about']
    a='\\b('+"|".join(wordtable)+')\\b'
    b=re.sub(a,r". \1",text)

    return b