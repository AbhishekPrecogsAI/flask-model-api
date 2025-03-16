from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

def detect_language(code: str):
    try:
        lexer = guess_lexer(code)
        return lexer.name
    except ClassNotFound:
        return None