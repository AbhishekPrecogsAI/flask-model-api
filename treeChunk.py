from tree_sitter import Parser
from tree_sitter_languages import get_parser, get_language

class CodeChunker:
    def __init__(self, language_name: str):
        """
        Initializes the CodeChunker with the specified language parser.
        Args:
            language_name (str): The name of the programming language (e.g., 'python', 'javascript').
        """
        self.language = get_language(language_name)
        if self.language is None:
            raise ValueError(f"Unsupported language: {language_name}")
        self.parser = get_parser(language_name)
        if self.parser is None:
            raise ValueError(f"Parser not available for language: {language_name}")

    def parse_code(self, code: str):
        """
        Parses the provided source code and returns the syntax tree.
        Args:
            code (str): The source code to parse.
        Returns:
            Tree: The syntax tree resulting from parsing the code.
        """
        return self.parser.parse(bytes(code, "utf8"))

    def extract_function_chunks(self, code: str):
        """
        Extracts code chunks based on function definitions.
        Args:
            code (str): The source code to analyze.
        Returns:
            List[str]: A list of code chunks representing functions.
        """
        tree = self.parse_code(code)
        root_node = tree.root_node
        function_chunks = []

        for child in root_node.children:
            if child.type == 'function_definition':
                function_code = code[child.start_byte:child.end_byte]
                function_chunks.append(function_code)

        return function_chunks

# Example usage:
if __name__ == "__main__":
    with open('rag.py', 'r', encoding='utf-8') as f:
        code = f.read()

    chunker = CodeChunker(language_name='python')
    function_chunks = chunker.extract_function_chunks(code)
    for i, chunk in enumerate(function_chunks):
        print(f"Function {i + 1}:\n{chunk}\n")
