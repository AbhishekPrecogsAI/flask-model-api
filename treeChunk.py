import os
from typing import List, Dict
from tree_sitter import Parser
from tree_sitter_languages import get_parser, get_language


class CodeChunker:
    def __init__(self, languages: List[str]):
        """
        Initializes the CodeChunker with parsers for the specified languages.
        Args:
            languages (List[str]): A list of programming language names (e.g., 'python', 'javascript').
        """
        self.language_parsers = {}
        for lang in languages:
            print(lang)
            # language = get_language(lang)
            # if language is None:
            #     raise ValueError(f"Unsupported language: {lang}")
            parser = get_parser(lang)
            if parser is None:
                raise ValueError(f"Parser not available for language: {lang}")
            self.language_parsers[lang] = parser

    def parse_code(self, code: str, language_name: str):
        """
        Parses the provided source code using the specified language parser.
        """
        parser = self.language_parsers.get(language_name)
        if parser is None:
            raise ValueError(f"Parser not available for language: {language_name}")
        return parser.parse(bytes(code, "utf8"))

    def get_function_prefixes(self, language_name: str) -> List[str]:
        """
        Returns a list of expected function definition prefixes for a given language.
        """
        mapping = {
            'python': ['def ', 'async def '],
            'javascript': ['function ', 'async function '],
            'java': ['public ', 'protected ', 'private '],  # Note: Java method signatures are more complex
            'cpp': [],  # For C/C++, you might have to detect patterns based on return types and the '('
            'c': [],
            'go': ['func '],
            'ruby': ['def '],
            'php': ['function '],
            # Add more languages and patterns as needed
        }
        # Default to empty list if language is not explicitly handled
        return mapping.get(language_name, [])

    def extract_function_chunks(self, code: str, language_name: str, file_path: str) -> List[Dict]:
        """
        Extracts function definitions from the code along with their starting and ending line numbers, and file path.
        Uses language-specific expected prefixes to verify that the function chunk starts correctly.
        """
        tree = self.parse_code(code, language_name)
        root_node = tree.root_node
        function_chunks = []
        prefixes = self.get_function_prefixes(language_name)

        # Traverse the syntax tree using a cursor
        cursor = root_node.walk()
        reached_end = False
        while not reached_end:
            node = cursor.node
            if node.type == 'function_definition':
                func_text = code[node.start_byte:node.end_byte]
                # Check if the function chunk starts with any of the expected prefixes
                if prefixes and not any(func_text.lstrip().startswith(prefix) for prefix in prefixes):
                    # Backtrack to the nearest newline before node.start_byte to try to recover missing characters
                    start_index = node.start_byte
                    while start_index > 0 and code[start_index - 1] not in ['\n', '\r']:
                        start_index -= 1
                    func_text = code[start_index:node.end_byte]
                start_line = node.start_point[0] + 1  # 1-based line number
                end_line = node.end_point[0] + 1
                function_chunks.append({
                    'function_code': func_text,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': file_path,
                    'lang': language_name
                })
            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue
            while True:
                if not cursor.goto_parent():
                    reached_end = True
                    break
                if cursor.goto_next_sibling():
                    break

        return function_chunks

    def detect_language(self, file_name: str) -> str:
        """
        Detects the programming language based on the file extension.
        """
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.scala': 'scala',
            '.sh': 'bash',
            '.pl': 'perl',
            '.r': 'r',
            '.sql': 'sql',
            '.lua': 'lua',
            '.h': 'c',
            '.m': 'objective_c',
            '.clj': 'clojure',
            '.el': 'emacs_lisp',
            '.lisp': 'lisp',
            '.ml': 'ocaml',
            '.mli': 'ocaml',
            '.v': 'v',
            '.nim': 'nim',
            '.zig': 'zig',
            '.dart': 'dart',
            '.erl': 'erlang',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.coffee': 'coffee_script',
            '.scss': 'scss',
            '.less': 'less',
            '.styl': 'stylus',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'text',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.asciidoc': 'asciidoc',
            '.org': 'org',
            '.tex': 'latex',
            '.bib': 'bibtex',
            '.xml': 'xml'
        }
        ext = os.path.splitext(file_name)[1]
        return extension_map.get(ext, 'unknown')

    import os
    from typing import List, Dict

    def chunk_codebase(self, directory: str, file_extensions: List[str], exclude_dirs: List[str] = None) -> List[Dict]:
        """
        Processes all code files in the specified directory (recursively) that match one of the file extensions,
        extracts function chunks, and preserves file paths along with start and end line numbers.

        Args:
            directory (str): The root directory to start the codebase traversal.
            file_extensions (List[str]): A list of file extensions to include in the processing.
            exclude_dirs (List[str], optional): A list of directory names to exclude from traversal. Defaults to None.

        Returns:
            List[Dict]: A list of dictionaries containing function chunk information.
        """
        if exclude_dirs is None:
            exclude_dirs = []

        all_function_chunks = []
        for root, dirs, files in os.walk(directory):
            # Modify dirs in-place to exclude specified directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    language_name = self.detect_language(file)
                    if language_name == 'unknown':
                        print(f"Skipping unknown language file: {file_path}")
                        continue

                    chunks = self.extract_function_chunks(code, language_name, file_path)
                    all_function_chunks.extend(chunks)

        return all_function_chunks


# Example usage:
if __name__ == "__main__":
    # Define the languages you wish to support.
    supported_languages = ['python', 'javascript', 'java', 'cpp', 'c', 'go', 'ruby', 'php', 'html', 'css']
    chunker = CodeChunker(supported_languages)

    # Specify the directory and list of file extensions to process.
    directory = "./"  # Update this to your codebase directory
    file_extensions = [".py", ".html"]  # Process both Python and JavaScript files, for example

    all_chunks = chunker.chunk_codebase(directory, file_extensions)
    for chunk_info in all_chunks:
        print(f"File: {chunk_info['file_path']}")
        print(f"Function starts at line {chunk_info['start_line']} and ends at line {chunk_info['end_line']}")
        print("Function chunk:")
        print(chunk_info['function_code'])
        print("-" * 40)
