import os
from typing import List, Dict
from tree_sitter import Parser
from tree_sitter_languages import get_parser, get_language

import os
from typing import List, Dict, Optional
from tree_sitter import Parser
from tree_sitter_languages import get_parser
import logging

import os
import hashlib
from typing import List, Dict, Optional
from tree_sitter import Parser
from tree_sitter_languages import get_parser
import logging


import os
import hashlib
from typing import List, Dict, Optional
from tree_sitter import Parser
from tree_sitter_languages import get_parser
import logging


class CodeChunker:
    def __init__(self, languages: List[str]):
        self.language_parsers = {}
        for lang in languages:
            try:
                parser = get_parser(lang)
                if parser is None:
                    raise ValueError(f"Parser not available for language: {lang}")
                self.language_parsers[lang] = parser
            except Exception as e:
                logging.warning(f"Failed to initialize parser for {lang}: {e}")

    def parse_code(self, code: str, language_name: str):
        parser = self.language_parsers.get(language_name)
        if parser is None:
            raise ValueError(f"Parser not available for language: {language_name}")
        return parser.parse(bytes(code, "utf8"))

    def detect_language(self, file_name: str) -> str:
        extension_map = {
            '.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
            '.rb': 'ruby', '.php': 'php', '.html': 'html', '.css': 'css', '.ts': 'typescript', '.tsx': 'typescript',
            '.jsx': 'javascript', '.swift': 'swift', '.kt': 'kotlin', '.rs': 'rust', '.scala': 'scala',
            '.sh': 'bash', '.pl': 'perl', '.r': 'r', '.sql': 'sql', '.lua': 'lua', '.h': 'c',
        }
        ext = os.path.splitext(file_name)[1]
        return extension_map.get(ext, 'unknown')

    def extract_chunks(self, code: str, language_name: str, file_path: str) -> List[Dict]:

        # code_bytes = code.encode('utf8')

        try:
            tree = self.parse_code(code, language_name)
        except Exception as e:
            logging.error(f"Failed to parse code for {file_path}: {e}")
            return self.fallback_semantic_chunks(code, file_path, language_name)

        root_node = tree.root_node
        node_types_by_lang = {
            'python': ['function_definition', 'class_definition', 'lambda'],
            'javascript': ['function', 'method_definition', 'class_declaration', 'lexical_declaration',
                           'variable_declarator', 'arrow_function', 'function_expression'],
            'java': ['method_declaration', 'class_declaration', 'field_declaration', 'interface_declaration'],
            'cpp': ['function_definition', 'class_specifier', 'declaration', 'struct_specifier', 'enum_specifier',
                    'namespace_definition', 'template_declaration', 'preproc_def', 'preproc_if', 'preproc_include',
                    'lambda_expression'],
            'c': ['function_definition', 'declaration', 'struct_specifier', 'enum_specifier', 'preproc_def'],
            'go': ['function_declaration', 'method_declaration', 'type_declaration', 'function_literal'],
            'php': ['function_definition', 'method_declaration', 'class_declaration'],
            'ruby': ['method', 'class'],
            'rust': ['function_item', 'struct_item', 'enum_item', 'trait_item', 'impl_item', 'mod_item',
                     'macro_definition', 'use_declaration', 'const_item', 'closure_expression']
        }
        target_node_types = set(node_types_by_lang.get(language_name, ['function_definition']))

        chunks = []
        seen_ranges = []

        def hash_chunk(text: str) -> str:
            return hashlib.sha256(text.encode('utf-8')).hexdigest()



        def is_duplicate_range(start: int, end: int) -> bool:
            return any(start >= s and end <= e for s, e in seen_ranges)



        def is_trivial_function(node) -> bool:
            return (node.end_point[0] - node.start_point[0]) < 2

        def traverse(node, parent_type=None):
            if node.type in target_node_types:
                start_byte, end_byte = node.start_byte, node.end_byte
                if not is_duplicate_range(start_byte, end_byte):
                    chunk_text = code[start_byte:end_byte]
                    chunk_text = node.text.decode("utf‑8")
                    if node.type == 'variable_declarator' and len(chunk_text.strip().splitlines()) < 2 and len(
                            chunk_text.strip()) < 30:
                        return

                    if parent_type in target_node_types and is_trivial_function(node):
                        return

                    chunks.append({
                        'function_code': chunk_text,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'file_path': file_path,
                        'lang': language_name,
                        'hash': hash_chunk(chunk_text)
                    })
                    seen_ranges.append((start_byte, end_byte))
                    return

            for child in node.children:
                traverse(child, node.type)

        traverse(root_node)

        if not chunks:
            return self.fallback_semantic_chunks(code, file_path, language_name)

        chunks.sort(key=lambda c: (c["start_line"], c["end_line"]))

        merged: List[Dict] = []
        cur = chunks[0]
        for nxt in chunks[1:]:
            # both chunks are fallback (gap) blocks?
            both_gap = cur.get("is_gap") and nxt.get("is_gap")
            same_file = cur["file_path"] == nxt["file_path"]

            # merge only if the very next line continues the code
            directly_adjacent = nxt["start_line"] == cur["end_line"] + 1

            if same_file and both_gap and directly_adjacent:
                cur["function_code"] += "\n" + nxt["function_code"]
                cur["end_line"] = nxt["end_line"]
                cur["hash"] = hashlib.sha256(cur["function_code"].encode()).hexdigest()
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    def fallback_semantic_chunks(self, code: str, file_path: str, language_name: str) -> List[Dict]:
        lines = code.splitlines()
        chunks = []
        buffer = []
        current_start = 1

        thresholds_by_language = {
            'python': 40, 'javascript': 40, 'java': 50, 'cpp': 60, 'c': 60, 'go': 50,
            'php': 40, 'ruby': 30, 'html': 80, 'sql': 100
        }
        threshold_lines = thresholds_by_language.get(language_name, 50)

        indent_levels = []

        def flush_buffer():
            nonlocal buffer, current_start
            if buffer:
                chunk_text = "\n".join(buffer)
                chunks.append({
                    'function_code': chunk_text,
                    'start_line': current_start,
                    'end_line': current_start + len(buffer) - 1,
                    'file_path': file_path,
                    'lang': language_name,
                    'hash': hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                })
                current_start += len(buffer)
                buffer.clear()
                indent_levels.clear()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            indent_levels.append(indent)
            buffer.append(line)
            if len(buffer) >= threshold_lines or (len(indent_levels) >= 2 and indent_levels[-1] < indent_levels[-2]):
                flush_buffer()

        flush_buffer()
        return chunks

    def chunk_codebase(self, directory: str, file_extensions: List[str], exclude_dirs: Optional[List[str]] = None) -> List[Dict]:
        if exclude_dirs is None:
            exclude_dirs = []

        all_chunks = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                    except Exception as e:
                        logging.warning(f"Error reading {file_path}: {e}")
                        continue

                    language_name = self.detect_language(file)
                    if language_name == 'unknown':
                        logging.info(f"Skipping unknown language file: {file_path}")
                        continue

                    chunks = self.extract_chunks(code, language_name, file_path)
                    all_chunks.extend(chunks)

        return all_chunks

    def get_supported_languages(self) -> List[str]:
        return list(self.language_parsers.keys())

    def is_supported_file(self, file_name: str, file_extensions: List[str]) -> bool:
        return any(file_name.endswith(ext) for ext in file_extensions)

if __name__ == "__main__":
    supported_languages = ['python', 'javascript', 'java', 'cpp', 'c', 'go', 'ruby', 'php', 'html', 'css', 'rust']
    chunker = CodeChunker(supported_languages)

    # Specify the directory and list of file extensions to process.
    directory = "./data/"  # Update this to your codebase directory
    file_extensions = [".js"]  # Process both Python and JavaScript files, for example

    all_chunks = chunker.chunk_codebase(directory, file_extensions)

    for chunk_info in all_chunks:

            # Optionally, print chunk information for debugging
        print(f"File: {chunk_info['file_path']}")
        print(f"Function starts at line {chunk_info['start_line']} and ends at line {chunk_info['end_line']}")
        print("Function chunk:")
        print(chunk_info['function_code'])
        print("-" * 40)


