import unittest
from treeChunk import CodeChunker  # Assuming your file is named code_chunker.py


class TestCodeChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = CodeChunker(languages=['cpp'])
        self.test_file = "test.cpp"

    def test_template_class_chunking(self):
        code = """
        template<typename T>
        class Container {
        public:
            void add(T item) {
                // Method implementation
            }
        };
        """
        chunks = self.chunker.extract_chunks(code, 'cpp', self.test_file)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['type'], 'template_class')
        self.assertEqual(chunks[0]['start_line'], 2)
        self.assertEqual(chunks[0]['end_line'], 7)
        self.assertIn("template<typename T>", chunks[0]['code'])
        self.assertIn("class Container", chunks[0]['code'])

    def test_namespace_chunking(self):
        code = """
        namespace utils {
            void helper() {}
        }
        """
        chunks = self.chunker.extract_chunks(code, 'cpp', self.test_file)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['type'], 'namespace')
        self.assertEqual(chunks[0]['start_line'], 2)
        self.assertEqual(chunks[0]['end_line'], 4)
        self.assertIn("namespace utils", chunks[0]['code'])

    def test_combined_template_and_namespace(self):
        code = """
        template<typename T>
        class Outer {
        public:
            template<typename U>
            class Inner {
                U process(T input) {
                    return static_cast<U>(input);
                }
            };
        };

        namespace geometry {
            template<typename T>
            class Point {
                T x, y;
            };
        }
        """
        chunks = self.chunker.extract_chunks(code, 'cpp', self.test_file)

        # Should detect 3 chunks: Outer class, Inner class, and namespace
        self.assertEqual(len(chunks), 3)

        # Verify Outer template
        outer = chunks[0]
        self.assertEqual(outer['type'], 'template_class')
        self.assertIn("class Outer", outer['code'])

        # Verify Inner template
        inner = chunks[1]
        self.assertEqual(inner['type'], 'template_class')
        self.assertIn("class Inner", inner['code'])

        # Verify namespace
        ns = chunks[2]
        self.assertEqual(ns['type'], 'namespace')
        self.assertIn("namespace geometry", ns['code'])

    def test_method_detection(self):
        code = """
        class Calculator {
        public:
            template<typename T>
            T add(T a, T b) {
                return a + b;
            }

            double sqrt(double x);
        };
        """
        chunks = self.chunker.extract_chunks(code, 'cpp', self.test_file)

        # Should detect the class and its methods
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("class Calculator", chunks[0]['code'])
        self.assertIn("T add(T a, T b)", chunks[0]['code'])
        self.assertIn("double sqrt(double x)", chunks[0]['code'])

    def test_empty_file(self):
        chunks = self.chunker.extract_chunks("", 'cpp', self.test_file)
        self.assertEqual(len(chunks), 0)

    def test_invalid_language(self):
        with self.assertRaises(ValueError):
            CodeChunker(languages=['invalid_lang'])

    def test_checksum_generation(self):
        code = "int x = 42;"
        chunks = self.chunker.extract_chunks(code, 'cpp', self.test_file)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]['hash']), 32)  # MD5 hash length

    def test_chunk_codebase(self):
        supported_languages = ['python', 'javascript', 'java', 'cpp', 'c', 'go', 'ruby', 'php', 'html', 'css', 'rust']
        chunker = CodeChunker(supported_languages)

        # Specify the directory and list of file extensions to process.
        directory = "./data/"  # Update this to your codebase directory
        file_extensions = [".rs", ".c", ".cpp"]  # Process both Python and JavaScript files, for example

        all_chunks = chunker.chunk_codebase(directory, file_extensions)
        self.assertIsInstance(all_chunks, list)

        for chunk_info in all_chunks:
            with self.subTest(file=chunk_info['file_path']):
                self.assertIn('file_path', chunk_info)
                self.assertIn('start_line', chunk_info)
                self.assertIn('end_line', chunk_info)
                self.assertIn('function_code', chunk_info)
                self.assertIsInstance(chunk_info['function_code'], str)
                self.assertGreaterEqual(chunk_info['end_line'], chunk_info['start_line'])
                # Optionally, print chunk information for debugging
                print(f"File: {chunk_info['file_path']}")
                print(f"Function starts at line {chunk_info['start_line']} and ends at line {chunk_info['end_line']}")
                print("Function chunk:")
                print(chunk_info['function_code'])
                print("-" * 40)

if __name__ == '__main__':
    unittest.main()