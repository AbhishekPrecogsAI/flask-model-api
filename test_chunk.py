from codeChunk import CodeChunker
if __name__ == "__main__":
    chunker = CodeChunker('python')

    sample_code = """
    import os

    class MyClass:
        def __init__(self):
            self.value = 42

        def calculate(self, x):
            return x * self.value

    def helper():
        print("Hello World")
    """

    chunks = chunker.chunk_code(sample_code)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 40}")