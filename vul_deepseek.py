from pydantic import BaseModel
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.docstore.document import Document

import re
import json


# ------------------- Data Models -------------------
class VulLine(BaseModel):
    lineNum: int
    lineCode: str


class DetectionResult(BaseModel):
    language: str
    is_vulnerability: bool
    vulnerabilityType: str
    vulnerabilityLines: List[VulLine]
    riskLevel: str
    explanation: str
    fixCode: str


# ------------------- Knowledge Base Setup -------------------
vulnerability_db = [
    {
        "text": "Buffer overflow via strcpy in C/C++",
        "metadata": {
            "language": "c",
            "vulnerability_type": "CWE-120: Buffer Copy Without Checking Size of Input",
            "risk_level": "High",
            "remediation": "Use strncpy with proper bounds checking",
            "example_fix": "strncpy(dest, src, dest_size - 1); dest[dest_size - 1] = '\\0';",
            "patterns": ["strcpy(", "gets("]
        }
    },
    {
        "text": "Code injection through eval in Python",
        "metadata": {
            "language": "python",
            "vulnerability_type": "CWE-94: Code Injection",
            "risk_level": "Critical",
            "remediation": "Avoid eval with untrusted input, use safe parsers",
            "example_fix": "json.loads(sanitized_input)",
            "patterns": ["eval(", "pickle.loads("]
        }
    }
]


# ------------------- RAG Components -------------------
class VectorStoreBuilder:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ";"]
        )

    def build_vector_store(self, documents):
        chunks = []
        for doc in documents:
            doc = Document(str(doc))
            chunks.extend(self.text_splitter.split_documents([doc]))

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        embeddings = self.encoder.encode(texts)
        vector_store = FAISS.from_embeddings(
            list(zip(texts, embeddings)),
            metadatas=metadatas,
            embedding=self.encoder
        )
        vector_store.save_local("code_vuln_index")
        return vector_store


# Initialize vector store
vector_builder = VectorStoreBuilder()
vector_store = vector_builder.build_vector_store(vulnerability_db)


# ------------------- Code Analysis -------------------
class CodeAnalyzer:
    def __init__(self):
        self.pattern_cache = {}

    def _get_patterns(self, language: str) -> List[str]:
        if language not in self.pattern_cache:
            docs = vector_store.similarity_search(
                f"Show me {language} vulnerability patterns",
                k=5,
                filter={"language": language}
            )
            patterns = []
            for doc in docs:
                patterns.extend(doc.metadata.get("patterns", []))
            self.pattern_cache[language] = list(set(patterns))
        return self.pattern_cache[language]

    def find_vulnerable_lines(self, code: str, language: str) -> List[VulLine]:
        vul_lines = []
        lines = code.split('\n')

        for pattern in self._get_patterns(language):
            for idx, line in enumerate(lines):
                if re.search(re.escape(pattern), line):
                    vul_lines.append(VulLine(
                        lineNum=idx + 1,
                        lineCode=line.strip()
                    ))
        return vul_lines


# ------------------- Enhanced Retriever -------------------
class VulnRetriever:
    def __init__(self):
        self.vector_store = FAISS.load_local(
            "code_vuln_index",
            SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        )
        self.analyzer = CodeAnalyzer()

    def retrieve_context(self, code: str, language: str) -> List[Dict]:
        # Semantic search
        semantic_results = self.vector_store.similarity_search(
            code,
            k=3,
            filter={"language": language}
        )

        # Pattern-based matches
        pattern_matches = self.analyzer.find_vulnerable_lines(code, language)

        # Combine and deduplicate results
        combined = []
        seen = set()

        # Add semantic results first
        for doc in semantic_results:
            key = (doc.metadata["vulnerability_type"], doc.page_content)
            if key not in seen:
                combined.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                seen.add(key)

        # Add pattern matches
        for line in pattern_matches:
            key = (line.lineCode, line.lineNum)
            if key not in seen:
                combined.append({
                    "text": f"Pattern match at line {line.lineNum}: {line.lineCode}",
                    "metadata": {
                        "vulnerability_type": "Pattern-based detection",
                        "risk_level": "High"
                    }
                })
                seen.add(key)

        return combined[:5]  # Return top 5 results


# ------------------- Detection Generator -------------------
class DetectionGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        self.retriever = VulnRetriever()

    def analyze_code(self, code: str, language: str) -> DetectionResult:
        # Retrieve relevant context
        context = self.retriever.retrieve_context(code, language)

        # Generate prompt
        prompt = self._build_prompt(code, language, context)

        # Generate response
        response = self._generate_response(prompt)

        # Parse and validate
        return self._parse_response(response, code, language)

    def _build_prompt(self, code: str, language: str, context: List[Dict]) -> str:
        context_str = "\n".join([
            f"- {item['text']} (Risk: {item['metadata']['risk_level']})"
            for item in context
        ])

        return f"""Analyze this {language} code for security vulnerabilities using the following context:

        Context:
        {context_str}

        Code:
        {code}

        Output JSON format:
        {{
            "language": "string",
            "is_vulnerability": boolean,
            "vulnerabilityType": "string",
            "vulnerabilityLines": [{{"lineNum": number, "lineCode": "string"}}],
            "riskLevel": "string",
            "explanation": "string",
            "fixCode": "string"
        }}"""

    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _parse_response(self, raw_response: str, original_code: str, language: str) -> DetectionResult:
        try:
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
            result = json.loads(json_str)

            # Validate line numbers
            lines = original_code.split('\n')
            valid_lines = []
            for line in result.get("vulnerabilityLines", []):
                if 1 <= line["lineNum"] <= len(lines):
                    valid_lines.append(VulLine(**line))

            # Validate fix code
            fix_code = result.get("fixCode", original_code)
            if len(fix_code.split('\n')) != len(lines):
                fix_code = original_code

            return DetectionResult(
                language=language,
                is_vulnerability=result.get("is_vulnerability", False),
                vulnerabilityType=result.get("vulnerabilityType", ""),
                vulnerabilityLines=valid_lines,
                riskLevel=result.get("riskLevel", "Medium"),
                explanation=result.get("explanation", ""),
                fixCode=fix_code
            )
        except:
            return DetectionResult(
                language=language,
                is_vulnerability=False,
                vulnerabilityType="",
                vulnerabilityLines=[],
                riskLevel="None",
                explanation="Analysis failed",
                fixCode=original_code
            )


# ------------------- Usage Example -------------------
if __name__ == "__main__":
    test_code = """
    #include <string.h>

    void process_input(char* user_input) {
        char buffer[128];
        strcpy(buffer, user_input);
    }
    """

    analyzer = DetectionGenerator()
    result = analyzer.analyze_code(test_code, "c")

    print(result.model_dump_json(indent=2))