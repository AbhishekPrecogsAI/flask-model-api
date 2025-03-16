import os
import logging
import json
from typing import Union, List, Dict, Any
from openai import OpenAI, AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio

from pydantic import BaseModel, conint, Field, validator
import difflib
import markdown2
from rag_openai import chunk_source_files, index_chunks, retrieve_relevant_chunks
from langDetect import detect_language

from semgrep_util import run_semgrep

# Import logging configuration
from logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Load sensitive data from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=API_KEY)

if not API_KEY:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),  # Retry only on these errors
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff (2s, 4s, 8s, max 10s)
    stop=stop_after_attempt(3)  # Stop after 3 attempts
)

# Pydantic models for structured responses
class VulLine(BaseModel):
    lineNum: int
    lineCode: str




class DetectionResult(BaseModel):
    language: str
    is_vulnerability: bool
    vulnerabilityType: str
    cwe: str
    vulnerabilityLines: List[VulLine]
    riskLevel: Union[float, str] = Field(..., description="CVSS-like risk score (0-10) or severity label.")
    explanation: str
    fixCode: str

    @validator("riskLevel", pre=True)
    def convert_risk_level(cls, value):
        """Convert risk level from string labels to CVE-like numeric scores."""
        severity_mapping = {
            "None": 0.0,
            "Low": 1.0,
            "Medium": 4.0,
            "High": 7.0,
            "Critical": 9.0
        }
        if isinstance(value, str):
            value = value.capitalize()  # Normalize input (e.g., "medium" → "Medium")
            if value in severity_mapping:
                return severity_mapping[value]
            raise ValueError(
                f"Invalid risk level '{value}'. Use a number (0-10) or labels: {list(severity_mapping.keys())}.")

        if not (0.0 <= value <= 10.0):
            raise ValueError("Risk level must be between 0 and 10.")

        return value

    @classmethod
    def validate_response(cls, response_data: Dict[str, Any]) -> "DetectionResult":
        """Validates and converts API response into a DetectionResult object."""
        try:
            return cls(**response_data)
        except ValidationError as e:
            logger.error(f"Validation error in API response: {e.json()}")
            return None


async def analyze_code_vulnerability(code_snippet: str,  semgrep_results=None) -> Union[DetectionResult, dict]:
    """
    Analyze a code snippet for vulnerabilities using OpenAI's API.
    Asynchronously calls GPT API for vulnerability analysis with retries.

    Args:
        code_snippet (str): The code snippet to analyze.
        use_semgrep: use static analyze to furter

    Returns:
        Union[DetectionResult, dict]: The structured analysis result or an error message.
    """



    try:
        # client = OpenAI(api_key=API_KEY)
        semgrep_info = ""
        if semgrep_results:
            semgrep_info = "\n\n### Semgrep Findings:\n"
            for result in semgrep_results:
                semgrep_info += f"- Rule: {result.get('check_id')}\n"
                semgrep_info += f"  - Issue: {result.get('extra', {}).get('message', 'No description')}\n"
                semgrep_info += f"  - Line: {result.get('start', {}).get('line', 'Unknown')}\n"

        prompt = (
            f""""
                You are an advanced cybersecurity expert proficient in all programming languages. 
                Make sure to check the findings from Semgrep (provided below), but don't rely entirely on them. Use your expertise to identify any potential vulnerabilities that Semgrep may have missed or incorrectly flagged.
                Analyze the following code snippet at the function level to identify vulnerabilities.
                Internally, perform a hidden chain-of-thought reasoning process over the code’s property graph—including its Abstract Syntax Tree (AST), Control Flow Graph (CFG), and Program Dependence Graph (PDG)—but do not include any of that internal reasoning in your final response.

                Following the steps for output.

                1. Identify the programming language of the code snippet.
                2. Analyze the code for any vulnerabilities or security issues.
                3. If vulnerabilities are found:
                   - Specify the type of vulnerability.
                   - Map vulnerabilities to CWE categories.
                   - Identify the vulnerable lines of code with the line numbers and the actual code.
                   - Provide a detailed explanation of why these lines are vulnerable and the potential risks.
                4. Suggest efficient fixes for the vulnerable lines based on best practices in the identified programming language, *return the entire code block with the fix** included (not just the modified lines)
                5. Format your entire response as valid JSON.

                ### Code snippet:
                {code_snippet}

                {semgrep_info}  # Add Semgrep findings to GPT for context.
                """
        )


        response = await client.beta.chat.completions.parse(
            model="gpt-4o-2024-11-20", # gpt-4o-2024-08-06
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert."},
                {"role": "user", "content": prompt}
            ],
            response_format=DetectionResult
        )
        result = response.choices[0].message.parsed
        logger.info("Vulnerability analysis completed successfully.")
        return result

    except RateLimitError as e:
        logger.error("Rate limit exceeded. Retrying...")
        raise e  # Retry due to @retry decorator

    except APIConnectionError as e:
        logger.error("Network error. Retrying...")
        raise e  # Retry due to @retry decorator

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return {"error": str(e)}  # Don't retry if it's a fatal API error

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}  # Catch-all for unexpected issues


def analyze_code_vulnerability_with_context(code_snippet: str, retrived_chunks: [str]) -> Union[DetectionResult, dict]:
    """
    Analyze a code snippet for vulnerabilities using OpenAI's API with context retrieved using RAG.

    Args:
        code_snippet (str): The code snippet to analyze. perhaps put at function level
        retrived_chunks: list of code for context

    Returns:
        Union[DetectionResult, dict]: The structured analysis result or an error message.
    """
    try:

        prompt = f"""
        You are an advanced cybersecurity expert proficient in all programming languages. 
        Analyze the following code snippet for vulnerabilities at the function level with context of sourcefile.
        Before providing your final answer, internally reason through the code's property graph—including its Abstract Syntax Tree (AST), 
        Control Flow Graph (CFG), and Program Dependence Graph (PDG)—to identify potential vulnerabilities. 
        Do not output this internal chain-of-thought; only provide the final result in the JSON format specified below.\n\n
        Following the steps for output.
        1. Identify the programming language of the code snippet.
        2. Analyze the code for any vulnerabilities or security issues within the context provided.
        3. If vulnerabilities are found:
           - Specify the type of vulnerability.
           - Identify the vulnerable lines of code with the line numbers and the actual code.
           - Provide a detailed explanation of why these lines are vulnerable and the potential risks.
           - Suggest a complete and efficient fix for the vulnerable code based on root cause and best practise, and **return the entire code block with the fix** included (not just the modified lines).
        4. Format your entire response as valid JSON.
        
        ### Code snippet:
        {code_snippet}
        
        ### Source file context:
        {retrived_chunks}
        """

        # Step 4: Call OpenAI API with the constructed prompt
        # client = OpenAI(api_key=API_KEY)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert."},
                {"role": "user", "content": prompt}
            ],
            response_format=DetectionResult
        )
        analysis_result = response.choices[0].message.parsed
        logger.info("Vulnerability analysis completed successfully. See result below")

        logger.info(analysis_result)


        return analysis_result

    except Exception as e:
        logger.error(f"Error during vulnerability analysis: {str(e)}")
        return {"error": str(e)}

def generate_commit_view_diff(old_code: str, new_code: str) -> str:
    """
    Generate a GitHub commit view diff in Markdown format.

    Args:
        old_code (str): The original code snippet.
        new_code (str): The modified code snippet.

    Returns:
        str: A Markdown-formatted string representing the commit-style diff.
    """
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    # Generate the unified diff
    diff = difflib.unified_diff(
        old_lines, new_lines, lineterm="", fromfile="Old Code", tofile="New Code"
    )

    # Format the diff in Markdown
    markdown_diff = "```diff\n"  # Start a diff code block
    for line in diff:
        if line.startswith("+++ ") or line.startswith("--- "):  # File header lines
            continue  # Skip these lines to focus on content
        elif line.startswith("+"):  # Added line
            markdown_diff += f"+ {line[1:]}\n"
        elif line.startswith("-"):  # Deleted line
            markdown_diff += f"- {line[1:]}\n"
        else:  # Context (unchanged) lines
            markdown_diff += f"  {line}\n"
    markdown_diff += "```"  # End the diff code block

    return markdown_diff


def normalize_code(code: str) -> str:
    """
    Normalize the code by:
    - Stripping leading/trailing whitespace
    - Standardizing line endings
    - Collapsing excessive blank lines
    """
    lines = code.strip().splitlines()
    normalized = [line.strip() for line in lines if line.strip()]  # Remove blank lines
    return "\n".join(normalized)


def filter_relevant_lines(diff: list, relevant_line_nums: list) -> list:
    """
    Filter the diff to include only relevant lines based on line numbers.
    """
    filtered_diff = []
    for i, line in enumerate(diff):
        line_num = i + 1
        if line_num in relevant_line_nums or line.startswith(("+", "-")):
            filtered_diff.append(line)
    return filtered_diff


def generate_incident_diff(old_code: str, new_code: str, relevant_lines: list[int]) -> str:
    """
    Generate a diff for an incident based on highlighted vulnerable lines.
    """
    # Normalize the input code
    old_code = normalize_code(old_code)
    new_code = normalize_code(new_code)

    # Compute the diff
    diff = list(difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        lineterm="",
        fromfile="Old Code",
        tofile="New Code"
    ))

    # Filter to only include relevant lines
    filtered_diff = filter_relevant_lines(diff, relevant_lines)

    # Format the diff as a Markdown block
    markdown_diff = "```diff\n" + "\n".join(filtered_diff) + "\n```"
    return markdown_diff




async def run_detection_no_context(code_snippet: str):
    """
    Main function to demonstrate vulnerability analysis.
    """


    # Step 1: Detect the language
    language = detect_language(code_snippet)




    if not language:
        logger.error("Could not detect language.")
        semgrep_results = None

    logger.info(f"Detected language: {language}")
    logger.info("Starting vulnerability analysis...")


    # Step 2: Run Semgrep on the code
    semgrep_results = run_semgrep(code_snippet, language)

    if semgrep_results:
        logger.info(f"Semgrep found {len(semgrep_results)} potential issues. Passing to GPT.")
        result = await analyze_code_vulnerability(code_snippet, semgrep_results)
    else:
        logger.info("Semgrep found no issues. Proceeding with GPT analysis.")
        result = await analyze_code_vulnerability(code_snippet)


    if isinstance(result, DetectionResult):
        logger.info(json.dumps(result.model_dump(), indent=4))

        # print(result.json(indent=4))
        logger.info("Showing the diff...")

        # Generate Markdown diff
        relevant_lines = [line.lineNum for line in result.vulnerabilityLines]

        markdown_diff_1 = generate_incident_diff(code_snippet, result.fixCode, relevant_lines)
        logger.info(markdown_diff_1)

        # Optionally, convert Markdown to HTML for better viewing (e.g., in a browser)
        html_diff = markdown2.markdown(markdown_diff_1)

        file_id = 'code2'  # use commit id

        with open(f"./{file_id}_diff.html", "w") as f:  # Save as HTML if needed
            f.write(html_diff)
    else:
        logger.error("Analysis failed with error: %s", result.get("error"))



def run_detection_with_context():
    # Path to the codebase folder
    folder_path = "./data/"

    # Step 1: Chunk the source files
    print("Chunking source files...")
    chunks, chunk_mapping = chunk_source_files(folder_path)

    # Step 2: Index the chunks
    print("Indexing chunks...")
    index, embeddings = index_chunks(chunks)

    # index, embeddings, tokenizer,  model = index_chunks(chunks)

    # Step 3: user input code

    code_snippet2 = """static int perf_trace_event_perm(struct ftrace_event_call *tp_event,
    				 struct perf_event *p_event)
             {
                /* The ftrace function trace is allowed only for root. */
                if (ftrace_event_is_function(tp_event) &&
                    perf_paranoid_kernel() && !capable(CAP_SYS_ADMIN))
                    return -EPERM;

                /* No tracing, just counting, so no obvious leak */
                if (!(p_event->attr.sample_type & PERF_SAMPLE_RAW))
                    return 0;

                /* Some events are ok to be traced by non-root users... */
                if (p_event->attach_state == PERF_ATTACH_TASK) {
                    if (tp_event->flags & TRACE_EVENT_FL_CAP_ANY)
                        return 0;
                }

                /*
                 * ...otherwise raw tracepoint data can be a severe data leak,
                 * only allow root to have these.
                 */
                if (perf_paranoid_tracepoint_raw() && !capable(CAP_SYS_ADMIN))
                    return -EPERM;

                return 0;
            }
        """

    # Step 3: Retrieve relevant chunks
    print("Retrieving relevant chunks...")
    # relevant_chunks = retrieve_relevant_chunks(code_snippet2, chunks, index, tokenizer, model, top_k=3)
    relevant_chunks = retrieve_relevant_chunks(code_snippet2, chunks, index, top_k=3)

    logger.info("Starting vulnerability analysis...")
    result = analyze_code_vulnerability_with_context(code_snippet2, relevant_chunks)

    if isinstance(result, DetectionResult):
        # print(result.json(indent=4))
        logger.info(json.dumps(result.model_dump(), indent=4))
    else:
        logger.error("Analysis failed with error: %s", result.get("error"))

    logger.info("Showing the diff...")

    # Generate Markdown diff
    relevant_lines = [line.lineNum for line in result.vulnerabilityLines]

    markdown_diff_1 = generate_incident_diff(code_snippet2, result.fixCode, relevant_lines)
    logger.info(markdown_diff_1)

    # Optionally, convert Markdown to HTML for better viewing (e.g., in a browser)
    html_diff = markdown2.markdown(markdown_diff_1)

    file_id = 'test'  # use commit id

    with open(f"./{file_id}_diff.html", "w") as f:  # Save as HTML if needed
        f.write(html_diff)



if __name__ == "__main__":
    code_snippet = """
                            import sqlite3

                            def get_user_data(username):
                                conn = sqlite3.connect('example.db')
                                cursor = conn.cursor()
                                query = f"SELECT * FROM users WHERE username = '{username}'"
                                cursor.execute(query)
                                return cursor.fetchall()
                            """
    code_2 = """
                def list_directory(directory):
                # Vulnerable: User input is directly passed to the shell command
                command = "ls " + directory
                os.system(command)
    """


    asyncio.run(run_detection_no_context(code_2))

    # run_detection_with_context()
