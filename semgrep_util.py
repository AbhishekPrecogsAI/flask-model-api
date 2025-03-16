import json, subprocess
import logging

from logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def run_semgrep(code: str, language: str):
    """Runs Semgrep dynamically based on the detected language."""
    # Write code to a temporary file
    with open("temp_code.txt", "w") as f:
        f.write(code)

    # Dynamically select Semgrep config based on language
    semgrep_config = get_semgrep_config(language)

    # Run Semgrep with the correct configuration via subprocess
    semgrep_cmd = ["semgrep", "--config", semgrep_config, "--json", "temp_code.txt"]
    logger.info(f"Running Semgrep with config: {semgrep_config} for language: {language}")

    try:
        result = subprocess.run(semgrep_cmd, capture_output=True, text=True, check=True)
        # Parse the JSON output
        findings = json.loads(result.stdout)
        logger.info(f"Semgrep analysis complete. Found {len(findings.get('results', []))} issues.")
        return findings.get("results", [])
    except subprocess.CalledProcessError as e:
        # Capture the error output (stderr) and log it
        logger.error(f"Semgrep error: {e.stderr}")
        logger.error(f"Semgrep output: {e.stdout}")
        return []
    except json.JSONDecodeError:
        logger.error("Semgrep output is not in expected JSON format.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return []


def get_semgrep_config(language: str):
    """Returns the appropriate Semgrep configuration URL based on the language."""
    language_configs = {
        "Python": "https://semgrep.dev/p/python",
        "JavaScript": "https://semgrep.dev/p/javascript",
        "TypeScript": "https://semgrep.dev/p/typescript",
        "Java": "https://semgrep.dev/p/java",
        "Go": "https://semgrep.dev/p/go",
        "Ruby": "https://semgrep.dev/p/ruby",
        "PHP": "https://semgrep.dev/p/php",
        "C": "https://semgrep.dev/p/c",
        "C++": "https://semgrep.dev/p/cpp",
        "C#": "https://semgrep.dev/p/csharp",
        "Objective-C": "https://semgrep.dev/p/objectivec",
        "Swift": "https://semgrep.dev/p/swift",
        "Kotlin": "https://semgrep.dev/p/kotlin",
        "Shell": "https://semgrep.dev/p/shell",
        "Dockerfile": "https://semgrep.dev/p/dockerfile",
        "YAML": "https://semgrep.dev/p/yaml",
        "JSON": "https://semgrep.dev/p/json",
        "Terraform": "https://semgrep.dev/p/terraform",
        "Markdown": "https://semgrep.dev/p/markdown",
        "Rust": "https://semgrep.dev/p/rust",
        "HCL": "https://semgrep.dev/p/hcl",
        "Scala": "https://semgrep.dev/p/scala"
    }
    # If language not found, fallback to 'auto'
    return language_configs.get(language, "auto")