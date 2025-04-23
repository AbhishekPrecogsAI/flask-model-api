import os
import shutil
import logging
import asyncio
import git
import yaml
import stat
from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import Any, Dict, Optional

# Import your existing functions
from treeChunk import CodeChunker
from vul_app import process_and_scan_codebase_batched

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)

# Securely fetch GitHub token
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN", "your-github-token")

class ScanTaskResult(BaseModel):
    status: str
    repo_name: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clone_repo(repo_name: str, access_token: str, clone_dir: str = "./cloned_repos") -> Optional[str]:
    try:
        os.makedirs(clone_dir, exist_ok=True)
        repo_url = f"https://{access_token}:x-oauth-basic@github.com/{repo_name}.git"
        repo_path = os.path.join(clone_dir, repo_name.split("/")[-1])
        logging.info(f"Cloning {repo_name} into {repo_path}...")
        git.Repo.clone_from(repo_url, repo_path)
        return repo_path
    except git.exc.GitCommandError as e:
        logging.error(f"Git error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def delete_repo(repo_path: str) -> None:
    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=remove_readonly)
        logging.info(f"Deleted repo: {repo_path}")
    except Exception as e:
        logging.error(f"Error deleting repo {repo_path}: {e}")

def serialize_pydantic(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, list):
        return [serialize_pydantic(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_pydantic(value) for key, value in obj.items()}
    else:
        return obj

def run_scan(repo_name: str, access_token: str, config: Dict) -> Dict:
    try:
        repo_path = clone_repo(repo_name, access_token)
        if not repo_path:
            raise ValueError("Failed to clone repository")

        chunker = CodeChunker(config.get("scan", {}).get("supported_languages", ["python"]))
        scan_config = config.get("scan", {})
        file_extensions = scan_config.get("file_extensions", [".py"])
        exclude_dirs = scan_config.get("exclude_dirs", [])
        batch_size_chars = scan_config.get("batch_size_chars", 4000)
        max_concurrent = scan_config.get("max_concurrent", 5)

        results = asyncio.run(process_and_scan_codebase_batched(
            chunker, repo_path, file_extensions, exclude_dirs,
            batch_size_chars=batch_size_chars, max_concurrent=max_concurrent
        ))

        delete_repo(repo_path)

        return {
            "status": "SUCCESS",
            "repo_name": repo_name,
            "results": serialize_pydantic(results)
        }

    except Exception as e:
        if 'repo_path' in locals():
            delete_repo(repo_path)
        logging.error(f"Error in scan: {e}")
        return {
            "status": "FAILURE",
            "repo_name": repo_name,
            "error": str(e)
        }

@app.route("/scan", methods=["POST"])
def scan_repo():
    data = request.json
    repo_name = data.get("repo_name")
    access_token = data.get("access_token", GITHUB_ACCESS_TOKEN)

    if not repo_name or not access_token:
        return jsonify({"error": "repo_name and access_token are required"}), 400

    config = load_config("config.yaml")
    result = run_scan(repo_name, access_token, config)

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)


    
