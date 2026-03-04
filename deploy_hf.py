"""
Upload CardProcessor-Pro to Hugging Face Spaces.
Uses token directly - no Git required.
"""
import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "CardProcessor-Pro"
REPO_TYPE = "space"

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "processor/__init__.py",
    "processor/pipeline.py",
    "processor/bg_removal.py",
    "processor/upscale.py",
    "processor/redaction.py",
    "processor/zipper.py",
    "static/index.html",
    "static/style.css",
    "static/script.js",
]


def main():
    api = HfApi(token=HF_TOKEN)
    
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"
    
    print(f"Logged in as: {username}")
    print(f"Creating HF Space: {repo_id}")
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            space_sdk="docker",
            exist_ok=True,
            token=HF_TOKEN,
        )
        print(f"Space created successfully!")
    except Exception as e:
        print(f"Note: {e}")
    
    for filepath in FILES_TO_UPLOAD:
        if os.path.exists(filepath):
            print(f"  Uploading {filepath}...")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filepath,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
        else:
            print(f"  SKIPPED (not found): {filepath}")
    
    print(f"\nDone! Your app will be live at:")
    print(f"  https://huggingface.co/spaces/{repo_id}")
    print(f"\nIMPORTANT: Set your GEMINI_API_KEY secret:")
    print(f"  Go to https://huggingface.co/spaces/{repo_id}/settings")
    print(f"  Scroll to 'Repository secrets' -> Add GEMINI_API_KEY")


if __name__ == "__main__":
    main()
