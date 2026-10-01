import os
import argparse
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def fetch_gemini_models():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set.")
        return
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"\n{'Model Identifier':<55} | {'Display Name':<55}")
    print("-" * 115)
    for m in client.models.list():
        print(f"{m.name:<55} | {m.display_name:<55}")


def fetch_hf_models():
    if not HUGGINGFACE_API_KEY:
        print("Error: HUGGINGFACE_API_KEY not set.")
        return
    url = "https://router.huggingface.co/v1/models"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        models = r.json().get("data", [])
        if not models:
            print("No models found.")
            return
        print(f"\n{'Model ID':<75} | {'Owned By':<30}")
        print("-" * 110)
        for m in sorted(models, key=lambda x: x.get("id", "")):
            print(f"{m.get('id', ''):<75} | {m.get('owned_by', ''):<30}")
    except Exception as e:
        print(f"Error fetching HF models: {e}")


def fetch_nvidia_models():
    if not NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY not set.")
        return
    url = "https://integrate.api.nvidia.com/v1/models"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        models = r.json().get("data", [])
        if not models:
            print("No models found.")
            return
        seen = set()
        unique = []
        for m in models:
            mid = m.get("id", "")
            if mid not in seen:
                seen.add(mid)
                unique.append(m)
        print(f"\n{'Model ID':<75} | {'Owned By':<30}")
        print("-" * 110)
        for m in sorted(unique, key=lambda x: x.get("id", "")):
            print(f"{m.get('id', ''):<75} | {m.get('owned_by', ''):<30}")
    except Exception as e:
        print(f"Error fetching Nvidia models: {e}")


def fetch_groq_models():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set.")
        return
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        models = r.json().get("data", [])
        if not models:
            print("No models found.")
            return
        print(f"\n{'Model ID':<55} | {'Owned By':<20} | {'Context Window':<15}")
        print("-" * 95)
        for m in sorted(models, key=lambda x: x.get("id", "")):
            ctx = m.get("context_window", "")
            print(f"{m.get('id', ''):<55} | {m.get('owned_by', ''):<20} | {ctx:<15}")
    except Exception as e:
        print(f"Error fetching Groq models: {e}")


PROVIDERS = {
    "gemini": fetch_gemini_models,
    "hf": fetch_hf_models,
    "nvidia": fetch_nvidia_models,
    "groq": fetch_groq_models,
}


def main():
    parser = argparse.ArgumentParser(description="List available models from AI providers.")
    parser.add_argument("--gemini", action="store_true", help="List Gemini models")
    parser.add_argument("--hf", action="store_true", help="List HuggingFace models")
    parser.add_argument("--nvidia", action="store_true", help="List Nvidia NIM models")
    parser.add_argument("--groq", action="store_true", help="List Groq models")
    args = parser.parse_args()

    any_flag = args.gemini or args.hf or args.nvidia or args.groq
    providers = []
    if not any_flag or args.gemini:
        providers.append(("gemini", fetch_gemini_models))
    if not any_flag or args.hf:
        providers.append(("hf", fetch_hf_models))
    if not any_flag or args.nvidia:
        providers.append(("nvidia", fetch_nvidia_models))
    if not any_flag or args.groq:
        providers.append(("groq", fetch_groq_models))

    for name, func in providers:
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()} Models")
        print(f"{'=' * 60}")
        func()


if __name__ == "__main__":
    main()
