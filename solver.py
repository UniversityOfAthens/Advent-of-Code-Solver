#!/usr/bin/env python3
"""
Advent of Code solver bot: fetches puzzles, generates Python solutions with AI,
runs them locally against your input, and submits the answers.

Options: --day N (default: today), --year YYYY, --part 1|2, --model gemini|hf|nvidia|groq.
  Parts already solved on adventofcode.com are skipped (their answers are saved locally).
"""
import os
import re
import sys
import time
import argparse
import datetime
import subprocess
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- CONFIG ---
AOC_SESSION = os.environ.get("AOC_SESSION")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BASE_URL = "https://adventofcode.com"
MAX_RETRIES = 10
EXECUTION_TIMEOUT = 240
WRONG_ANSWERS_BEFORE_SWITCH = 2  # switch to the next AI provider after this many wrong answers

# AoC unlocks puzzles at midnight US Eastern (UTC-5)
AOC_TZ = datetime.timezone(datetime.timedelta(hours=-5))

SCRIPTS_DIR = "scripts"
INPUTS_DIR = "inputs"
SOLUTIONS_DIR = "solutions"

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash"
]

# Hugging Face Inference Providers (OpenAI-compatible router)
HF_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "codellama/CodeLlama-34b-Instruct-hf",
]

# Nvidia NIM (integrate.api.nvidia.com) – code-capable models
NVIDIA_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "z-ai/glm5",
    "z-ai/glm4.7",
    "nvidia/llama-nemotron-embed-vl-1-v2",
    "moonshotai/kimi-k2.5",
    "minimaxai/minimax-m2.1",
    "deepseek-ai/deepseek-v3.2",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# GROQ (api.groq.com) – OpenAI-compatible chat completions
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

PROVIDER_ORDER = ["gemini", "hf", "nvidia", "groq"]
PROVIDER_KEYS = {
    "gemini": ("GEMINI_API_KEY", GEMINI_API_KEY),
    "hf": ("HUGGINGFACE_API_KEY", HUGGINGFACE_API_KEY),
    "nvidia": ("NVIDIA_API_KEY", NVIDIA_API_KEY),
    "groq": ("GROQ_API_KEY", GROQ_API_KEY),
}


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# --- ADVENT OF CODE ---
def aoc_now() -> datetime.datetime:
    return datetime.datetime.now(AOC_TZ)


def days_in_year(year: int) -> int:
    """AoC ran 25 days per year until 2024; from 2025 on it has 12."""
    return 25 if year < 2025 else 12


def last_unlocked_day(year: int) -> int:
    """Return the last day of `year` whose puzzle is already unlocked (0 if none)."""
    now = aoc_now()
    total = days_in_year(year)
    if year < now.year or (year == now.year and now.month == 12 and now.day > total):
        return total
    if year == now.year and now.month == 12:
        return now.day
    return 0


def aoc_request(method: str, url: str, **kwargs) -> requests.Response:
    """Send a request to adventofcode.com with the session cookie."""
    return requests.request(
        method,
        url,
        cookies={"session": AOC_SESSION},
        timeout=30,
        **kwargs,
    )


def day_url(year: int, day: int) -> str:
    return f"{BASE_URL}/{year}/day/{day}"


def fetch_problem(url: str) -> tuple[str | None, dict[int, str]]:
    """Fetch the puzzle description and any answers already accepted on the server.
    Returns (description, {part: answer}). Description includes Part 2 once Part 1 is solved."""
    log(f"Fetching problem from: {url}")
    r = aoc_request("GET", url)
    if r.status_code != 200:
        log(f"Failed to fetch problem. Status: {r.status_code}")
        return None, {}

    soup = BeautifulSoup(r.text, "html.parser")

    # AoC shows accepted answers as "Your puzzle answer was <code>...</code>"
    existing_answers = {}
    answer_paragraphs = [p for p in soup.find_all("p") if "Your puzzle answer was" in p.get_text()]
    for i, p in enumerate(answer_paragraphs):
        code_tag = p.find("code")
        if code_tag:
            existing_answers[i + 1] = code_tag.get_text()

    articles = soup.find_all("article", class_="day-desc")
    if not articles:
        log("No problem description found.")
        return None, existing_answers
    return "\n".join(a.get_text() for a in articles), existing_answers


def fetch_input(url: str, day: int) -> str | None:
    """Download the puzzle input and save it under inputs/. Returns the file path."""
    r = aoc_request("GET", f"{url}/input")
    if r.status_code != 200:
        log(f"Failed to fetch input. Status: {r.status_code}")
        return None
    path = os.path.join(INPUTS_DIR, f"input_day_{day}.txt")
    with open(path, "w") as f:
        f.write(r.text)
    return path


def submit_answer(url: str, part: int, answer: str) -> tuple[str, str]:
    """Submit an answer. Returns (status, info); waits and resubmits if rate-limited."""
    while True:
        log(f"Submitting Part {part} answer: {answer}")
        r = aoc_request("POST", f"{url}/answer", data={"level": part, "answer": answer})
        text = r.text
        if "That's the right answer" in text:
            return "CORRECT", "Correct answer"
        if "You gave an answer too recently" in text:
            m = re.search(r"You have (?:(\d+)m )?(\d+)s left to wait", text)
            wait = int(m.group(1) or 0) * 60 + int(m.group(2)) if m else 60
            log(f"Rate limited by AoC. Waiting {wait}s...")
            time.sleep(wait + 5)
            continue
        if "too high" in text:
            return "WRONG_HIGH", "Answer is too high"
        if "too low" in text:
            return "WRONG_LOW", "Answer is too low"
        if "already complete it" in text or "already solved" in text:
            return "SOLVED", "Already solved"
        return "WRONG", "Incorrect answer"


def save_solution(day: int, part: int, answer: str) -> None:
    """Save the answer to solutions/day_N_solutions.txt, preserving the other part."""
    path = os.path.join(SOLUTIONS_DIR, f"day_{day}_solutions.txt")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = re.match(r"Part ([12]):\s*(.*)", line)
                if m:
                    data[int(m.group(1))] = m.group(2).strip()
    data[part] = answer
    with open(path, "w") as f:
        for p in sorted(data):
            f.write(f"Part {p}: {data[p]}\n")
    log(f"Saved answer to {path}")


# --- LOCAL EXECUTION ---
def extract_code(text: str) -> str | None:
    """Extract Python code from an AI response (fenced block or raw)."""
    m = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "def " in text or "import " in text:
        return text
    return None


def run_python_script(path: str) -> tuple[str, str, int]:
    """Run a generated script. Returns (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=EXECUTION_TIMEOUT
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timeout: script ran longer than {EXECUTION_TIMEOUT} seconds.", 1


# --- AI PROVIDERS ---
# History is provider-agnostic: a list of {"role": "user"|"assistant", "content": str}.
def send_message_gemini(messages: list[dict]) -> str | None:
    if not GEMINI_API_KEY:
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    for model_name in GEMINI_MODELS:
        log(f"Trying Gemini: {model_name}")
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            if response.text:
                return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                log(f"Quota exceeded on {model_name}, trying next...")
            else:
                log(f"Gemini {model_name}: {e}")
    return None


def _query_openai_compatible(
    name: str, url: str, api_key: str | None, models: list[str], messages: list[dict]
) -> str | None:
    """Try each model on an OpenAI-compatible chat completions endpoint."""
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for model_id in models:
        log(f"Trying {name}: {model_id}")
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.1,
            "stream": False,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code == 503:
                log(f"{name} {model_id} loading, waiting 20s...")
                time.sleep(20)
                r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code in (402, 404, 410, 429):
                log(f"{name} {model_id} unavailable (HTTP {r.status_code}), trying next...")
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log(f"{name} {model_id}: {e}")
    return None


def query_huggingface(messages: list[dict]) -> str | None:
    return _query_openai_compatible(
        "HF", "https://router.huggingface.co/v1/chat/completions",
        HUGGINGFACE_API_KEY, HF_MODELS, messages,
    )


def query_nvidia(messages: list[dict]) -> str | None:
    return _query_openai_compatible(
        "Nvidia", "https://integrate.api.nvidia.com/v1/chat/completions",
        NVIDIA_API_KEY, NVIDIA_MODELS, messages,
    )


def query_groq(messages: list[dict]) -> str | None:
    return _query_openai_compatible(
        "GROQ", "https://api.groq.com/openai/v1/chat/completions",
        GROQ_API_KEY, GROQ_MODELS, messages,
    )


PROVIDERS = {
    "gemini": send_message_gemini,
    "hf": query_huggingface,
    "nvidia": query_nvidia,
    "groq": query_groq,
}


def available_providers(preferred: str) -> list[str]:
    """Providers with an API key set, starting from `preferred`."""
    idx = PROVIDER_ORDER.index(preferred)
    rotated = PROVIDER_ORDER[idx:] + PROVIDER_ORDER[:idx]
    return [p for p in rotated if PROVIDER_KEYS[p][1]]


def send_message(messages: list[dict], preferred: str) -> str:
    """Query the preferred provider; if all its models fail, fall back to the others in order."""
    for provider in available_providers(preferred):
        text = PROVIDERS[provider](messages)
        if text:
            return text
        log(f"All {provider} models failed. Falling back to next provider...")
    raise RuntimeError("All AI providers failed.")


# --- SOLVING ---
def build_prompt(year: int, day: int, part: int, description: str, input_path: str) -> str:
    return f"""You are an expert Python programmer participating in Advent of Code {year}.

--- DAY {day} PART {part} DESCRIPTION ---
{description}

--- INSTRUCTIONS ---
1. Write a very efficient Python script that solves Part {part}.
2. The puzzle input is located at: "{input_path}"
3. YOUR CODE MUST READ THE FILE AT "{input_path}". Do not use input().
4. Print ONLY the final answer to stdout (last line).
5. Use only the Python standard library.
6. Return the code inside a single ```python block.
"""


def solve_part(
    url: str,
    year: int,
    day: int,
    part: int,
    description: str,
    existing_answers: dict[int, str],
    input_path: str,
    model_choice: str,
) -> bool:
    """Generate, run and submit a solution for one part. Returns True if accepted."""
    log(f"--- DAY {day} PART {part} ---")

    if part in existing_answers:
        answer = existing_answers[part]
        log(f"Part {part} already solved on the server. Answer: {answer}")
        save_solution(day, part, answer)
        return True

    providers = available_providers(model_choice)
    messages = [{"role": "user", "content": build_prompt(year, day, part, description, input_path)}]
    script_path = os.path.join(SCRIPTS_DIR, f"solution_day{day}_p{part}.py")
    wrong_answers = 0

    for attempt in range(MAX_RETRIES):
        provider = providers[(wrong_answers // WRONG_ANSWERS_BEFORE_SWITCH) % len(providers)]
        log(f"Part {part} - Attempt {attempt + 1}/{MAX_RETRIES} ({provider})")
        try:
            response_text = send_message(messages, provider)
        except RuntimeError as e:
            log(f"Fatal error getting response: {e}")
            return False
        messages.append({"role": "assistant", "content": response_text})

        code = extract_code(response_text)
        if not code:
            messages.append({"role": "user", "content": "Please provide the Python code inside a ```python block."})
            continue

        with open(script_path, "w") as f:
            f.write(code)

        log(f"Running {script_path}...")
        stdout, stderr, ret_code = run_python_script(script_path)
        if ret_code != 0:
            log(f"Script error:\n{stderr[-2000:]}")
            messages.append({
                "role": "user",
                "content": f"Error running code:\n{stderr[-4000:]}\nPlease fix it. Remember the input is at '{input_path}'.",
            })
            continue
        if not stdout:
            messages.append({"role": "user", "content": "Script finished but printed nothing. Print the answer to stdout."})
            continue

        answer = stdout.splitlines()[-1].strip()
        status, info = submit_answer(url, part, answer)
        if status in ("CORRECT", "SOLVED"):
            log(f"SUCCESS! Day {day} Part {part} solved: {answer}")
            save_solution(day, part, answer)
            return True

        wrong_answers += 1
        log(f"Wrong answer ({status}): {info}")
        messages.append({
            "role": "user",
            "content": f"The server said answer '{answer}' is wrong ({info}). Re-read the problem and fix the logic.",
        })
        time.sleep(2)

    log(f"FAILED Day {day} Part {part}")
    return False


def solve_day(year: int, day: int, part: int | None, model_choice: str) -> bool:
    """Solve one or both parts of a day. Returns True if every requested part is solved."""
    url = day_url(year, day)
    description, existing = fetch_problem(url)
    if not description:
        return False
    input_path = fetch_input(url, day)
    if not input_path:
        return False

    if part in (None, 1):
        ok = solve_part(url, year, day, 1, description, existing, input_path, model_choice)
        if part == 1:
            return ok
        if not ok:
            log("Part 1 failed. Stopping.")
            return False
        if 1 not in existing:
            log("Waiting 5s for Part 2...")
            time.sleep(5)
        # Part 2's text only appears on the page once Part 1 is solved
        description, existing = fetch_problem(url)
        if not description:
            return False

    if "--- Part Two ---" not in description and 2 not in existing:
        log("Part 2 description not available. Ensure Part 1 is solved.")
        return False
    return solve_part(url, year, day, 2, description, existing, input_path, model_choice)


def _ensure_session_and_model(model_choice: str) -> None:
    if not AOC_SESSION:
        log("Error: AOC_SESSION is not set. Set it in .env or environment.")
        sys.exit(1)
    env_name, key = PROVIDER_KEYS[model_choice]
    if not key:
        log(f"Error: {env_name} is not set (required for --model {model_choice}).")
        sys.exit(1)


def run_day(year: int, day: int, part: int | None, model_choice: str) -> None:
    """Solve a single day."""
    _ensure_session_and_model(model_choice)
    if day > last_unlocked_day(year):
        log(f"Day {day} of {year} is not unlocked yet.")
        sys.exit(1)
    ok = solve_day(year, day, part, model_choice)
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description="Advent of Code solver bot: solve and submit with AI.")
    parser.add_argument("--day", type=int, help="The day to solve (default: current day)")
    parser.add_argument("--year", type=int, help="The year to solve (default: YEAR env var or current year)")
    parser.add_argument("--part", type=int, choices=[1, 2], help="Specific part to solve (1 or 2). Default: both.")
    parser.add_argument(
        "--model",
        type=str,
        choices=PROVIDER_ORDER,
        default="gemini",
        help="AI provider: gemini, hf (Hugging Face), nvidia (NIM), or groq. Default: gemini",
    )
    args = parser.parse_args()

    for d in (SCRIPTS_DIR, INPUTS_DIR, SOLUTIONS_DIR):
        os.makedirs(d, exist_ok=True)

    if args.day is not None:
        year = args.year or int(os.environ.get("YEAR") or aoc_now().year)
        run_day(year, args.day, args.part, args.model)
        return

    now = aoc_now()
    if now.month != 12:
        log("No puzzle today: Advent of Code runs in December. Use --day N.")
        sys.exit(1)
    run_day(now.year, now.day, args.part, args.model)


if __name__ == "__main__":
    main()
