# Advent-of-Code-Solver

A script that solves Advent of Code puzzles using AI (Gemini, Hugging Face, Nvidia NIM, or Groq). It writes a Python solution, runs it locally against your puzzle input, submits the answer, and feeds the result back to the model until it is accepted.

## Features

- **Today's puzzle**: with no arguments, solves the current day (AoC unlocks at midnight US Eastern).
- **Any day / year**: `--day 7 --year 2024`.
- **Single part**: `--part 1` or `--part 2`.
- **Multiple AI providers**: Gemini (Google), Hugging Face, Nvidia NIM, or Groq. If every model of one provider fails, the next provider is tried.
- **Self-correcting**: script errors, empty output and "too high / too low" verdicts are sent back to the model. After 2 wrong answers it switches to the next configured provider.
- **No double work**: parts already solved on adventofcode.com are skipped, and their answers are saved locally.

## Requirements

- Python 3.10+
- An Advent of Code account (session cookie)
- At least one AI API key (Gemini, Hugging Face, Nvidia, or Groq)

## Installation

```bash
cd Advent-of-Code-Solver
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

1. Copy the example env file and edit it:

   ```bash
   cp .env.example .env
   ```

2. Set these in `.env`:

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `AOC_SESSION` | Yes | `session` cookie from adventofcode.com (see below). |
   | `YEAR` | No | Default year when `--day` is given without `--year`. |
   | `GEMINI_API_KEY` | For `--model gemini` | From [Google AI Studio](https://aistudio.google.com/apikey). |
   | `HUGGINGFACE_API_KEY` | For `--model hf` | From [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens). |
   | `NVIDIA_API_KEY` | For `--model nvidia` | From [build.nvidia.com](https://build.nvidia.com). |
   | `GROQ_API_KEY` | For `--model groq` | From [Groq Console](https://console.groq.com/keys). |

   Keys for the providers you don't select are optional. When set, those providers are used as fallbacks.

### Getting the AoC session cookie

1. Log in at [adventofcode.com](https://adventofcode.com).
2. Open DevTools (F12) → **Application** (or Storage) → **Cookies** → `https://adventofcode.com`.
3. Copy the value of **`session`** into `AOC_SESSION` in `.env`.

## Usage

```bash
# Today's puzzle, both parts, Gemini
python solver.py

# A specific day (year from YEAR in .env, or the current year)
python solver.py --day 7

# A specific day and year, only Part 2, using Groq
python solver.py --day 7 --year 2024 --part 2 --model groq
```

### Options summary

| Option | Description |
|--------|-------------|
| `--day N` | Day to solve (default: today). |
| `--year YYYY` | Event year (default: `YEAR` env var or the current year). |
| `--part {1,2}` | Solve only this part (default: both). |
| `--model {gemini,hf,nvidia,groq}` | AI provider (default: `gemini`). |

## How it works

1. **Fetch**: downloads the puzzle page and your input (saved to `inputs/input_day_N.txt`) using `AOC_SESSION`. Answers already accepted on the site are read from the page.
2. **Generate**: the chosen provider gets the puzzle text and the input path, and returns a Python script (saved to `scripts/solution_dayN_pP.py`).
3. **Run**: the script runs locally (240s timeout). The last line of stdout is taken as the answer.
4. **Submit**: the answer is POSTed to AoC. If AoC asks you to wait, the bot waits and resubmits. On a wrong answer, the verdict goes back to the model and it tries again (up to 10 attempts).
5. **Save**: accepted answers go to `solutions/day_N_solutions.txt`. Part 2's description is only shown after Part 1 is solved, so the page is fetched again before Part 2.

## Project layout

```bash
Advent-of-Code-Solver/
├── solver.py        # Main script (CLI, AoC requests, AI calls, run & submit)
├── find_models.py   # List available models from AI providers
├── requirements.txt # requests, beautifulsoup4, google-genai, python-dotenv
├── .env.example     # Template for AOC_SESSION and API keys
├── .env             # Your secrets
├── inputs/          # Puzzle inputs
├── scripts/         # Generated solution scripts
└── solutions/       # Accepted answers per day
```

## Listing available models

Use `find_models.py` to see which models each provider offers, then edit the `*_MODELS` lists at the top of `solver.py`:

```bash
python3 find_models.py              # list all providers
python3 find_models.py --gemini     # Gemini only
python3 find_models.py --hf         # HuggingFace only
python3 find_models.py --nvidia     # Nvidia NIM only
python3 find_models.py --groq       # Groq only
```

## Troubleshooting

- **"AOC_SESSION is not set"**: add your session cookie to `.env` (see above). Cookies expire after about a month.
- **"Failed to fetch problem. Status: 400/500"**: your session cookie is invalid or expired.
- **"Day N of YYYY is not unlocked yet"**: puzzles unlock at midnight US Eastern (UTC-5). Since 2025, AoC has 12 days instead of 25.
- **AttributeError about `genai.configure`**: use the `google-genai` package (see `requirements.txt`), not `google-generativeai`.

## License

Use and modify as you like. Automated submissions are subject to Advent of Code's rules and rate limits. Please don't share your puzzle inputs.
