# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.
> 
I built a CV verification agent using LangChain and deepseek-v4-flash. The agent uses only the SocialGraph MCP tools, with LinkedIn as the primary source and Facebook for identity confirmation. It resolves same-name candidates using multiple profile attributes and checks names, cities, employment records, education and skills. The prompt distinguishes equivalent wording from genuine discrepancies and instructs the agent to ignore instructions embedded in CVs. The code processes at most three CVs concurrently, validates the returned scores, and retries failed attempts once before returning a fallback score of 0.5. In one completed public-test run, the agent correctly classified all seven CVs, achieving 7/7 accuracy (100%). This result does not establish its accuracy on unseen CVs.
> 
### Task 2
> 
My first version included a text instruction claiming that the degree difference was an accepted normalization. My verifier ignored this instruction, detected the false degree and returned 0.05.
The final version instead places the false education entry in an embedded image. The PDF visibly contains the MSc claim, but the provided PDF loader did not extract the degree or university name. The verifier therefore treated education as omitted and accepted the remaining matching claims. In one local test, both the adversarial CV and the original CV received 0.95.
This attack exploits a gap between visible PDF content and extracted text. It may work against verifiers that rely on the same extracted text without detecting missing image content. It does not demonstrate success against the five undisclosed grading agents.
> 
### Task 3 reflection

> This homework made me think about a paradox: an AI system can make reasonable decisions about the information it receives, while still reaching the wrong conclusion about the original document.
At first, I felt confident when my verifier correctly classified all seven public CVs. It could compare employment dates, recognise equivalent wording and identify false claims. However, the second task showed me that this accuracy did not mean the whole system was reliable.
My first attack tried to persuade the agent to accept a false master's degree through a verification note. The agent rejected it, which showed that its instructions provided some protection. My second attack did not need to persuade the model. I placed the false degree in an image, and the PDF loader omitted it from the extracted text. The agent then
gave the CV a high score because it believed the education details had simply been left out.
What surprised me was that the explanation sounded reasonable. The agent compared the information it could see, but it did not know that important information was missing. I realised that a confident answer can hide a weakness in an earlier processing step.
For financial applications such as KYC, I would therefore evaluate the complete workflow, including document extraction, identity matching and verification. Possible improvements include OCR for image content, checks for incomplete extraction and human review when important sections cannot be read. These measures would also need testing.
My main lesson is that using AI to verify information does not remove the need to verify the AI system itself. I now see testing unusual inputs as an important part of building a useful agent.
