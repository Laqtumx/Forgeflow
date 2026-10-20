#ForgeFlow 🔧

AI-assisted issue triage and reviewed patch generation for Python repositories.

ForgeFlow is an open-source project that aims to help developers identify promising GitHub issues, investigate possible root causes, validate proposed fixes, and prepare human-reviewed draft pull requests.

The goal is not to automate code changes blindly. Every proposed patch should be tested, fingerprinted, and explicitly approved before publishing.

«🚧 Status: Early development. The complete workflow is not yet implemented. Do not use ForgeFlow to modify or publish changes to important repositories until its security controls have been implemented and tested.»

✨ Planned Features

- 🔎 Fetch and rank open GitHub issues using transparent heuristics.
- 🧠 Analyze promising issues using an AI provider such as Gemini or Claude.
- 🛠️ Generate precise search-and-replace edits rather than unrestricted diffs.
- 🧪 Run baseline and post-patch tests in an isolated environment.
- 🔐 Require human approval using a SHA-256 patch fingerprint.
- 📦 Store issue and validation results in SQLite.
- 🚀 Prepare draft pull requests on a user's fork after approval.
- 🔁 Retry failed patches with bounded attempts and recorded errors.

🏗️ Planned Workflow

1. Fetch public repository issues.
2. Exclude pull requests and issues with excluded labels.
3. Rank candidates using simple, explainable heuristics.
4. Analyze the strongest candidate with an AI model.
5. Generate a constrained patch proposal.
6. Validate the patch against baseline test results.
7. Show the patch and its SHA-256 fingerprint for review.
8. Publish a disclosed AI-assisted draft PR only after explicit approval.

🧰 Planned Technology Stack

Component| Technology
Language| Python
Issue discovery| GitHub REST API
AI analysis| Gemini / Claude
Local storage| SQLite
Testing| pytest
Patch isolation| Docker
Change verification| SHA-256
Code quality| Python linting tools

🔒 Security Principles

- Never expose API keys or access tokens in source code.
- Treat issue descriptions, repository files, and model output as untrusted input.
- Do not give the AI model unrestricted shell access or arbitrary tools.
- Run untrusted tests in a hardened, resource-limited sandbox.
- Disable network access during test execution.
- Keep publishing credentials separate from read-only credentials.
- Verify the exact patch fingerprint immediately before publishing.
- Require human approval for every proposed pull request.
- Never auto-merge changes.

Docker isolation is a security layer, not a guarantee that untrusted code is harmless.

🎯 Initial Scope

The first release targets Python repositories that use pytest.

The project will be developed incrementally, starting with issue discovery, ranking, and persistent storage before introducing AI-generated patches and publishing.

🗺️ Roadmap

- [ ] Create the Python project structure.
- [ ] Fetch public GitHub issues.
- [ ] Exclude pull requests and excluded labels.
- [ ] Rank and store issue candidates in SQLite.
- [ ] Add automated unit tests.
- [ ] Integrate structured AI issue analysis.
- [ ] Implement constrained patch generation.
- [ ] Add isolated baseline and post-patch testing.
- [ ] Implement SHA-256 approval verification.
- [ ] Add safe draft PR publishing.
- [ ] Evaluate performance on real repositories, including failed attempts.

📊 Evaluation Goals

ForgeFlow will be evaluated using measurable outcomes rather than successful demonstrations alone:

- Issue-ranking usefulness.
- AI analysis accuracy.
- Patch acceptance and test-pass rates.
- Regression rate compared with baseline failures.
- Security-control effectiveness.
- Failed attempts and their recorded causes.

Results will be reported only after experiments have been run.

🤝 Contributing

Contributions, bug reports, tests, documentation improvements, and security reviews are welcome.

Please open an issue to discuss substantial changes before submitting a pull request. Never include real credentials, private repository data, or sensitive logs in issues.

📄 License

License selection is pending. A license will be added before the project is distributed as a formally licensed open-source release.

---

ForgeFlow — investigate carefully, test rigorously, publish only with human approval.