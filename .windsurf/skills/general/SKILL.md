---
name: samutrain_ops
description: Tools and rules for managing samuTrain server, database resets, and code style.
---

### Code Style Rules (CRITICAL)
- Use 2-space indentation for all files.
- Never write comments at the end of a line.
- Use English for comments.
- Never use Hungarian for comments.
- Keep the code dense and compact.

### Terminal Commands
When asked to run the server or manage data, use these exact patterns:
- Run Server: `python scripts/run_server.py --data-folder "data/single_case" --model-folder "models/new_model"`
- Reset DB: `python scripts/run_server.py --reset-db --data-folder "data/single_case"`

### Troubleshooting Knowledge
- Calamari Load Error: If loading fails with "checkpoint version 2", it is a known TF 2.4 compatibility issue. Proceed by performing one training cycle to force a version 5 export.
- Static Files: Always serve static files using absolute paths via `os.path.dirname(os.path.abspath(__file__))`.