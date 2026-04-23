# ARCHITECTURE.md version 260323 - samuTrain.py

## 1. Project Vision & Philosophy
The `samuTrain.py` project is a training and server framework. 
The project focuses on creating a man-in-the-loop, supervised online learing process.

## 2. Project Logic/Architecture
### A. Failset-Driven Training
- **Concept:** The training is not based on random batches, but on a curated **Failset**.
  - A traditional online learning is based on a **Trainset**.
	  - Trainset cases are taken one by one, guessed, and compared against ground truth. Model weights with good predictions are made greater and bad predoictons smaller.
	- Here a separate set, **Failset** is kept besides Trainset. Test cases can be either in the Trainset (exclusive) or in the Failset. The Union of the Trainset and the Failset is all the test cases (in an ordinary training model, this is the Trainset).
	  - If a prediction during learning fails (prediction is NOT equal to the ground truth), the case will be in the Failset. (Either it was in the Testset or in the Trainset before.) 
		- If a prediction during learning is OK (prediction is equal to the ground truth), the case will be in the Trainset. (Either it was in the Testset or in the Trainset before.)
		- During online learning the learner pick a test case ranedomly from the Testset or from the Trainset (the probability from which set is it taken from is driven by a parameter, like 10% Failset and 90% Trainset).
	- **Testset** has the same function (checking model accuracy) like for any other online supervised learning.
- **Benefit:** Massive reduction in compute time by focusing only on "unlearned" data.
		
### B. Man-in-the-Loop Integration
- **Concept:** Human-in-the-loop verification at critical checkpoints.
- **Workflow:**
  - If a case has a ground truth, everything goes as any other online learning.
  - If a case doesn't have a ground truth, then the learner guesses its ground truth and it is kept as a ground truth. 
	  - The learning goes forward as like the test case had a ground truth
	  - The test case is marked so that it doesn't have a verified ground truth.
		- On the UI the ground truth (either preset or learner-guessed) are displayed and can be modified by the human supervisor.

### C. Learner/Project Agnostic Architecture
- **Concept:** samuTrain.py is designed to be a learning orchestrator independent of the actual project or AI that it improves.
  - **Calamari OCR** is ised as a test project
	- Custom projects are in the `lib/` folder
	  - **calamari/** OCR library
		- **samuLearnUI.ts** Custom .ts frontend
		
## Strategic Decoupling
A core goal of **samuTrain** is the strict separation of management logic from the ML engine:
- **Engine Agnosticism**: All engine-specific calls (currently Calamari) are isolated in `src/bridge.py`. The orchestrator treats the engine as a replaceable component.
		
## Technical Specifications
- **Current Engine**: Calamari OCR (v2.3.1)
- **Backend**: TensorFlow 2.15.0+
- **Database**: Built-in `sqlite3`

