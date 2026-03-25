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

### D. Calamari OCR working
- **samu.db** NoSQL database as the bridge between frontend and backend
  - calamari uses samu.db for learning
	- samuLearnUI.ts displays samu.db data
- Calamari DB structure
  - Cases Table Fields
    - id (INTEGER PRIMARY KEY AUTOINCREMENT)
    - img_path (TEXT NOT NULL)
    - ocr_text (TEXT NOT NULL)
    - model_prediction (TEXT)
    - gt_text (TEXT)
    - confidence (REAL NOT NULL)
    - timestamp (DATETIME DEFAULT CURRENT_TIMESTAMP)
    - is_corrected (BOOLEAN DEFAULT FALSE)
    - is_failset (BOOLEAN DEFAULT FALSE)
  - Training Sessions Table Fields
    - id (INTEGER PRIMARY KEY AUTOINCREMENT)
    - session_id (TEXT UNIQUE NOT NULL)
    - data_folder (TEXT NOT NULL)
    - checkpoint_folder (TEXT NOT NULL)
    - network (TEXT)
    - backup_folder (TEXT)
    - model_folder (TEXT)
    - status (TEXT DEFAULT 'started')
    - chars_count (INTEGER DEFAULT 0)
    - started_at (DATETIME DEFAULT CURRENT_TIMESTAMP)
    - completed_at (DATETIME)
    - error_message (TEXT)

### E. Architecture
flowchart TB
    %% --- STYLE DEFINITIONS ---
    classDef BE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef FE fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef DB fill:#e8f5e9,stroke:#2e7d32,stroke-dasharray: 5 5
    classDef EN fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    %% --- SYSTEM LAUNCHER ---
    LAUNCHER[scripts/run_server.py<br/>Server Launcher]:::BE
    
    %% --- CORE SYSTEM ---
    subgraph CORE ["CORE (STABLE)"]
        direction TB
        UI_SHELL[samuLearnUI.ts<br/>index.html]:::FE
        ORCHESTRATOR[main.py<br/>FastAPI Server]:::BE
        STATIC[static/]:::FE
    end
    
    %% --- PROJECT DOMAIN ---
    subgraph PROJECT ["PROJECT (SWAPPABLE)"]
        direction TB
        BRIDGE[bridge.py<br/>OCR Wrapper]:::BE
        
        subgraph ENGINES ["Learning Engines"]
            CAL_LEARNER[engines/calamari_learner.py]:::EN
            CONT_LEARN[engines/continue_learning.py]:::EN
            LEARN_INT[engines/learner_interface.py]:::EN
        end
        
        LIB_CAL[lib/calamari<br/>OCR Library]:::BE
        REPO[repository.py<br/>Data Access]:::BE
        
        subgraph STORAGE ["Data & Models"]
            DB_FILE[(samu.db)]:::DB
            DATA_DIR[data/]:::DB
            MODELS_DIR[models/]:::DB
        end
    end
    
    %% --- INTERFACES (Labeled Relations) ---
    LAUNCHER -- "Launches" --> ORCHESTRATOR
    
    UI_SHELL -- "HTTP API" --> ORCHESTRATOR
    UI_SHELL <-- "JSON Response" --> ORCHESTRATOR
    UI_SHELL -- "Static Files" --> STATIC
    
    ORCHESTRATOR -- "Calls" --> BRIDGE
    ORCHESTRATOR -- "Direct Access" --> DB_FILE
    ORCHESTRATOR -- "Uses" --> REPO
    
    BRIDGE -- "Delegates to" --> ENGINES
    ENGINES -- "Uses" --> LIB_CAL
    
    REPO -- "CRUD Operations" --> DB_FILE
    
    LIB_CAL -- "Read/Write" --> DATA_DIR
    LIB_CAL -- "Load/Save" --> MODELS_DIR
    
    %% --- Data Flow Annotations ---
    ORCHESTRATOR -- "State/Config" --> DB_FILE
    ENGINES -- "Training Progress" --> DB_FILE

