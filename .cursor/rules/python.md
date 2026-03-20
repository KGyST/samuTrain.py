# Python Development Protocols & Rules

## 1. Environment & Package Management
- **No Global Installs:** Strictly forbidden to install packages to the global Python interpreter.
- **Venv Lifecycle:**
  - **Usage:** ALWAYS use a virtual environment (`venv`).
  - **Discovery:** If no active `venv` is detected or known, **STOP** and ask the user to provide one or create it.
  - **Switching:** Before changing an existing virtual environment, **ALWAYS** ask for confirmation: *"Should I use the active environment, or create a new one?"*

## 2. Type Safety & Input Validation
- **Type Hinting:** All function signatures shall use Python type hints for all arguments and return values if applicable.
- **Pre-condition Asserts:** Every function shall start with `assert` statements to validate input integrity.
  - **Type Check:** If type hints are applied, assert commands shall not check arguments' type
  - **Path Validation:** For string paths, check if the path exists and if appropriate permissions (R/W) are granted.
  - **State Validation:** If a folder must be empty for a process, include an explicit assert for it.
  - **Custom functions:** When defining multiple items having same assertions, create a helper function to avoid code duplication.
- **Example:**
  ```python
    def assert_empty_directory(target_path: str) -> str | None:
      """Helper function for assertions"""
			"""Note that a directory being empty is a common need so it is a good idea to put this into a function"""
      import os
      if not os.path.isabs(target_path): return f"Path must be absolute: {target_path}"
      if not os.path.exists(target_path): return f"Directory missing: {target_path}"
      if len(os.listdir(target_path)) != 0: return f"Target folder is not empty: {target_path}"
      return None

    def initialize_workspace(target_path: str):
      # Assertions for safety using the Walrus operator
      # Validation passes only if sError is None.
			# `target_path` is an agrument, so snake_case, while `sError` is a local variable so Hungarian naming order + camelCase
      assert (sError := assert_empty_directory(target_path)) is None, sError
        
      # Logic...
      ``` 

## 3. Subprocess & OS Specifics (Windows Compatibility)
### Encoding and Safety
- **Mandatory Encoding:** Always use encoding='utf-8' and errors='replace' for subprocess.Popen or subprocess.run.
- **Stream Decoding:** When monitoring stdout in real-time, ensure the stream is decoded correctly (UTF-8) to avoid UnicodeDecodeError from progress bars or special characters.

## 4. Formatting & PEP8
### Variable conventions
- **Function argument names:** use snake_case
- **Local variables:** use
  - **camelCase** for local variables. Abbreviations in camelCase can be capitalized, like `GUIObserver` instead of `guiObserver`
  - **Hungarian naming order:** for primitive values
    - `b`, `is`, `has` for boolean variables names
    - `s` for string variable names
    - `i` for integer variable names
    - `f` or `r` for float variable names
    - for non-primitive variables, like class instances, no hinting char
    - **Good examples**: `bVerbose`, `isActive`, `hasPermission`, `sPath`, `iItems`
	- **Variable Name Length:** 
	  - Variables used in a small region (a few lines directly following each other) shall have shorter names.
		- Variables used in the whole code (instances are many lines from each other) can have longer, more descriptive names.
		- NEVER use 1 or 2 char long variable names.
		- NEVER use non-English character in variable names.

### Constants and literals
- **Constants:** UPPER_SNAKE_CASE for constants and literals. For short names, leading/trailing underscores are ok, like `_A_` 
- **Literals:** if a literal is used at least 2 times and instances are logically the same, use a constant

### Comments and Documentation
- **Line Placement:** Comments must be on their own lines. Inline comments are strictly forbidden.
- **Language:** All comments and documentation must be in English. 
  - Ask for translating non-English comments if found in a heritage code. Translate only if explicitely commanded.
- **Content:** Use few, but meaningful comments. Focus on the "why" (logic) rather than the "what" (obvious code).

### Code Style
- **File Ending:** Per PEP8, ensure exactly two newlines at the end of every file/macro.
- **Indentation:** Use 2 spaces for indentation.

## 5. Software Design/Architecture
- **Use Design Patterns:** and naming should hint them
  - **Examples:** `settingsSingleton`, `GUIObserver` 
- **Use Nested Functions or Classes:** if needed, for example, avoiding import issues.
  - **Example:**
  ```python
  def recursive_function(parameter: str) -> str:
    # recursion initialization goes here

    def _recursee(param: str) -> str:
		  # as a nested function, its name starts using `_` 
      return param
    
    return _recursee(parameter)
  ```

- **Clean Code:** When applicable, use Clean Code principles
  - **Create a New Function:** Organize a new function when a set of commands can be driven by a 
    - LImited number of simple parameters
    - Resulting an easy to return simple result (or `None`)

## 6. FOSS Priority
### License and Costs
- **Red Alert:** Strictly prefer Free and Open Source Software (FOSS).
- **Hard Constraint:** Avoid any libraries, tools, or SaaS solutions with freemium models, paid tiers, or usage limits.

