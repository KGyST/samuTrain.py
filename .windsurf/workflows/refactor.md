---
auto_execution_mode: 0
description: Refactor code
---
You are a senior software engineer tasked with refactoring code to improve its structure, readability, and maintainability.

Your task is:
1. Check code in order to be in concert with python.md
2. Remove unused imports and commented out code
3. Check for duplications in code
  - Check for duplicated literals and define them as constants
  Example before:
  ```
  print("Hello World")
  # Some code
  print("Hello World")
  ```
  Example after:
  ```
  MESSAGE = "Hello World"

  print(MESSAGE)
  # Some code
  print(MESSAGE)
  ```
  - Check for duplicated code and extract it to a function
  Example before:
  ```
  def function1():
      print("Hello World")
      # Some code
      print("Hello World")

      # Some code
  
      print("Hello World")
      # Some code
      print("Hello World")
  ```
  Example after:
  ```
  def print_message():
      print("Hello World")
      # Some code
      print("Hello World")

  def function1():
      print_message()

      # Some code

      print_message()
  ```
  - Check function parameters and return values. Create assertions if a parameter or return value can be defined well.
  Example before:
  ```
  def use_temporary_folder(folder_path: str):
      return _use_folder(folder_path)
  ```
  Example after:
  ```
  def use_temporary_folder(folder_path: str):
      assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"
      assert os.path.isdir(folder_path), f"Folder {folder_path} is not a directory"
      assert os.access(folder_path, os.W_OK), f"Folder {folder_path} is not writable"
      assert len(os.listdir(folder_path)) == 0, f"Folder {folder_path} is not empty"
      
      return _use_folder(folder_path)
  ```