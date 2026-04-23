import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from try_learning_db_writeback import run_learning_db_writeback


def test_learning_db_writeback() -> None:
  run_learning_db_writeback(
    os.path.join(project_root, "data", "single_case"),
    os.path.join(project_root, "models", "generic_ocr_model"),
    epochs=1,
  )


if __name__ == "__main__":
  test_learning_db_writeback()

