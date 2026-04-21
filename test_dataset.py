"""
Quick evaluation helper for a trained FER model.

Uses the same environment variables as `test_model.py`:
  - FER_MODEL_PATH
  - FER_TEST_DIR
  - FER_OUTPUT_DIR
"""

from test_model import cm_path, report

print(report)
print(f"Confusion matrix written to: {cm_path}")
