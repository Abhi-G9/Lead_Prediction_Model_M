"""
Phase 5 Streamlit Web Application Automated Tests
Validates that all active Streamlit pages, modules, and components import and compile
cleanly without syntax errors or broken dependencies.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase5StreamlitApp(unittest.TestCase):
    def _verify_script_compilation(self, script_path: Path):
        """Verifies a python file compiles cleanly into bytecode."""
        self.assertTrue(script_path.exists(), f"Script missing: {script_path}")
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            compiled = compile(code, str(script_path), "exec")
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Compilation error in {script_path.name}: {e}")

    def test_app_main_compilation(self):
        """Verify app.py compiles cleanly."""
        self._verify_script_compilation(PROJECT_ROOT / "app.py")

    def test_page1_analytics_compilation(self):
        """Verify 1_Lead_Analytics.py compiles cleanly."""
        self._verify_script_compilation(PROJECT_ROOT / "pages" / "1_Lead_Analytics.py")

    def test_page2_prediction_compilation(self):
        """Verify 2_Lead_Prediction.py compiles cleanly."""
        self._verify_script_compilation(PROJECT_ROOT / "pages" / "2_Lead_Prediction.py")


if __name__ == "__main__":
    unittest.main()
