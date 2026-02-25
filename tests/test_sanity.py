import py_compile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRepositorySanity(unittest.TestCase):
    def test_core_python_files_compile(self):
        files = [
            REPO_ROOT / "scripts" / "deconv.py",
            REPO_ROOT / "scripts" / "DIP.py",
            REPO_ROOT / "scripts" / "PET.py",
            REPO_ROOT / "scripts" / "PET_reg.py",
            REPO_ROOT / "src" / "utils" / "losses.py",
        ]
        for path in files:
            py_compile.compile(str(path), doraise=True)

    def test_pet_reg_has_no_hardcoded_local_drive(self):
        text = (REPO_ROOT / "scripts" / "PET_reg.py").read_text(encoding="utf-8")
        self.assertNotIn("D:/distribution_loss", text)

    def test_dip_file_is_ascii_only(self):
        data = (REPO_ROOT / "scripts" / "DIP.py").read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"DIP.py contains non-ASCII bytes: {exc}")


if __name__ == "__main__":
    unittest.main()
