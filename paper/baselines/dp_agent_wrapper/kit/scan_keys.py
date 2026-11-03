"""Key scan for a publish folder (fork of main-session (2), 2026-10-05). Reads every file under the given folders as bytes
(text and npz alike) and reports files that contain a credential-like pattern. It prints the file, the byte offset and
the pattern NAME only - never the matched bytes. Exit 0 = no hit, 1 = hits (do not publish), 2 = usage.
Usage: python scan_keys.py <folder> [<folder> ...]
"""
import re
import sys
from pathlib import Path

PATTERNS = {
    "anthropic_key": rb"sk-ant-[A-Za-z0-9_\-]{10,}",
    "openai_style_key": rb"sk-(?:proj-)?[A-Za-z0-9]{20,}",
    "github_token": rb"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}",
    "aws_access_key": rb"AKIA[0-9A-Z]{16}",
    "bearer_header": rb"[Bb]earer\s+[A-Za-z0-9_\-.=]{20,}",
    "key_assignment": rb"(?i)(?:api[_-]?key|x-api-key|authorization|secret|password)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+=]{16,}",
    "env_key_line": rb"(?m)^[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET)[A-Z0-9_]*=\S+",
}
COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}


def scan(paths):
    hits, files = [], 0
    for root in paths:
        root = Path(root)
        for f in ([root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())):
            files += 1
            if f.name.startswith(".env") or f.suffix == ".env":
                hits.append((str(f), 0, "dotenv_file_name"))
                continue
            data = f.read_bytes()
            for name, rx in COMPILED.items():
                m = rx.search(data)
                if m:
                    hits.append((str(f), m.start(), name))
    return files, hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    files, hits = scan(sys.argv[1:])
    for f, off, name in hits:
        print(f"HIT {name} at byte {off}: {f}")
    print(f"scanned {files} files, {len(hits)} hits")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
