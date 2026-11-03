import os
import sys
from pathlib import Path

root = Path(".")
storage_map = {}
total_files = 0
total_bytes = 0

for p in root.rglob("*"):
    if p.is_file():
        # Skip git directory if any
        if ".git" in p.parts:
            continue
        total_files += 1
        size = p.stat().st_size
        total_bytes += size
        
        # Categorize path
        if len(p.parts) == 1:
            cat = "root"
        elif len(p.parts) == 2:
            cat = p.parts[0]
        else:
            cat = f"{p.parts[0]}/{p.parts[1]}"
            if len(p.parts) > 3 and p.parts[0] == "data":
                cat = f"{p.parts[0]}/{p.parts[1]}/{p.parts[2]}"

        if cat not in storage_map:
            storage_map[cat] = {"count": 0, "size": 0, "extensions": set(), "sample_files": []}
        storage_map[cat]["count"] += 1
        storage_map[cat]["size"] += size
        storage_map[cat]["extensions"].add(p.suffix.lower() if p.suffix else "(no-ext)")
        if len(storage_map[cat]["sample_files"]) < 3:
            storage_map[cat]["sample_files"].append(p.name)

print(f"Total Files: {total_files}, Total Size: {total_bytes / (1024*1024):.2f} MB\n")
print(f"{'Directory Path':<40} | {'Files':<6} | {'Size (MB)':<10} | {'Extensions'}")
print("-" * 85)
for d, info in sorted(storage_map.items()):
    exts = ", ".join(sorted(list(info["extensions"]))[:6])
    mb = info["size"] / (1024 * 1024)
    print(f"{d:<40} | {info['count']:6d} | {mb:10.2f} | {exts}")
