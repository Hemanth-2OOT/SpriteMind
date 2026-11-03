"""SpriteMind Server Runner."""
import sys
import os
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn


def main():
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"==================================================")
    print(f"  Starting SpriteMind Local AI Server")
    print(f"  URL: http://{host}:{port}")
    print(f"==================================================")
    uvicorn.run("spritemind.server.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
