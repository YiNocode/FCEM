"""Launch the FCEM Vue visualization backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="FCEM real-time visualization server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    import socket

    import uvicorn

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((args.host, args.port))
    except OSError as exc:
        print(
            f"无法绑定 {args.host}:{args.port} ({exc}).\n"
            f"端口可能已被占用。请换端口，例如：\n"
            f"  python scripts/run_viz_server.py --port 19090"
        )
        raise SystemExit(1) from exc
    finally:
        probe.close()

    print(f"FCEM viz server: http://{args.host}:{args.port}")
    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "server")],
    )


if __name__ == "__main__":
    main()
