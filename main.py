import sys
import warnings

warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py [deterministic|agentic]")
        print("  deterministic  — fast fixed pipeline (phi4 sentiment only)")
        print("  agentic        — agent-driven pipeline (qwen3.5 decisions + tools)")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "deterministic":
        from src.deterministic.evaluate import main as run
    elif mode == "agentic":
        from src.agentic.evaluate import main as run
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'deterministic' or 'agentic'")
        sys.exit(1)

    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
