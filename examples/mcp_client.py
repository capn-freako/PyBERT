"""Example: drive PyBERT's MCP server from a plain Python MCP client.

This spawns `pybert mcp` as a subprocess over stdio (the same way an MCP-aware
tool like Claude Code/Desktop would) and calls a few of its tools:

  * `get_default_config`   - fetch PyBERT's default configuration
  * `run_simulation`       - run a (small, fast) headless simulation
  * `list_ibis_models`     - introspect an IBIS file bundled with the repo

Requires the `mcp` extra: `uv sync --extra mcp` (or `pip install "pipbert[mcp]"`).

Run from anywhere with:

    uv run python examples/mcp_client.py
"""
import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_IBIS_FILE = "models/ibisami/example_rx.ibs"  # relative to REPO_ROOT


def _text(result) -> str:
    """Pull the JSON text out of a `CallToolResult`."""
    return result.content[0].text


async def main() -> None:
    # `cwd=REPO_ROOT` matters: PyBERT config/model file paths (like EXAMPLE_IBIS_FILE
    # below) are resolved relative to the server process's working directory.
    params = StdioServerParameters(command="uv", args=["run", "pybert", "mcp"], cwd=str(REPO_ROOT))

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        print("Available tools:", ", ".join(sorted(t.name for t in tools.tools)))

        # 1. Fetch PyBERT's default configuration.
        result = await session.call_tool("get_default_config", {})
        config = json.loads(_text(result))
        print(f"\nDefault config has {len(config)} settings, e.g. bit_rate={config['bit_rate']} Gbps")

        # 2. Shrink it down and run a quick simulation.
        config["nbits"] = 1000
        config["eye_bits"] = 500
        result = await session.call_tool("run_simulation", {"config": config})
        metrics = json.loads(_text(result))
        print(f"\nSimulation status: {metrics['status']}")
        print(f"Total jitter/noise-limited performance: {metrics['total_perf']:.3g} bits/s")

        # 3. Introspect an IBIS-AMI model.
        result = await session.call_tool("list_ibis_models", {"ibis_file": EXAMPLE_IBIS_FILE})
        ibis_info = json.loads(_text(result))
        print(f"\n{EXAMPLE_IBIS_FILE}: components={ibis_info['components']}, models={ibis_info['models']}")


if __name__ == "__main__":
    asyncio.run(main())
