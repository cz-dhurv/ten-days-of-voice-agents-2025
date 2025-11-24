
"""
MCP client helper for Todoist (corrected).
StdioServerParameters.command must be a single string (not a list).
Choose the proper 'command' below depending on how you start the server:
 - If you use the MCP CLI: "mcp run todoist"
 - If you use the todoist-mcp entrypoint: "todoist-mcp"
"""

from mcp import ClientSession, StdioServerParameters
from typing import Tuple, Dict

# --- Option A: If you run MCP via the MCP CLI (recommended)
MCP_SERVERS = {
    "todoist": StdioServerParameters(command="mcp run todoist"),
}

# --- Option B: If your system has a 'todoist-mcp' entrypoint (uncomment to use)
# MCP_SERVERS = {
#     "todoist": StdioServerParameters(command="todoist-mcp"),
# }


async def get_mcp_client(server_name: str) -> Tuple[ClientSession, Dict]:
    """
    Connect to a running MCP server and return (session, tools_dict).
    The server will be started via the configured command (which must be a string).
    """
    if server_name not in MCP_SERVERS:
        raise ValueError(f"Unknown MCP server: {server_name}")
    params = MCP_SERVERS[server_name]
    session = await ClientSession.connect(params)
    tools = await session.list_tools()
    return session, tools
