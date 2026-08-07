#!/usr/bin/env python3
"""Live conformance smoke test through the fully generated Python SDK."""

import asyncio
import sys
from pathlib import Path

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "sdk" / "python"))

from brokk_anvil_sdk.api.models_api import ModelsApi
from brokk_anvil_sdk.api.runs_api import RunsApi
from brokk_anvil_sdk.api.server_api import ServerApi
from brokk_anvil_sdk.api.sessions_api import SessionsApi
from brokk_anvil_sdk.api.tools_api import ToolsApi
from brokk_anvil_sdk.api_client import ApiClient
from brokk_anvil_sdk.configuration import Configuration
from brokk_anvil_sdk.models.create_run_request import CreateRunRequest
from brokk_anvil_sdk.models.create_session_request import CreateSessionRequest


async def main(base_url: str, cwd: str) -> None:
    configuration = Configuration(host=base_url)
    configuration.user_agent = "brokk-anvil-sdk-conformance"
    async with ApiClient(configuration) as client:
        assert (await ServerApi(client).get_health()).status == "ok"
        await ModelsApi(client).list_models()
        await ToolsApi(client).list_tools()

        sessions = SessionsApi(client)
        runs = RunsApi(client)
        session = await sessions.create_session(
            CreateSessionRequest(cwd=cwd, permission_mode="acceptEdits")
        )
        run = await runs.create_run(
            session.id, CreateRunRequest(prompt="Python SDK conformance turn")
        )

        for _ in range(800):
            terminal = await runs.get_run(run.id)
            if terminal.status != "running":
                break
            await asyncio.sleep(0.025)
        else:
            raise AssertionError("Python SDK run timed out")

        assert terminal.status == "completed"
        assert terminal.result_text == "SDK conformance complete"
        assert any(candidate.id == run.id for candidate in (await runs.list_runs(session.id)).runs)
        assert (await sessions.delete_session(session.id)).deleted is True

    print("Python SDK conformance passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python.py BASE_URL CWD")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
