"""Integration test: verify agent can use search.py and make edits via tool calls."""

import os
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# 1. Test search.py works standalone
# ---------------------------------------------------------------------------
def test_search_standalone():
    print("=== Test 1: search.py standalone ===")
    result = subprocess.run(
        [sys.executable, "search.py", "muon optimizer neural network"],
        capture_output=True, text=True, timeout=15,
        cwd=os.path.dirname(__file__),
    )
    assert result.returncode == 0, f"search.py failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0, "search.py returned empty"
    lines = result.stdout.strip().splitlines()
    assert len(lines) >= 3, f"Expected at least 3 lines of results, got {len(lines)}"
    print(f"  OK — got {len(lines)} lines of search results")
    print(f"  First result: {lines[0]}")


def test_search_fetch():
    print("\n=== Test 2: search.py --fetch ===")
    result = subprocess.run(
        [sys.executable, "search.py", "--fetch", "https://example.com"],
        capture_output=True, text=True, timeout=15,
        cwd=os.path.dirname(__file__),
    )
    assert result.returncode == 0, f"fetch failed: {result.stderr}"
    assert "Example Domain" in result.stdout, "Expected 'Example Domain' in fetched text"
    print(f"  OK — fetched {len(result.stdout)} chars from example.com")


# ---------------------------------------------------------------------------
# 2. Test workdir creation includes search.py
# ---------------------------------------------------------------------------
def test_workdir_has_search():
    print("\n=== Test 3: workdir includes search.py ===")
    # Simulate what create_isolated_workdir does
    src = os.path.join(os.path.dirname(__file__), "autoresearch")
    tmpdir = tempfile.mkdtemp(prefix="test_workdir_")
    try:
        shutil.copytree(src, tmpdir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc", ".venv"))
        search_py = os.path.join(os.path.dirname(__file__), "search.py")
        if os.path.exists(search_py):
            shutil.copy2(search_py, os.path.join(tmpdir, "search.py"))

        assert os.path.exists(os.path.join(tmpdir, "train.py")), "train.py missing from workdir"
        assert os.path.exists(os.path.join(tmpdir, "search.py")), "search.py missing from workdir"

        # Verify search.py runs from inside the workdir
        result = subprocess.run(
            [sys.executable, "search.py", "test query"],
            capture_output=True, text=True, timeout=15, cwd=tmpdir,
        )
        assert result.returncode == 0, f"search.py failed in workdir: {result.stderr}"
        print(f"  OK — search.py works from workdir ({tmpdir})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Test agent episode with vLLM (requires running vLLM server)
# ---------------------------------------------------------------------------
def test_agent_episode():
    print("\n=== Test 4: agent episode with vLLM ===")
    vllm_url = "http://20.125.45.203:8000/v1"

    # Quick check vLLM is reachable
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{vllm_url}/models", timeout=5)
        resp.read()
    except Exception as e:
        print(f"  SKIP — vLLM not reachable at {vllm_url}: {e}")
        return

    from bash_tool import create_isolated_workdir, run_agent_episode
    from minisweagent.models.litellm_model import LitellmModel

    model = LitellmModel(
        model_name="openai/Qwen/Qwen3-14B",
        model_kwargs={
            "api_base": vllm_url,
            "api_key": "dummy",
            "temperature": 1.0,
        },
        cost_tracking="ignore_errors",
    )

    # Prompt that forces sequential steps — search first, wait for results, then edit
    system_prompt = """\
You are testing a tool integration. You have access to bash.
You MUST do these steps ONE AT A TIME, waiting for each result before proceeding:

Step 1: Search the web
Step 2: Read the search results, then edit train.py based on what you learned
Step 3: Submit

IMPORTANT: Do NOT batch multiple commands. Run one command, wait for the output, then decide your next command."""

    instance_prompt = (
        "First, run: python3 search.py \"muon optimizer learning rate\"\n"
        "Wait for results. Then based on what you find, change MATRIX_LR in train.py.\n"
        "Then submit."
    )

    # Use /tmp instead of /data/tmp for local testing
    workdir = tempfile.mkdtemp(prefix="test_agent_")
    src = os.path.join(os.path.dirname(__file__), "autoresearch")
    shutil.copytree(src, workdir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc", ".venv"))
    search_py = os.path.join(os.path.dirname(__file__), "search.py")
    shutil.copy2(search_py, os.path.join(workdir, "search.py"))

    original = open(os.path.join(workdir, "train.py")).read()

    try:
        modified, trajectory = run_agent_episode(
            workdir=workdir,
            model=model,
            system_prompt=system_prompt,
            instance_prompt=instance_prompt,
            step_limit=10,
        )

        # Check trajectory has messages
        assert len(trajectory) > 0, "Empty trajectory"
        print(f"  Trajectory: {len(trajectory)} messages")

        # Check if search was called (look for search.py in any tool call)
        traj_text = str(trajectory)
        used_search = "search.py" in traj_text
        print(f"  Used search.py: {used_search}")

        # Check tool calls happened
        tool_calls = sum(1 for m in trajectory if m.get("role") == "tool")
        print(f"  Tool calls: {tool_calls}")

        # Check if train.py was modified
        was_modified = modified != original
        print(f"  Modified train.py: {was_modified}")

        # Dump trajectory for inspection
        print(f"\n  --- Trajectory dump ---")
        for i, msg in enumerate(trajectory):
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:2000]
            # Also dump any extra keys beyond role/content
            extra = {k: str(v)[:200] for k, v in msg.items() if k not in ("role", "content")}
            print(f"  [{i}] role={role} extra={extra}")
            print(f"       {content}")

        print(f"\n  OK — agent completed episode")

    except Exception as e:
        print(f"  FAILED — {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    test_search_standalone()
    test_search_fetch()
    test_workdir_has_search()
    test_agent_episode()
    print("\n=== All tests complete ===")
