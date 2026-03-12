"""
Smoke test for power sampling loop. Runs through every stage of the pipeline
with minimal params to verify everything works end-to-end.

Does NOT run a full 5-min experiment — dispatches to the H100 then kills
after 15s, just enough to confirm SSH + file sync + process launch work.

Usage:
    python smoke_test_power.py [--vllm-base-url URL]
"""

import argparse
import sys
import time
from pathlib import Path

from openai import OpenAI

from environment import BASELINE_VAL_BPB
from loop_baseline import (
    EXPERIMENT_FLEET,
    MODEL,
    VLLM_BASE_URL,
    make_diff,
)
from loop_power_sampling import (
    POWER_SYSTEM_PROMPT,
    apply_sed_commands,
    build_power_prompt,
    parse_sed_commands,
)
from power_sampling import best_of_n, power_sample
from runners import SSHRunner


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def step(n, title):
    print(f"\n{'='*60}")
    print(f"STEP {n}: {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--vllm-base-url", default=VLLM_BASE_URL)
    args = parser.parse_args()

    baseline = Path("autoresearch/train.py").read_text()
    client = OpenAI(base_url=args.vllm_base_url, api_key="dummy")
    failures = []
    modified = None
    result_ps = None
    result_bn = None

    # ------------------------------------------------------------------
    step(1, "vLLM connectivity + logprobs")
    # ------------------------------------------------------------------
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Say hello in 5 words."}],
            max_tokens=20,
            logprobs=True,
            top_logprobs=1,
        )
        text = resp.choices[0].message.content
        lps = resp.choices[0].logprobs.content
        print(f"  Response: {text!r}")
        print(f"  Logprobs: {len(lps)} tokens, first={lps[0].logprob:.3f}")
        assert lps and len(lps) > 0
        print(f"  {PASS}")
    except Exception as e:
        print(f"  {FAIL}: {e}")
        failures.append("vLLM connectivity")

    # ------------------------------------------------------------------
    step(2, "continue_final_message (needed for MCMC resampling)")
    # ------------------------------------------------------------------
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "user", "content": "Count from 1 to 10."},
                {"role": "assistant", "content": "1, 2, 3,"},
            ],
            max_tokens=30,
            logprobs=True,
            top_logprobs=1,
            extra_body={"continue_final_message": True, "add_generation_prompt": False},
        )
        text = resp.choices[0].message.content
        lps = resp.choices[0].logprobs.content
        print(f"  Continuation: {text!r}")
        print(f"  Logprobs: {len(lps)} tokens")
        assert lps and len(lps) > 0
        print(f"  {PASS}")
    except Exception as e:
        print(f"  {FAIL}: {e}")
        print("  MCMC resampling requires this. best-of-n will still work.")
        failures.append("continue_final_message")

    # ------------------------------------------------------------------
    step(3, "power_sample (2 blocks, 2 MCMC steps, 4096 tokens)")
    # ------------------------------------------------------------------
    user_prompt = build_power_prompt(baseline, "")
    try:
        t0 = time.time()
        result_ps = power_sample(
            client=client, model=args.model,
            system_prompt=POWER_SYSTEM_PROMPT, user_prompt=user_prompt,
            alpha=4.0, max_tokens=4096, block_num=2, mcmc_steps=2,
        )
        elapsed = time.time() - t0
        print(f"  Tokens: {len(result_ps.tokens)}, generated: {result_ps.total_tokens_generated}")
        print(f"  Acceptance rate: {result_ps.acceptance_rate:.2f}")
        print(f"  Wall time: {elapsed:.1f}s")
        print(f"  Preview: {result_ps.text[:200]!r}")
        assert len(result_ps.tokens) > 0
        assert len(result_ps.logprobs) == len(result_ps.tokens)
        print(f"  {PASS}")
    except Exception as e:
        print(f"  {FAIL}: {e}")
        import traceback; traceback.print_exc()
        failures.append("power_sample")

    # ------------------------------------------------------------------
    step(4, "best_of_n (n=2, 4096 tokens)")
    # ------------------------------------------------------------------
    try:
        t0 = time.time()
        result_bn = best_of_n(
            client=client, model=args.model,
            system_prompt=POWER_SYSTEM_PROMPT, user_prompt=user_prompt,
            n=2, max_tokens=4096,
        )
        elapsed = time.time() - t0
        print(f"  Tokens: {len(result_bn.tokens)}, generated: {result_bn.total_tokens_generated}")
        print(f"  Wall time: {elapsed:.1f}s")
        print(f"  Preview: {result_bn.text[:200]!r}")
        assert len(result_bn.tokens) > 0
        print(f"  {PASS}")
    except Exception as e:
        print(f"  {FAIL}: {e}")
        failures.append("best_of_n")

    # ------------------------------------------------------------------
    step(5, "Sed parsing + application from model output")
    # ------------------------------------------------------------------
    # Try results from whichever steps succeeded
    best_output = None
    candidates = []
    if result_ps is not None:
        candidates.append(("power_sample", result_ps))
    if result_bn is not None:
        candidates.append(("best_of_n", result_bn))

    for label, res in candidates:
        cmds = parse_sed_commands(res.text)
        if cmds:
            print(f"  Found {len(cmds)} sed commands in {label} output")
            best_output = (label, res, cmds)
            break

    if best_output is None:
        print(f"  {WARN}: No sed commands found. Retrying with 2048 tokens...")
        result_retry = best_of_n(
            client=client, model=args.model,
            system_prompt=POWER_SYSTEM_PROMPT, user_prompt=user_prompt,
            n=1, max_tokens=2048,
        )
        cmds = parse_sed_commands(result_retry.text)
        if cmds:
            best_output = ("retry", result_retry, cmds)
        else:
            print(f"  Full output:\n{result_retry.text[:500]}")

    if best_output:
        label, res, cmds = best_output
        for cmd in cmds[:5]:
            print(f"    {cmd}")
        modified = apply_sed_commands(baseline, cmds)
        diff = make_diff(baseline, modified)
        changed = baseline != modified
        print(f"  File changed: {changed}")
        if changed:
            for line in diff.splitlines()[:8]:
                print(f"    {line}")
            print(f"  {PASS}")
        else:
            print(f"  {WARN}: sed commands parsed but didn't change the file")
    else:
        print(f"  {FAIL}: Model not producing sed commands. Prompt may need tuning.")
        failures.append("sed_parsing")

    # ------------------------------------------------------------------
    step(6, "Experiment dispatch (15s then kill — just testing the pipe)")
    # ------------------------------------------------------------------
    if modified and baseline != modified:
        slot = EXPERIMENT_FLEET[0]
        runner = SSHRunner(slot, timeout=15)
        print(f"  Dispatching to {slot.name} with 15s timeout...")
        t0 = time.time()
        output = runner.run(modified)
        elapsed = time.time() - t0
        print(f"  Return code: {output.returncode}")
        print(f"  Wall time: {elapsed:.1f}s")

        if output.returncode == 255:
            print(f"  {FAIL}: SSH connection failed — {output.stderr[:200]}")
            failures.append("dispatch_ssh")
        elif output.returncode == -1 and "TIMEOUT" in output.stderr:
            # Expected: we killed it after 15s. This means it started successfully.
            stdout_lines = output.stdout.strip().splitlines() if output.stdout else []
            print(f"  Timed out as expected (experiment was running)")
            print(f"  Captured {len(stdout_lines)} lines of output")
            if stdout_lines:
                for line in stdout_lines[:5]:
                    print(f"    {line}")
            print(f"  {PASS} (dispatch + file sync + process launch all work)")
        elif output.returncode == 0:
            # Finished in 15s? Probably a crash, but let's check
            val_bpb = None
            from environment import parse_metrics
            metrics = parse_metrics(output.stdout) if output.stdout else {}
            val_bpb = metrics.get("val_bpb")
            if val_bpb:
                print(f"  Experiment completed in {elapsed:.0f}s! val_bpb={val_bpb}")
            else:
                print(f"  Exited with 0 but no val_bpb (fast crash?)")
                print(f"  Last output: {(output.stdout or '')[-300:]}")
            print(f"  {PASS} (dispatch works)")
        else:
            reason = output.stderr or output.stdout or "no output"
            print(f"  Experiment crashed (exit {output.returncode})")
            print(f"  {reason[-200:]}")
            print(f"  {PASS} (dispatch works — crash is fine for smoke test)")
    else:
        print(f"  {SKIP}: no modified train.py to dispatch")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if not failures:
        print(f"{PASS} — all steps passed, ready for full run")
    else:
        print(f"{FAIL} — failed steps: {', '.join(failures)}")
    print(f"{'='*60}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
