"""
Passthrough reward function for autoresearch SDPO.

Reward is already computed by the agent loop (env.step() → result.reward).
This just extracts it from extra_info where the agent loop stores it.
"""


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    if extra_info and "reward_score" in extra_info:
        return {"score": extra_info["reward_score"], "feedback": extra_info.get("feedback", "")}
    return {"score": 0.0, "feedback": "no reward_score in extra_info"}
