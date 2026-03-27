"""
Evaluate voice agent system prompt across booking, complaint, and edge case scenarios.
Simulates a scoring pipeline based on prompt characteristics.
Prints metrics in key_value format for the scoring pipeline.
"""

import hashlib
import time
import re

start = time.time()

with open("system_prompt.txt") as f:
    prompt = f.read()

prompt_lower = prompt.lower()
prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

# --- Scoring rubric (deterministic, based on prompt content) ---

def check_features(text, features):
    """Count how many features are present in the prompt."""
    return sum(1 for f in features if f in text)

# Booking score: does the prompt cover booking-related instructions?
booking_features = [
    "booking", "dates", "room", "guest", "payment",
    "confirm", "reservation", "check-in", "check-out", "availability",
]
booking_hits = check_features(prompt_lower, booking_features)
booking_score = min(1.0, 0.3 + 0.07 * booking_hits)

# Complaint score: does the prompt handle complaints well?
complaint_features = [
    "complaint", "apologize", "sorry", "resolution", "refund",
    "acknowledge", "issue", "feedback", "escalat", "compensat",
]
complaint_hits = check_features(prompt_lower, complaint_features)
complaint_score = min(1.0, 0.25 + 0.08 * complaint_hits)

# Edge case score: does the prompt handle unusual situations?
edge_features = [
    "edge case", "background noise", "pause", "repeat", "clarif",
    "transfer", "human agent", "escalat", "still there", "unclear",
]
edge_hits = check_features(prompt_lower, edge_features)
edge_case_score = min(1.0, 0.2 + 0.08 * edge_hits)

# Length penalty: prompts that are too short or too long score worse
word_count = len(prompt.split())
if word_count < 50:
    length_factor = 0.7
elif word_count > 500:
    length_factor = 0.85
else:
    length_factor = 1.0

# Combined eval score (weighted average)
raw_score = 0.4 * booking_score + 0.35 * complaint_score + 0.25 * edge_case_score
eval_score = raw_score * length_factor

# Simulate latency (based on prompt length — longer prompts = higher latency)
base_latency = 180.0  # ms
latency_p95_ms = base_latency + word_count * 0.5

elapsed = time.time() - start

# Print metrics in key_value format
print("---")
print(f"eval_score: {eval_score:.6f}")
print(f"booking_score: {booking_score:.6f}")
print(f"complaint_score: {complaint_score:.6f}")
print(f"edge_case_score: {edge_case_score:.6f}")
print(f"latency_p95_ms: {latency_p95_ms:.1f}")
