"""
Benchmark for the Liquid-style template engine.
Processes 1000 templates and reports timing + allocation metrics.
"""

import sys
import time
import tracemalloc

sys.path.insert(0, '.')
from lib.liquid.template import Template

# --- Generate test templates ---
TEMPLATES = []
for i in range(200):
    TEMPLATES.append(f"Hello {{{{ name }}}}! Welcome to {{{{ company }}}}. Your order #{{{{ order_id }}}} is confirmed.")
for i in range(200):
    TEMPLATES.append(
        "Items: {{% for item in items %}}{{{{ item.name }}}}: ${{{{ item.price }}}}, {{% endfor %}}"
    )
for i in range(200):
    TEMPLATES.append(
        "{{% if premium %}}Welcome back, VIP {{{{ name }}}}!{{% endif %}}"
        "Your balance: ${{{{ balance }}}}"
    )
for i in range(200):
    TEMPLATES.append(
        "Report for {{{{ company }}}}:\n"
        "{{% for dept in departments %}}"
        "  Department: {{{{ dept.name }}}} - {{{{ dept.count }}}} employees\n"
        "{{% endfor %}}"
        "Total: {{{{ total_employees }}}} employees"
    )
for i in range(200):
    TEMPLATES.append(f"Static content block {i} with no variables at all.")

CONTEXT = {
    "name": "Alice",
    "company": "Acme Corp",
    "order_id": "12345",
    "items": [
        {"name": "Widget", "price": "9.99"},
        {"name": "Gadget", "price": "24.99"},
        {"name": "Doohickey", "price": "4.50"},
    ],
    "premium": True,
    "balance": "142.50",
    "departments": [
        {"name": "Engineering", "count": "42"},
        {"name": "Sales", "count": "18"},
        {"name": "Marketing", "count": "12"},
    ],
    "total_employees": "72",
}

# --- Benchmark parse ---
tracemalloc.start()
alloc_before = tracemalloc.get_traced_memory()[0]

parse_start = time.perf_counter()
parsed = []
for src in TEMPLATES:
    t = Template(src)
    t.parse()
    parsed.append(t)
parse_time_ms = (time.perf_counter() - parse_start) * 1000

# --- Benchmark render ---
render_start = time.perf_counter()
for t in parsed:
    t.render(CONTEXT)
render_time_ms = (time.perf_counter() - render_start) * 1000

alloc_after = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()

combined_time_ms = parse_time_ms + render_time_ms
allocations = alloc_after - alloc_before

# Print metrics in key_value format
print("---")
print(f"combined_time_ms: {combined_time_ms:.6f}")
print(f"parse_time_ms: {parse_time_ms:.6f}")
print(f"render_time_ms: {render_time_ms:.6f}")
print(f"allocations: {allocations}")
