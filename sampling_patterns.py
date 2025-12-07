"""
Different sampling patterns for network RB experiments.
All patterns designed to have approximately 760 total samples.
"""
import numpy as np

min_bounces = 2
max_bounces = 20
n_bounces = max_bounces - min_bounces + 1  # 19 bounces

def create_linear_increasing(target_total=760):
    """Linear increase: more samples at higher bounce numbers"""
    # Start low, end high
    # Formula: samples[m] = a + b*(m - min_bounces)
    # We want: sum over m of (a + b*(m-2)) = 760
    # That's: 19*a + b*(0+1+2+...+18) = 760
    # And: 19*a + b*171 = 760
    # Let's set minimum to 25, maximum to 60
    min_samples = 25
    max_samples = 60

    # Linear interpolation
    slope = (max_samples - min_samples) / (max_bounces - min_bounces)
    samples = {}
    total = 0
    for m in range(min_bounces, max_bounces + 1):
        s = int(min_samples + slope * (m - min_bounces))
        samples[m] = s
        total += s

    # Adjust to hit exactly 760
    diff = target_total - total
    # Distribute the difference
    m = min_bounces
    for _ in range(abs(diff)):
        if diff > 0:
            samples[m] += 1
        elif diff < 0:
            samples[m] -= 1
        m += 1
        if m > max_bounces:
            m = min_bounces

    return samples

def create_linear_decreasing(target_total=760):
    """Linear decrease: more samples at lower bounce numbers"""
    # Start high, end low
    min_samples = 25
    max_samples = 60

    slope = (min_samples - max_samples) / (max_bounces - min_bounces)
    samples = {}
    total = 0
    for m in range(min_bounces, max_bounces + 1):
        s = int(max_samples + slope * (m - min_bounces))
        samples[m] = s
        total += s

    # Adjust to hit exactly 760
    diff = target_total - total
    m = min_bounces
    for _ in range(abs(diff)):
        if diff > 0:
            samples[m] += 1
        elif diff < 0:
            samples[m] -= 1
        m += 1
        if m > max_bounces:
            m = min_bounces

    return samples

def create_endpoints_heavy(target_total=760):
    """Heavy at both ends: more samples at low and high bounce numbers"""
    # U-shaped distribution
    samples = {}
    total = 0

    # Use a quadratic that's high at both ends
    mid = (min_bounces + max_bounces) / 2
    for m in range(min_bounces, max_bounces + 1):
        # Quadratic: high at ends, low in middle
        normalized_pos = (m - mid) / (max_bounces - min_bounces)
        # y = a*x^2 + c, where x is normalized position from center
        s = int(25 + 35 * (2 * normalized_pos)**2)
        samples[m] = s
        total += s

    # Adjust to hit exactly 760
    diff = target_total - total
    step = 1 if diff > 0 else -1
    m = min_bounces
    for _ in range(abs(diff)):
        samples[m] += step
        m += 1
        if m > max_bounces:
            m = min_bounces

    return samples

def create_center_heavy(target_total=760):
    """Heavy in the center: more samples at middle bounce numbers (original weighted pattern)"""
    samples = {
        2: 30, 3: 35, 4: 35, 5: 40, 6: 45,
        7: 45, 8: 48, 9: 48, 10: 48, 11: 48,
        12: 48, 13: 45, 14: 45, 15: 40, 16: 38,
        17: 34, 18: 32, 19: 29, 20: 27
    }
    return samples

# Generate all patterns
patterns = {
    'Linear Increasing': create_linear_increasing(),
    'Linear Decreasing': create_linear_decreasing(),
    'Endpoints Heavy': create_endpoints_heavy(),
    'Center Heavy': create_center_heavy()
}

# Print summary
print("="*70)
print("Sampling Pattern Summary")
print("="*70)
for name, pattern in patterns.items():
    total = sum(pattern.values())
    min_s = min(pattern.values())
    max_s = max(pattern.values())
    print(f"\n{name}:")
    print(f"  Total samples: {total}")
    print(f"  Range: {min_s} - {max_s} samples per bounce")
    print(f"  Pattern: {[pattern[m] for m in range(2, 21)]}")

print("\n" + "="*70)
