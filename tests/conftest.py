import os

from hypothesis import HealthCheck, settings

# Local: 100 examples (Hypothesis default), full health checks.
# CI: 50 examples, suppress too_slow — faster feedback, no flakiness.
settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("default", max_examples=100)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
