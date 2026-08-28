"""Optional integrations with external RL frameworks (plan.md §16).

Importing this package is free. Each integration module imports its
framework lazily, so the core suite keeps zero runtime dependencies;
a missing framework raises an ImportError with install instructions
only when an integration estimator is actually used.
"""
