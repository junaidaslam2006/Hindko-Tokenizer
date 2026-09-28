"""Hindko newspaper corpus pipeline.

Stages (see build.py): extract -> segment -> clean -> dedup -> gate -> emit.
"""
__all__ = ['inpage', 'segment', 'clean', 'build']
