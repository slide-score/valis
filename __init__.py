"""
VALIS package - Virtual Alignment of pathoLogy Image Series

This package is bundled with the registration package.
"""

# Make valis.valis modules available at valis.* level
from valis.valis import (
    registration,
    slide_io,
    affine_optimizer,
    feature_detectors,
    feature_matcher,
    non_rigid_registrars,
    preprocessing,
    serial_non_rigid,
    serial_rigid,
    slide_tools,
    valtils,
    viz,
    warp_tools,
    micro_rigid_registrar,
)

__all__ = [
    "registration",
    "slide_io",
    "affine_optimizer",
    "feature_detectors",
    "feature_matcher",
    "non_rigid_registrars",
    "preprocessing",
    "serial_non_rigid",
    "serial_rigid",
    "slide_tools",
    "valtils",
    "viz",
    "warp_tools",
    "micro_rigid_registrar",
]

