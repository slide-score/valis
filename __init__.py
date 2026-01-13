# Namespace package for VALIS - re-export modules from valis.valis
# This allows imports like "from valis import registration" to work
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Re-export modules from valis.valis for convenience
try:
    from valis.valis import (
        registration,
        slide_io,
        valtils,
        warp_tools,
        preprocessing,
        feature_detectors,
        feature_matcher,
        non_rigid_registrars,
        affine_optimizer,
        serial_rigid,
        serial_non_rigid,
        micro_rigid_registrar,
        viz,
        slide_tools,
    )
    __all__ = [
        "registration",
        "slide_io",
        "valtils",
        "warp_tools",
        "preprocessing",
        "feature_detectors",
        "feature_matcher",
        "non_rigid_registrars",
        "affine_optimizer",
        "serial_rigid",
        "serial_non_rigid",
        "micro_rigid_registrar",
        "viz",
        "slide_tools",
    ]
except ImportError:
    # If valis.valis is not available, just be a namespace package
    __all__ = []
