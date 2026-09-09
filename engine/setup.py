# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: engine/setup.py
# Role: Person 1 (Low-Latency Core & Memory Architect)
# Description: Setuptools build script compiling Cython modules with aggressive
#              compiler optimization flags (-O3, -march=native, -ffast-math) and
#              generating Cython HTML profiling annotations (annotate=True).
# ==============================================================================

import os
import sys
import platform
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    Options.docstrings = True
    Options.annotate = True  # Generate interactive HTML highlighting GIL interactions
except ImportError:
    cythonize = None

# Platform-specific optimization flags
if platform.system() == "Windows":
    extra_compile_args = [
        "/O2",               # Maximize speed
        "/Oi",               # Enable intrinsic functions
        "/Ot",               # Favor fast code
        "/fp:fast",          # Fast floating point model
        "/arch:AVX2",        # Enable AVX2 vector extensions if supported
        "/GS-",              # Disable buffer security checks for low latency
    ]
    extra_link_args = []
else:
    extra_compile_args = [
        "-O3",                      # Aggressive optimization
        "-march=native",            # Target host CPU architecture
        "-ffast-math",              # Fast floating point
        "-fomit-frame-pointer",     # Free register for performance
        "-funroll-loops",           # Unroll loops for pipeline efficiency
        "-fstrict-aliasing",        # Enforce strict aliasing
        "-DNDEBUG",                 # Strip C asserts
    ]
    extra_link_args = [
        "-O3",
        "-march=native"
    ]

extensions = [
    Extension(
        name="engine.ipc_ring_buffer",
        sources=["engine/ipc_ring_buffer.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=["."],
    ),
    Extension(
        name="engine.matching_engine",
        sources=["engine/matching_engine.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=["."],
    ),
]

if __name__ == "__main__":
    if cythonize is None:
        print("[ERROR] Cython is not installed. Please run: pip install cython")
        sys.exit(1)

    setup(
        name="chronosmatch-engine",
        version="1.0.0",
        author="Person 1 (Low-Latency Architect)",
        description="Zero-Copy Lock-Free Matching Engine for ChronosMatch HFT",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
                "initializedcheck": False,
                "overflowcheck": False,
            },
            annotate=True,  # Generates matching_engine.html and ipc_ring_buffer.html
        ),
    )
