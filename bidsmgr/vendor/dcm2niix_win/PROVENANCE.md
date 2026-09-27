# dcm2niix.exe — a Windows build that can convert MR spectroscopy

## Why this is here

The released Windows dcm2niix cannot convert MR spectroscopy. Its linker
reserves 16,388,608 bytes of stack, an MSVC build needs more than that on a
Siemens `svs_se` series, and Windows terminates the process with
`0xC00000FD` (STATUS_STACK_OVERFLOW) **and an empty stderr**. Not to be
confused with `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN), which is what
dcm2niix dies with when handed a path past `MAX_PATH`: a different
failure with a different fix. Nothing is
written and nothing is said, so every caller reads it as "this folder holds no
DICOM images" rather than "the converter was killed".

It is not a packaging problem. The wheel and the official GitHub release are
different builds — MSC1951 and MSC1944, different sizes — and both carry the
same reserve and fail identically. It is also not path length, not the DICOMs,
and not `dcm2niix`'s spectroscopy support, which works.

Measured on a 65-file Siemens `svs_se` series (`1x1x1x1024x64`), same machine,
same input:

| binary | stack reserve | result |
| --- | --- | --- |
| released wheel (MSVC) | 16,388,608 | `0xC00000FD`, 0 files |
| that binary, PE reserve patched | 16,777,216 | converts |
| built from source, GCC | 16,388,608 | converts |
| built from source, GCC | 16,777,216 | converts |

The third row is why this is described as a reserve that is too tight rather
than a wrong constant: 16,388,608 is enough for a GCC build. What overflows is
the wider stack frames MSVC emits. Windows fixes the reserve at link time and
cannot grow it, so the margin has to suit the widest build. macOS and Linux
grow or reserve more and never hit it.

## What this binary is

Built from `rordenlab/dcm2niix`, branch `development`, commit `fda9c11`, with
`console/CMakeLists.txt` asking for a true 16 MiB
(`-Wl,--stack,16777216`). Toolchain: GCC 16.2.0 (MinGW-w64 UCRT, WinLibs),
CMake + Ninja, `CMAKE_BUILD_TYPE=Release`. It reports

    Chris Rorden's dcm2niiX version v1.0.20260915  GCC16.2.0 x86-64 (64-bit Windows)

and its PE stack reserve is 16,777,216.

**It is a narrower build than the wheel's.** JPEG 2000 (OpenJPEG), JPEG-LS
(CharLS), TurboJPEG, Jasper and Zstandard are all off, because those are
`OFF` by default in that CMakeLists and were not needed for the failure this
fixes. It therefore cannot decode compressed transfer syntaxes that the
released binary handles.

## How it is used

Never in preference to the wheel. `classifier/dcm2niix_bidsguess.run_dcm2niix`
runs the pinned binary first and returns its result, and falls back here **only**
when the exit code is exactly `0xC00000FD`, which means the process was killed
before it could do anything. Any other failure is reported as itself, so a real
conversion error is never masked by a retry.

The fallback is Windows-only: `vendored_dcm2niix()` returns `None` elsewhere.

## When to delete this

When the fix lands upstream and the pinned `dcm2niix` wheel carries it. The
change is in `console/CMakeLists.txt`:

```cmake
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:0x1000000")
```

Submitted from `karellopez/dcm2niix`, branch
`fix/windows-msvc-stack-spectroscopy`. Once a released wheel reserves
16,777,216, this directory, its `pyproject.toml` package-data entry and the
fallback branch in `run_dcm2niix` should all go.

## Licence

dcm2niix is BSD 2-Clause, Chris Rorden and contributors. This is an unmodified
build of their source apart from the linker stack reserve described above; the
upstream `license.txt` applies unchanged and ships beside the binary as
`LICENSE`, since the BSD licence requires a binary redistribution to carry
its notice.
