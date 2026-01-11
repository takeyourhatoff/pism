# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core C++ implementation (primarily `.cc`/`.hh`).
- `CMake/` and `CMakeLists.txt`: build system and shared CMake macros.
- `test/`: regression and verification tests (Python, shell); see `test/regression/`.
- `tests/`: additional standalone tests (e.g., `tests/test_debm.py`).
- `doc/sphinx/`: Sphinx-based manual and contributor docs.
- `examples/`: example configurations and workflows.
- `util/`, `docker/`, `images/`: helper scripts, container recipes, and assets.

## Build, Test, and Development Commands
PISM uses an out-of-source CMake build. A typical local build looks like:

```sh
mkdir -p build
cd build
CC=mpicc CXX=mpicxx cmake -DCMAKE_BUILD_TYPE=Debug -DPism_BUILD_EXTRA_EXECS=ON ..
make -j
```

Run the test suite from the build directory:

```sh
make test        # CTest runner
ctest --output-on-failure
```

Documentation can be rebuilt from the build directory with:

```sh
make manual_html
```

## Coding Style & Naming Conventions
- C++ formatting is defined in `.clang-format`; use `clang-format -i path/to/file.cc`.
- Indentation: 2 spaces per level, no tabs; braces follow the keyword (OTBS style).
- Prefer existing naming patterns and file extensions (`.cc`/`.hh`) in `src/`.

## Testing Guidelines
- Tests are registered via CTest in `test/CMakeLists.txt`.
- Many regression tests run with Python `nose`; enable `Pism_BUILD_PYTHON_BINDINGS=ON`
  and install `nose` to execute them.
- Add new regression tests under `test/regression/` (or `test/`) and wire them into
  `test/CMakeLists.txt`.

## Commit & Pull Request Guidelines
- Commit messages follow `.gitmessage`: imperative summary, <= 70 chars, no period;
  blank line before a wrapped (90 cols) body explaining what/why; include issue refs.
- PRs should include a clear description, tests run, and relevant docs updates.
- Update `CHANGES.rst` for user-facing changes or bug fixes.
- Allow edits from maintainers on the PR branch (per `CONTRIBUTING.rst`).