"""Load the learner's implementation or the reference, selected by the LAB_IMPL env var.

LAB_IMPL unset or "student" -> <lab>/<student_file>.py   (yours; AI-off)
LAB_IMPL=reference          -> <lab>/reference.py
"""
import importlib.util
import os
import pathlib


def load(lab_dir: pathlib.Path, student_file: str):
    which = os.environ.get("LAB_IMPL", "student").lower()
    name = "reference" if which == "reference" else student_file
    path = lab_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{lab_dir.name.replace('-', '_')}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_file(path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
