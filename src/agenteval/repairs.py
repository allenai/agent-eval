from importlib import import_module
import sys


def repair(name: str):
    """
    """
    private_astabench_repairs = import_module("astabench.private_repairs")
    astabench_repairs = import_module("astabench.repairs")

    for v in private_astabench_repairs.REPAIRS.values():
        print("private")
        v("private")
        v(name)

    for v in astabench_repairs.REPAIRS.values():
        print("public")
        v("public")
        v(name)
