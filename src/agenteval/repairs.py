from importlib import import_module

from pydantic import BaseModel

from agenteval.leaderboard.models import InterventionInfo


CHANGES = {"say-hi": lambda x: print(f"from agent-eval (inner): hi {x}")}


class RegistryEntry(BaseModel):
    registry: str
    name: str

    @staticmethod
    def from_str(a_str):
        sep = ":"
        [registry, name] = a_str.split(sep)
        return RegistryEntry(registry=registry, name=name)


class Registry:
    def __init__(self, registry_pointer_strs: list[str]):
        self.registry = {"agenteval": CHANGES}

        registry_entries = [RegistryEntry.from_str(p) for p in registry_pointer_strs]
        for entry in registry_entries:
            assert entry.registry not in self.registry, "Multiple change registry entries with the same name."
            self.registry[entry.registry] = import_module(entry.name).CHANGES

    def find_change(self, change_pointer: InterventionInfo):
        return self.registry.get(change_pointer.registry, {}).get(change_pointer.name)



def repair(name: str, intervention_pointer_strs: list[str], registry_pointer_strs: list[str]):
    """
    """
    registry = Registry(registry_pointer_strs)
    intervention_pointers = [InterventionInfo.from_str(p) for p in intervention_pointer_strs]
    for intervention_pointer in intervention_pointers:
        maybe_change = registry.find_change(intervention_pointer)
        if maybe_change is not None:
            maybe_change(name)
        else:
            print(f"Unable to find change {intervention_pointer}.")
