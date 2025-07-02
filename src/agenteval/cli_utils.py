import click

def generate_choice_help(mapping, base_help=""):
    choices_text = ", ".join([f"{alias} ({full})" for alias, full in mapping.items()])
    return f"{base_help} Options: {choices_text}" if base_help else f"Options: {choices_text}"

class AliasedChoice(click.Choice):
    def __init__(self, choices_map):
        # maps short aliases to full strings
        self.choices_map = choices_map
        super().__init__(list(choices_map.keys()), case_sensitive=False)
    
    def convert(self, value, param, ctx):
        # validates alias
        alias = super().convert(value, param, ctx)
        # returns full string
        return self.choices_map[alias.lower()]