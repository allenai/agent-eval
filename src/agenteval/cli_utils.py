import click


def generate_choice_help(mapping, base_help=""):
    choices_text = ", ".join([f"{alias} ({full})" for alias, full in mapping.items()])
    return (
        f"{base_help} Options: {choices_text}"
        if base_help
        else f"Options: {choices_text}"
    )


class AliasedChoice(click.Choice):
    def __init__(self, choices_map):
        # maps short aliases to full strings
        self.choices_map = choices_map
        super().__init__(list(choices_map.keys()), case_sensitive=False)

    def convert(self, value, param, ctx):
        try:
            alias = super().convert(value, param, ctx)
            return self.choices_map[alias.lower()]
        except click.BadParameter:
            formatted_choices = ", ".join(
                f"{k} ({v})" for k, v in self.choices_map.items()
            )
            self.fail(
                f"Choose from: {formatted_choices})",
                param,
                ctx,
            )

    def get_missing_message(self, param):
        formatted_choices = ", ".join(f"{k} ({v})" for k, v in self.choices_map.items())
        return f"Choose from: {formatted_choices}"
