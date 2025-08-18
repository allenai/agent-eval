import click
from dataclasses import dataclass


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


@dataclass
class RepoPathsOfInterest:
    repo_id: str
    relative_paths: list[str]

    @staticmethod
    def from_urls(urls: list[str]):
        repo_ids = set()
        paths = []

        for url in urls:
            # validates submission_url format "hf://<repo_id>/<path>"
            repo_id, path = parse_hf_url(url)
            repo_ids.add(repo_id)
            paths.append(path)

        if len(repo_ids) > 1:
            raise Exception("All URLs must reference the same repo")

        repo_id_to_use = repo_ids.pop()

        return RepoPathsOfInterest(
            repo_id=repo_id_to_use,
            relative_paths=list(set(paths)),
        )
