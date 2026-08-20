# Private Archives

This directory contains password-protected archives with the **non-representative**
content of the repository — everything that is intentionally NOT published on the
`main` branch (experiments data, notebooks, presentations, references, etc.).

## Password

All archives use the same password:

```
michalthesis1234
```

## Contents

| Archive | Contents |
|---|---|
| `private_data.zip` | `data/` (trained models, results, plots, GIFs) + `docs/plots/` |
| `private_presentations.zip` | `docs/defense_presentation/`, `docs/presentation/`, `docs/formatka-1-1/` |
| `private_notebooks.zip` | `notebooks/` (Jupyter notebooks, checkpoints, CSVs) |
| `private_references.zip` | `references/` (books and papers — PDF) |
| `private_archive.zip` | `archive/` (old notes) |

## Extract

```bash
unzip -P michalthesis1234 private_data.zip -d .
# or interactively:
unzip private_data.zip   # prompts for the password
```

## Important caveat

The git *history* of this repository still contains the plain (unencrypted)
versions of all these files. Anyone with full repository access (including
`main`'s history) could recover them without the password. The password
protection only hides them from the *current* `main` tree — treat this as
"access control for casual viewers", not as real secrecy. For true protection,
use a separate private repository or rewrite history.
