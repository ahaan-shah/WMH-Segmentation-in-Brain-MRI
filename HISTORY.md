# A note on this repository's git history

On **2026-09-16** the history was rewritten once, to remove model checkpoint
files (`*.pt`) that had been committed in Weeks 3 and 4 before a `.gitignore`
rule was added for them. This note exists so that anyone tracing a provenance
record can resolve it.

## Why

Four commits carried 5 checkpoint blobs, about 160 MB. Model weights are
regenerable from the committed code plus the frozen seed in
`code/metadata/dataset.yaml`, so they are a local cache rather than a
reproducibility artefact — the training *histories* and the *scores* are the
evidence, and those are committed. The blobs had never been pushed, so the
rewrite affected nothing outside this machine.

Result: the repository went from **180 MB to 16 MB**.

## What was verified afterwards

Checked against a tarball of the pre-rewrite `.git`, not assumed:

- **22 commits before, 22 after.** None dropped, none added.
- **Every commit message identical.**
- **Author, email, author timestamp, committer and parent structure identical**
  on all 22.
- **The final tree is byte-for-byte identical** — all 206 files, same content
  hashes.
- Across every commit: **zero files added, zero file contents changed.** The
  only removals were the 8 `.pt` entries.
- **No `.pt` object is reachable anywhere** in the rewritten repository.
- The 92-test suite passes, and the working tree matches `HEAD` exactly.

## The consequence, and how to resolve it

Rewriting changes commit hashes from the first affected commit onward. Every
generated artefact in this project carries a provenance sidecar recording the
commit it was produced at (see `code/metadata/provenance.py`), so **143 sidecars
now name commits that no longer exist** — 23 in tracked files, 120 in the
gitignored `data/` tree.

Those artefacts were **not** regenerated. Several are training histories and
leave-one-site-out results whose regeneration would mean hours of retraining for
no change in content, and the underlying trees are provably identical anyway.

Instead, the mapping is recorded here. Five hashes appear in sidecars and are
affected:

| Sidecar records | Now lives at |
|---|---|
| `237f7cbf3e...` | `36da83d46a...` |
| `ab3d801607...` | `33f472764c...` |
| `63dce046ad...` | `6432ee8210...` |
| `b6208137f5...` | `dc5cafa69b...` |
| `46b8eb2dcf...` | `de37bdcb32...` |

The complete 22-entry table is in `.git/filter-repo/commit-map` on the machine
where the rewrite ran. Any sidecar hash not in the list above was unaffected and
still resolves directly.

A `-dirty` suffix on a recorded hash means the working tree had uncommitted
changes when the artefact was written. For figures that is expected rather than
a fault: writing the figure is itself the change that makes the tree dirty.

## Going forward

`*.pt` and the two checkpoint directories are in `.gitignore`. The checkpoints
remain on disk and are unaffected by any of this; only git's copy of them was
removed. No further rewrite is planned, and none should be needed — the
artefacts that grow from here (`data/`, checkpoints) are all ignored.
