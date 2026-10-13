# Colab runtime and persistence contract

The Colab notebook is a small GPT-2 implementation pilot. It prepares pinned
LOTUS data, trains a selected subset, resumes saved work, and evaluates held-out
validation questions. Running this notebook is a prerequisite to larger
experiments; it does not establish a benchmark result or an anchoring benefit.

The public repository is `https://github.com/sttruji/dynamic-state-anchoring`.
The notebook's first cell clones this repository without a GitHub token.
A private fork needs authenticated Git clone access configured
separately; keep credentials out of the saved notebook.

## Dependency policy

Install `requirements-colab.txt`, which pins the project's Transformers 4.46.2,
Tokenizers 0.20.3, Hugging Face Hub 0.36.2, and YAML interface. It deliberately
does not install PyTorch, torchvision, torchaudio, NVIDIA packages, or the local
CPU environment lock. PyTorch installation must match the CUDA platform, so
the notebook retains the managed runtime's existing GPU stack. See the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).

The notebook creates `/content/dsa-venv` with `--without-pip --system-site-packages`,
allowing access to that existing stack while installing DSA's Hugging Face pins
into the virtual environment. It uses Colab's existing pip with an explicit
`--python /content/dsa-venv/bin/python` target, so creation does not depend on
`ensurepip` being available. This is a supported pip workflow for pip-less
environments; see [pip's interpreter targeting guide](https://pip.pypa.io/en/stable/topics/python-option/).
The setup helper validates and repairs partial environments instead of treating
an existing `bin/python` as proof of successful installation. Training and verification use this environment's Python
through subprocesses. The notebook kernel itself uses the standard library and
the Colab Drive helper. This avoids changing the global Transformers/Hub packages
used by other preinstalled tools and avoids a kernel restart after installation.

The runtime guard and environment report must check the Python and PyTorch
versions, CUDA availability, GPU name, and the actual installed dependencies.
PyTorch 2.7.1 is the locally tested CPU version. A version guard admitting newer
PyTorch versions is not a claim that every such version has been GPU tested.
The notebook must exercise the imported GPT-2 implementation before training.

NumPy is allowed in the range `>=1.26,<3`, retaining a compatible preinstalled
version. The code uses NumPy for RNG state and does not need a NumPy upgrade.
The project's Transformers pin accepts Tokenizers `>=0.20,<0.21` and Hugging Face
Hub `>=0.23.2,<1.0`, so the selected pins satisfy its published dependency
constraints. This is a DSA environment, not an environment for every preinstalled
Colab ML package. See the [versioned Transformers dependency table](https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/dependency_versions_table.py).

Google announced the Python 3.13 rollout on August 18, 2026 and reported it live
on August 25. The runtime guard now accepts Python 3.10–3.13. The earlier
2026.07 inventory described Python 3.12 and is a historical snapshot, so the
notebook reports the actual runtime instead of assuming that version.
Tokenizers 0.20.3 and PyYAML 6.0.3 publish Python 3.13 Linux wheels; no Hugging Face
pin change is needed to address environment creation.
See [Google's upgrade announcement](https://github.com/googlecolab/colabtools/issues/6081).
See [Google's backend inventory](https://github.com/googlecolab/backend-info)
and [Tokenizers release files](https://pypi.org/project/tokenizers/0.20.3/).

## Persistence and reuse

Use a hosted Colab GPU runtime and mount Google Drive through `google.colab.drive`.
Authorize access in Google's mount prompt. Choose one project subdirectory under
`MyDrive` for data, model downloads, reports, and training checkpoints. Google's
FAQ explains that runtimes are temporary, their GPU allocation varies, and
mounted Drive can be slower than local VM storage. The notebook therefore keeps
the code checkout and active work locally while preserving reusable inputs and
saved training state on Drive. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Set `HF_HOME` before importing Transformers or Hugging Face Hub. Keep the pinned
model snapshot in persistent storage. Subsequent sessions reuse verified corpus
files and model files, then stage working inputs locally; they do not download
the full corpus again. Cache reuse must still check the recorded identity and
hashes. A mismatch must stop, preserving the existing file for inspection.
Hugging Face documents the cache environment and offline/local-file behavior in
its [versioned installation guide](https://huggingface.co/docs/transformers/v4.46.2/installation).

Checkpoint after a bounded number of updates and resume from the last complete
checkpoint. An interrupted update after that checkpoint may need to be repeated.
Save the first cell's resolved `REPO_COMMIT`; set `REPO_REF` to that commit when
reconnecting. The default `main` branch can advance after new code is published.
Keep run settings and input identities fixed when resuming; use a separate run
directory when changing the dataset subset, seed, write mode, loops, or losses.
Exact resume also checks the device type/name, package versions, CUDA/cuDNN,
and implementation hashes. A new session with different GPU hardware or a
changed runtime can require a fresh run; deterministic continuation across
different hardware is not claimed. The default retains the latest two complete
checkpoints. Reserve several GB for model weights and Adam state per run.
Use validation for pilot decisions and leave official test evaluation explicitly
disabled until the experimental settings are frozen.

## Scope of verification

Local tests check notebook syntax and its workflow contract, dependency policy,
data reuse, and training/evaluation integration. The notebook checks also run its
embedded runtime preflight on CPU and execute its disabled-test/summary cells
against fixtures, verifying that their persisted reports are valid JSON.
The complete 186-test suite passes on local Python 3.12 and Python 3.13 CPU
environments. The Python 3.13 setup command also completed a real installation
of the pinned requirements into a pip-less environment. These checks exercise
the environment repair but do not reproduce Colab's GPU image.
No hosted Colab session, Google Drive mount, or
GPU execution is implied by these checks. Start with the small FP32,
one-record-per-update pilot and inspect measured speed and peak memory before
increasing the training size or loop budget.
