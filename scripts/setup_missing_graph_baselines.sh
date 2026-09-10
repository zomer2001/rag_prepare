#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
downloads_dir="${repo_root}/data/benchmarks/_downloads/baselines"
venvs_dir="${repo_root}/data/benchmarks/_venvs"

mkdir -p "${downloads_dir}" "${venvs_dir}"

if [[ ! -d "${downloads_dir}/graphrag/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/microsoft/graphrag.git "${downloads_dir}/graphrag"
  git -C "${downloads_dir}/graphrag" sparse-checkout set docs tests
  git -C "${downloads_dir}/graphrag" checkout
fi

if [[ ! -d "${downloads_dir}/raptor/.git" ]]; then
  git clone --depth 1 https://github.com/parthsarthi03/raptor.git \
    "${downloads_dir}/raptor"
fi

if [[ ! -x "${venvs_dir}/e2rag/bin/python" ]]; then
  uv venv "${venvs_dir}/e2rag" --python 3.11
fi
uv pip install --python "${venvs_dir}/e2rag/bin/python" \
  -r "${downloads_dir}/entityeventrag/requirements.txt"
uv pip install --python "${venvs_dir}/e2rag/bin/python" --no-deps \
  -e "${downloads_dir}/entityeventrag"

if [[ ! -x "${venvs_dir}/graphrag/bin/python" ]]; then
  uv venv "${venvs_dir}/graphrag" --python 3.11
fi
uv pip install --python "${venvs_dir}/graphrag/bin/python" "graphrag==3.1.2"

if [[ ! -x "${venvs_dir}/raptor/bin/python" ]]; then
  uv venv "${venvs_dir}/raptor" --python 3.11
fi
uv pip install --python "${venvs_dir}/raptor/bin/python" \
  "faiss-cpu" \
  "numpy==1.26.3" \
  "openai>=1.30,<2" \
  "scikit-learn>=1.4,<2" \
  "sentence-transformers==2.2.2" \
  "tenacity>=8.2,<9" \
  "tiktoken>=0.5.1,<1" \
  "torch" \
  "transformers==4.38.1" \
  "umap-learn==0.5.5"

"${venvs_dir}/graphrag/bin/python" -c \
  "import graphrag; from importlib.metadata import version; print(version('graphrag'))"
"${venvs_dir}/e2rag/bin/python" -c \
  "import sys,types; m=types.ModuleType('oracledb'); m.defaults=types.SimpleNamespace(fetch_lobs=True); m.DB_TYPE_VECTOR=object(); sys.modules['oracledb']=m; import entity_event_rag; print('E2RAG import ready')"
"${venvs_dir}/raptor/bin/python" -c \
  "import numpy, openai, sklearn, tiktoken; print('RAPTOR adapter dependencies ready')"

printf 'E2RAG, GraphRAG, and RAPTOR environments are ready. KG2RAG remains intentionally excluded.\n'
