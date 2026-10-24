# Model files and adapters

The base model files are pinned by revision, size and SHA-256 in each family's `base_manifest.json`. Tokenizer/configuration files are included; the approximately30GB of base-model weight shards are fetched from the pinned upstream repositories rather than stored as ordinary Git objects.

The repository's private [model-adapters-v1 release](https://github.com/ArnabsHomePage/agent-beliefs-and-manager-decisions/releases/tag/model-adapters-v1) contains seven safetensors archives, including the retained training-stage adapters and canonical serving export. `release_assets.json` lists every archive and member hash. These archives contain only model weights and relative member names.

```sh
cd "YOUR PATHWAY"
python tools/fetch_models.py --base both
# Download the seven private release assets, then supply their directory:
python tools/fetch_models.py --adapters --asset-directory "YOUR PATHWAY"
# Or use an already-authorized GitHub CLI connection:
python tools/fetch_models.py --adapters --asset-directory "YOUR PATHWAY" --use-gh
```

The fetcher verifies bytes before replacing a destination and restores shared serving paths from identical checkpoints. The 400 instance definitions reference canonical weights; no weight file is duplicated200times. Mistral V8 inference uses adapter scale0.5, while manager inference uses0.9. Serving configs encode the corresponding alpha values without changing rank or checkpoint tensors.

Mistral adapters are under `v8_article`, `phase10_qa02_03`, `full_qa01_04_05`, `serving` and `v8_serving`. Qwen adapters are under `article`, `qa02_03`, `full_qa01_04_05` and `serving`. The Qwen B raw trained checkpoint is preserved; the serving directory references the filtered canonical export required by vLLM.

A configuration's `base_model_name_or_path` is `YOUR PATHWAY`. The runtime supplies the actual base path explicitly, so no private workstation or VM path is needed.

GitHub blocks normal Git files above100MiB; the adapter archives use its supported release-asset mechanism ([GitHub documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)). No storage plan or paid subscription has been changed.

The model cards and Qwen LICENSE/NOTICE files retain their upstream terms. Mistral's model card declares Apache2.0. This repository does not grant a new license to third-party model weights or article material.
