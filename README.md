ollama-file-find
=================

List and inspect locally installed [Ollama](https://ollama.com) models by directly reading the on‑disk `manifests` and `blobs` directories. Ships as:

* A CLI (`ollama-file-find-cli`) – fast, scriptable inventory of models.
* A library crate (`ollama-file-find`) – reusable scanning + path utilities.

Why? The official `ollama list` command gives a minimal view. This tool lets you:

* Enumerate every manifest (optionally including hidden tags that start with `.`)
* Obtain normalized model names mirroring Ollama's own display rules
* See per‑layer + config digests, declared sizes, summed size, and modification time
* Map digests to actual blob file paths and verify existence & size consistency

--------------------------------------------------
Quick Start (CLI)
--------------------------------------------------

Install (from this repo clone):

```
cargo install --path . --locked
```

Run:

```
ollama-file-find --help
```

Typical usage:

```
# Plain list of model names (default JSON suppressed with --plain)
ollama-file-find --plain

# Verbose JSON with layer + blob path info
ollama-file-find --verbose

# Include hidden tags (namespaces / tags beginning with a dot)
ollama-file-find --include-hidden --verbose

# Point at a non‑default models directory
ollama-file-find --models-dir "D:/Other/Ollama/models" --plain
```

Exit codes: non‑zero only on argument / IO errors (e.g. missing manifests directory).

--------------------------------------------------
Environment & Directory Resolution
--------------------------------------------------

The root models directory is resolved as:
1. `--models-dir <path>` if provided
2. `$OLLAMA_MODELS` if set and non‑empty
3. `$HOME/.ollama/models`

Within that root, the tool expects:
* `manifests/` – nested directories whose leaf files are JSON manifests
* `blobs/` – content blobs named like `sha256-<hex>`

--------------------------------------------------
CLI Output Formats
--------------------------------------------------

1. Plain text (`--plain` without `--verbose`): one normalized model name per line.
2. JSON array (default): each element is a `ListedModel` object (see schema below). If `--plain` is combined with `--verbose`, JSON is still emitted (because verbose details cannot be expressed in plain list form).

Example (plain):

```
mistral:7b
llama3:8b
apple/OpenELM:latest
```

Example (verbose JSON excerpt):

```json
[
	{
		"name": "mistral:7b",
		"host": "registry.ollama.ai",
		"namespace": "library",
		"model": "mistral",
		"tag": "7b",
		"manifest_path": "/home/user/.ollama/models/manifests/library/mistral/7b",
		"layers": [
			{ "digest": "sha256:…", "mediaType": "application/vnd.ollama.image.layer", "size": 123456789 }
		],
		"config": { "digest": "sha256:…", "mediaType": "application/vnd.ollama.image.config", "size": 1234 },
		"total_size": 123458023,
		"mtime": 1723590123,
		"primary_blob_path": "/home/user/.ollama/models/blobs/sha256-abcd…",
		"blob_paths": [
			{
				"digest": "sha256:…",
				"media_type": "application/vnd.ollama.image.layer",
				"declared_size": 123456789,
				"path": "/home/user/.ollama/models/blobs/sha256-abcd…",
				"exists": true,
				"size_ok": true,
				"actual_size": 123456789,
				"primary": true
			}
		]
	}
]
```

--------------------------------------------------
Library Overview
--------------------------------------------------

Add to your `Cargo.toml` (when published):

```
ollama-file-find = "0.1"
```

Currently (in this repo workspace) depend via path:

```
ollama-file-find = { path = "src/ollama-file-find-inner" }
```

Core API surface (simplified signatures):

* `fn ollama_models_dir() -> PathBuf` – resolve default models directory.
* `struct ScanArgs<'a> { root: &'a Path, blobs_root: &'a Path, include_hidden: bool, verbose: bool }`
* `fn scan_manifests(args: ScanArgs) -> Vec<ListedModel>` – walk `root`, parse manifests, compute optional details.
* `fn build_blob_infos(layers: &[LayerInfo], config: Option<LayerInfo>, blobs_root: &Path) -> (Option<String>, Vec<BlobPathInfo>)` – derive primary digest + blob info list.
* `fn build_blob_path_info(l: &LayerInfo, blobs_root: &Path) -> BlobPathInfo` – single layer/config mapping.
* `fn digest_to_blob_path(blobs_root: &Path, digest: &str) -> PathBuf` – convert `sha256:abcd` to on‑disk path `sha256-abcd`.

Data structures (selected fields):

* `ModelId { host: Option<String>, namespace: Option<String>, model: String, tag: String }` – plus `normalize()` for display name.
* `LayerInfo { digest: String, media_type: String, size: Option<u64> }`
* `BlobPathInfo { digest, media_type, declared_size, path, exists, size_ok, actual_size, primary }`
* `ListedModel { name, model_id parts, manifest_path, layers?, config?, total_size?, mtime?, primary_blob_path?, blob_paths? }`

Minimal library example:

```rust
use ollama_file_find::{ollama_models_dir, scan_manifests, ScanArgs};

fn main() {
		let models_root = ollama_file_find::ollama_models_dir();
		let manifests = models_root.join("manifests");
		let blobs = models_root.join("blobs");
		let models = scan_manifests(ScanArgs {
				root: &manifests,
				blobs_root: &blobs,
				include_hidden: false,
				verbose: true,
		});
		for m in models { println!("{} -> {:?}", m.name, m.total_size); }
}
```

--------------------------------------------------
Behavior & Notes
--------------------------------------------------

* Hidden filtering: any path component (namespace / model / tag) starting with `.` is skipped unless `include_hidden`.
* Component parsing accepts either `host/namespace/model/tag` (4) or `namespace/model/tag` (3) directory components under `manifests/`.
* Sorting: output is sorted lexicographically by normalized name.
* Size computation: sum of declared layer sizes (+ config) when available; omitted if no sizes present.
* Modification time (`mtime`): manifest file mtime (POSIX seconds since epoch); may differ from blob modification times.
* Primary blob heuristic: largest declared size layer; if none have size, falls back to config digest (if present).
* Error tolerance: unreadable entries or malformed JSON are skipped with stderr diagnostics; overall scan continues.

--------------------------------------------------
Testing
--------------------------------------------------

Run all tests:

```
cargo test --all
```

--------------------------------------------------
Troubleshooting
--------------------------------------------------

Issue: "Manifests directory not found" – Ensure Ollama is installed and has pulled at least one model, or specify the correct `--models-dir`.

Blob sizes mismatching (`size_ok: false`): partial downloads or corruption; re‑pull the model via `ollama pull <model>`.

Empty output: no manifest files detected (e.g. wrong directory, or only hidden tags without `--include-hidden`).

--------------------------------------------------
Roadmap / Ideas
--------------------------------------------------

* Publish crates.io package
* Optional output formats (table, CSV)
* Parallel blob metadata probing
* Filtering by namespace / tag pattern

--------------------------------------------------
License
--------------------------------------------------

This project is license under the [MIT License](./LICENSE.md)

--------------------------------------------------
Acknowledgements
--------------------------------------------------

Not affiliated with Ollama. Names & paths based on public on‑disk layout.

