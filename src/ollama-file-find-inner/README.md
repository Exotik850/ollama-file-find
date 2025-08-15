ollama-file-find (library)
==========================

Lightweight Rust utilities to enumerate and inspect locally installed [Ollama](https://ollama.com) models by reading the on‑disk `manifests` and `blobs` directories. Provides normalized model names, layer + config metadata, summed sizes, modification times, and verified blob file paths.

Core Features
-------------
* Pure local scan – no Ollama daemon API calls
* Mirrors (roughly) `ollama list` naming behavior (`ModelId::normalize`)
* Optional inclusion of hidden entries (namespaces / tags starting with `.`)
* Summed size + per‑layer size verification vs actual blob files
* Primary blob heuristic (largest declared layer or config)

Add to Cargo.toml (once published):
```toml
ollama-file-find = "0.1"
```
For workspace / path use:
```toml
ollama-file-find = { path = "src/ollama-file-find-inner" }
```

Quick Example
-------------
```rust
use ollama_file_find::{ollama_models_dir, scan_manifests, ScanArgs};

fn main() {
    let root = ollama_models_dir();
    let models = scan_manifests(ScanArgs {
        root: &root.join("manifests"),
        blobs_root: &root.join("blobs"),
        include_hidden: false,
        verbose: true,
    });
    for m in models {
        println!("{} total={:?}", m.name, m.total_size);
    }
}
```

Key Types & Functions
---------------------
* `ollama_models_dir() -> PathBuf` – resolve default models directory (`$OLLAMA_MODELS` or `$HOME/.ollama/models`).
* `ScanArgs { root, blobs_root, include_hidden, verbose }` – scan configuration.
* `scan_manifests(args) -> Vec<ListedModel>` – walk manifests and build model records.
* `ListedModel` – normalized name + optional verbose details: layers, config, total_size, mtime, primary + full blob path list.
* `LayerInfo`, `BlobPathInfo`, `ModelId` – supporting metadata structures (serde friendly).
* `digest_to_blob_path(blobs_root, digest)` – convert `sha256:abcd` to on‑disk `sha256-abcd` path.

Behavior Notes
--------------
* Directory layout expectation: `<models>/manifests/...` and `<models>/blobs/sha256-<hex>`.
* Hidden filtering: any component beginning with `.` skipped unless `include_hidden`.
* Sorting: output models alphabetically by normalized name.
* Resilience: unreadable / malformed manifests are logged to stderr and skipped.

Testing
-------
```bash
cargo test -p ollama-file-find
```

License
-------
Licensed under the [MIT License](https://github.com/Exotik850/ollama-file-find/blob/master/LICENSE.md)

See Also
--------
The companion CLI crate (`ollama-file-find-cli`) offers JSON / plain output for scripting.
