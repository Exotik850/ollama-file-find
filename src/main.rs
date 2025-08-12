use anyhow::Result;
use clap::Parser;
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

mod args;
use args::Args;
mod models;
use models::{ListedModel, ManifestJson, LayerInfo, BlobPathInfo};

fn main() -> Result<()> {
    let args = Args::parse();
    let models_dir = resolve_models_dir(args.models_dir.as_deref());
    let manifests_root = models_dir.join("manifests");
    let blobs_root = models_dir.join("blobs");

    if !manifests_root.is_dir() {
        anyhow::bail!(
            "Manifests directory not found: {}",
            manifests_root.display()
        );
    }

    let mut models = Vec::new();
    scan_manifests(
        &manifests_root,
        &blobs_root,
        &models_dir,
        &args,
        &mut models,
    )?;

    // Sort for deterministic output (by display name)
    models.sort_by(|a, b| a.name.cmp(&b.name));

    if args.plain && !args.blob_paths && !args.verbose {
        for m in &models {
            println!("{}", m.name);
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&models)?);
    }

    Ok(())
}

/// Locate the models directory (`OLLAMA_MODELS` or fallback to $HOME/.ollama/models)
fn resolve_models_dir(override_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = override_dir {
        return p.to_path_buf();
    }
    if let Ok(p) = env::var("OLLAMA_MODELS")
        && !p.is_empty() {
            return PathBuf::from(p);
        }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".ollama").join("models")
}

/// Scan manifests and construct `ListedModel` entries.
fn scan_manifests(
    root: &Path,
    blobs_root: &Path,
    models_dir: &Path,
    args: &Args,
    out: &mut Vec<ListedModel>,
) -> Result<()> {
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Skipping entry error: {e}");
                continue;
            }
        };
        if entry.file_type().is_dir() {
            continue;
        }

        let rel = match entry.path().strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let comps: Vec<_> = rel
            .iter()
            .map(|c| c.to_string_lossy().to_string())
            .collect();
        if comps.is_empty() {
            continue;
        }

        // Accept 4 components host/namespace/model/tag or 3 components namespace/model/tag
        if !(comps.len() == 4 || comps.len() == 3) {
            continue;
        }

        let tag = comps.last().unwrap();
        if !args.include_hidden && tag.starts_with('.') {
            continue;
        }
        if !args.include_hidden && comps[..comps.len() - 1].iter().any(|c| c.starts_with('.')) {
            continue;
        }

        let (host, namespace, model, tag) = if comps.len() == 4 {
            (
                Some(comps[0].clone()),
                Some(comps[1].clone()),
                comps[2].clone(),
                comps[3].clone(),
            )
        } else {
            (
                None,
                Some(comps[0].clone()),
                comps[1].clone(),
                comps[2].clone(),
            )
        };

        let display_name = normalize_name(host.as_deref(), namespace.as_deref(), &model, &tag);

        // Read manifest JSON
        let manifest_path = entry.path();
        let data = match fs::read(manifest_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed reading manifest {}: {e}", manifest_path.display());
                continue;
            }
        };

        let parsed: ManifestJson = match serde_json::from_slice(&data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "Skipping invalid manifest JSON {}: {e}",
                    manifest_path.display()
                );
                continue;
            }
        };

        let layers: Vec<LayerInfo> = parsed
            .layers
            .iter()
            .cloned()
            .map(|l| LayerInfo {
                digest: l.digest,
                media_type: l.media_type,
                size: l.size,
            })
            .collect();

        let config = parsed.config.map(|c| LayerInfo {
            digest: c.digest,
            media_type: c.media_type,
            size: c.size,
        });

        // Compute total size if verbose
        let total_size = if args.verbose {
            let mut sum = 0u64;
            let mut any = false;
            for l in &layers {
                if let Some(sz) = l.size {
                    sum += sz;
                    any = true;
                }
            }
            if let Some(cfg) = &config
                && let Some(sz) = cfg.size {
                    sum += sz;
                    any = true;
                }
            if any { Some(sum) } else { None }
        } else {
            None
        };

        let mtime = if args.verbose {
            fs::metadata(manifest_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
        } else {
            None
        };

        // Blob paths (optional)
        let (primary_blob_path, blob_paths) = if args.blob_paths {
            let (primary_digest, mut blob_infos) =
                build_blob_infos(&layers, config.as_ref(), blobs_root);

            let primary_blob_path = primary_digest
                .clone()
                .map(|d| digest_to_blob_path(blobs_root, &d));

            // annotate primary
            if let Some(pd) = primary_digest {
                for bi in &mut blob_infos {
                    if bi.digest == pd {
                        bi.primary = true;
                    }
                }
            }

            (primary_blob_path, Some(blob_infos))
        } else {
            (None, None)
        };

        out.push(ListedModel {
            name: display_name,
            host,
            namespace,
            model,
            tag,
            manifest_path: manifest_path.display().to_string(),
            layers: if args.verbose { Some(layers) } else { None },
            config,
            total_size,
            mtime,
            primary_blob_path,
            blob_paths,
        });
    }

    Ok(())
}

/// Build blob path info list and decide primary digest.
fn build_blob_infos(
    layers: &[LayerInfo],
    config: Option<&LayerInfo>,
    blobs_root: &Path,
) -> (Option<String>, Vec<BlobPathInfo>) {
    // Heuristic: primary = largest non-config layer with a size.
    let mut primary_digest: Option<String> = None;
    let mut max_size: u64 = 0;

    for l in layers {
        if let Some(sz) = l.size
            && sz > max_size {
                max_size = sz;
                primary_digest = Some(l.digest.clone());
            }
    }
    // Fallback to first layer if no sizes
    if primary_digest.is_none() {
        if let Some(first) = layers.first() {
            primary_digest = Some(first.digest.clone());
        } else if let Some(cfg) = config {
            primary_digest = Some(cfg.digest.clone());
        }
    }

    let mut out = Vec::new();

    for l in layers {
        out.push(build_blob_path_info(l, blobs_root, false));
    }
    if let Some(cfg) = config {
        out.push(build_blob_path_info(cfg, blobs_root, false));
    }

    (primary_digest, out)
}

fn build_blob_path_info(l: &LayerInfo, blobs_root: &Path, primary: bool) -> BlobPathInfo {
    let path = digest_to_blob_path(blobs_root, &l.digest);
    let (exists, actual_size, size_ok) = match fs::metadata(&path) {
        Ok(meta) => {
            let a = meta.len();
            let ok = l.size.map(|decl| decl == a);
            (true, Some(a), ok)
        }
        Err(_) => (false, None, None),
    };
    BlobPathInfo {
        digest: l.digest.clone(),
        media_type: l.media_type.clone(),
        declared_size: l.size,
        path: path.display().to_string(),
        exists,
        size_ok,
        actual_size,
        primary,
    }
}

fn digest_to_blob_path(blobs_root: &Path, digest: &str) -> PathBuf {
    // Expect "sha256:abcdef..."
    // Ollama stores as "sha256-abcdef..."
    if let Some(rest) = digest.strip_prefix("sha256:") {
        blobs_root.join(format!("sha256-{rest}"))
    } else {
        // Fallback: direct join (unusual)
        blobs_root.join(digest.replace(':', "-"))
    }
}

/// Attempt to mirror Ollama list naming rules
fn normalize_name(host: Option<&str>, namespace: Option<&str>, model: &str, tag: &str) -> String {
    let default_host = "registry.ollama.ai";
    let library_ns = "library";
    match (host, namespace) {
        (Some(h), Some(ns)) if h == default_host && ns == library_ns => format!("{model}:{tag}"),
        (Some(h), Some(ns)) if h == default_host => format!("{ns}/{model}:{tag}"),
        (None, Some(ns)) if ns == library_ns => format!("{model}:{tag}"),
        (None, Some(ns)) => format!("{ns}/{model}:{tag}"),
        (Some(h), Some(ns)) => format!("{h}/{ns}/{model}:{tag}"),
        _ => format!("{model}:{tag}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert_eq!(
            normalize_name(Some("registry.ollama.ai"), Some("library"), "mistral", "7b"),
            "mistral:7b"
        );
        assert_eq!(
            normalize_name(
                Some("registry.ollama.ai"),
                Some("apple"),
                "OpenELM",
                "latest"
            ),
            "apple/OpenELM:latest"
        );
        assert_eq!(
            normalize_name(Some("myhost"), Some("myns"), "lips", "code"),
            "myhost/myns/lips:code"
        );
        assert_eq!(
            normalize_name(None, Some("library"), "phi4", "latest"),
            "phi4:latest"
        );
    }

    #[test]
    fn test_digest_to_blob_path() {
        let root = PathBuf::from("/tmp/blobs");
        let p = digest_to_blob_path(&root, "sha256:1234abcd");
        assert_eq!(p, PathBuf::from("/tmp/blobs/sha256-1234abcd"));
    }
}
