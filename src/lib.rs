use anyhow::Result;
use clap::Parser;
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

mod models;
use models::{BlobPathInfo, LayerInfo, ListedModel, ManifestJson};

/// Locate the models directory (`OLLAMA_MODELS` or fallback to $HOME/.ollama/models)
pub fn resolve_models_dir(override_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = override_dir {
        return p.to_path_buf();
    }
    if let Ok(p) = env::var("OLLAMA_MODELS")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".ollama").join("models")
}

pub struct ScanArgs<'a> {
    pub root: &'a Path,
    pub blobs_root: &'a Path,
    pub include_hidden: bool,
    pub verbose: bool,
    pub blob_paths: bool,
}

fn relative_components(entry: &walkdir::DirEntry, root: &Path) -> Option<Vec<String>> {
    let rel = entry.path().strip_prefix(root).ok()?;
    let comps: Vec<String> = rel
        .iter()
        .map(|c| c.to_string_lossy().to_string())
        .collect();
    if comps.is_empty() {
        return None;
    }
    Some(comps)
}

fn parse_components(
    mut comps: Vec<String>,
    include_hidden: bool,
) -> Option<(Option<String>, Option<String>, String, String)> {
    // Accept 4 components host/namespace/model/tag or 3 components namespace/model/tag
    if !(comps.len() == 4 || comps.len() == 3) {
        return None;
    }

    let tag = comps.last().unwrap();
    if !include_hidden && tag.starts_with('.') {
        return None;
    }
    if !include_hidden && comps[..comps.len() - 1].iter().any(|c| c.starts_with('.')) {
        return None;
    }

    let (host, namespace, model, tag) = if comps.len() == 4 {
        (
            Some(std::mem::take(&mut comps[0])),
            Some(std::mem::take(&mut comps[1])),
            std::mem::take(&mut comps[2]),
            std::mem::take(&mut comps[3]),
        )
    } else {
        (
            None,
            Some(std::mem::take(&mut comps[0])),
            std::mem::take(&mut comps[1]),
            std::mem::take(&mut comps[2]),
        )
    };

    Some((host, namespace, model, tag))
}

fn read_manifest(path: &Path) -> Option<ManifestJson> {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed reading manifest {}: {e}", path.display());
            return None;
        }
    };
    match serde_json::from_slice(&data) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping invalid manifest JSON {}: {e}", path.display());
            None
        }
    }
}

fn manifest_to_layers(parsed: &ManifestJson) -> (Vec<LayerInfo>, Option<LayerInfo>) {
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

    let config = parsed.config.as_ref().map(|c| LayerInfo {
        digest: c.digest.clone(),
        media_type: c.media_type.clone(),
        size: c.size,
    });

    (layers, config)
}

fn compute_total_size(layers: &[LayerInfo], config: Option<&LayerInfo>) -> Option<u64> {
    let mut sum = 0u64;
    let mut any = false;
    for l in layers {
        if let Some(sz) = l.size {
            sum += sz;
            any = true;
        }
    }
    if let Some(cfg) = config {
        if let Some(sz) = cfg.size {
            sum += sz;
            any = true;
        }
    }
    if any { Some(sum) } else { None }
}

fn compute_mtime(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

fn gather_blob_info(
    layers: &[LayerInfo],
    config: Option<&LayerInfo>,
    blobs_root: &Path,
) -> (Option<PathBuf>, Vec<BlobPathInfo>) {
    let (primary_digest, mut blob_infos) = build_blob_infos(layers, config, blobs_root);
    let primary_blob_path = primary_digest
        .as_ref()
        .map(|d| digest_to_blob_path(blobs_root, d));
    if let Some(pd) = primary_digest {
        for bi in &mut blob_infos {
            if bi.digest == pd {
                bi.primary = true;
            }
        }
    }
    (primary_blob_path, blob_infos)
}

fn build_listed_model(
    host: Option<String>,
    namespace: Option<String>,
    model: String,
    tag: String,
    manifest_path: &Path,
    layers: Vec<LayerInfo>,
    config: Option<LayerInfo>,
    verbose: bool,
    blobs_root: &Path,
    want_blob_paths: bool,
) -> ListedModel {
    let display_name = normalize_name(host.as_deref(), namespace.as_deref(), &model, &tag);

    let total_size = if verbose {
        compute_total_size(&layers, config.as_ref())
    } else {
        None
    };

    let mtime = if verbose {
        compute_mtime(manifest_path)
    } else {
        None
    };

    let (primary_blob_path, blob_paths) = if want_blob_paths || verbose {
        gather_blob_info(&layers, config.as_ref(), blobs_root)
    } else {
        (None, Vec::new())
    };

    ListedModel {
        name: display_name,
        host,
        namespace,
        model,
        tag,
        manifest_path: manifest_path.display().to_string(),
        layers: if verbose { Some(layers) } else { None },
        config,
        total_size,
        mtime,
        primary_blob_path,
        blob_paths: if blob_paths.is_empty() {
            None
        } else {
            Some(blob_paths)
        },
    }
}

fn process_entry(
    entry: &walkdir::DirEntry,
    root: &Path,
    blobs_root: &Path,
    include_hidden: bool,
    verbose: bool,
    want_blob_paths: bool,
) -> Option<ListedModel> {
    if entry.file_type().is_dir() {
        return None;
    }
    let comps = relative_components(entry, root)?;
    let (host, namespace, model, tag) = parse_components(comps, include_hidden)?;

    let manifest_path = entry.path();
    let parsed = read_manifest(manifest_path)?;
    let (layers, config) = manifest_to_layers(&parsed);

    Some(build_listed_model(
        host,
        namespace,
        model,
        tag,
        manifest_path,
        layers,
        config,
        verbose,
        blobs_root,
        want_blob_paths,
    ))
}

/// Scan manifests and construct `ListedModel` entries.
pub fn scan_manifests(
    ScanArgs {
        root,
        blobs_root,
        include_hidden,
        verbose,
        blob_paths,
    }: ScanArgs,
) -> Result<Vec<ListedModel>> {
    let mut models = Vec::new();

    for entry_res in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry_res {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Skipping entry error: {e}");
                continue;
            }
        };
        if let Some(model) = process_entry(
            &entry,
            root,
            blobs_root,
            include_hidden,
            verbose,
            blob_paths,
        ) {
            models.push(model);
        }
    }
    Ok(models)
}

/// Build blob path info list and decide primary digest.
pub fn build_blob_infos(
    layers: &[LayerInfo],
    config: Option<&LayerInfo>,
    blobs_root: &Path,
) -> (Option<String>, Vec<BlobPathInfo>) {
    // Heuristic: primary = largest non-config layer with a size.
    let mut primary_digest_idx: Option<usize> = None;
    let mut max_size: u64 = 0;

    for (i, l) in layers.iter().enumerate() {
        if let Some(sz) = l.size
            && sz > max_size
        {
            max_size = sz;
            primary_digest_idx = Some(i);
        }
    }

    let primary_digest = primary_digest_idx
        .and_then(|i| layers.get(i).map(|l| l.digest.clone()))
        .or_else(|| config.map(|c| c.digest.clone()));

    let mut out = Vec::new();

    for l in layers {
        out.push(build_blob_path_info(l, blobs_root, false));
    }
    if let Some(cfg) = config {
        out.push(build_blob_path_info(cfg, blobs_root, false));
    }

    (primary_digest, out)
}

pub fn build_blob_path_info(l: &LayerInfo, blobs_root: &Path, primary: bool) -> BlobPathInfo {
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

pub fn digest_to_blob_path(blobs_root: &Path, digest: &str) -> PathBuf {
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
pub fn normalize_name(
    host: Option<&str>,
    namespace: Option<&str>,
    model: &str,
    tag: &str,
) -> String {
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
    pub fn test_normalize() {
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
    pub fn test_digest_to_blob_path() {
        let root = PathBuf::from("/tmp/blobs");
        let p = digest_to_blob_path(&root, "sha256:1234abcd");
        assert_eq!(p, PathBuf::from("/tmp/blobs/sha256-1234abcd"));
    }
}
