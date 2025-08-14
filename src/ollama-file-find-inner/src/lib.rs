use std::{
    env, fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

mod models;
use models::{BlobPathInfo, LayerInfo, ListedModel};

use crate::models::{ManifestData, ModelId};

/// Locate the models directory (`OLLAMA_MODELS` or fallback to $HOME/.ollama/models)
pub fn ollama_models_dir() -> PathBuf {
    if let Ok(p) = env::var("OLLAMA_MODELS") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".ollama").join("models")
}

/// Arguments controlling a scan of the manifests directory.
pub struct ScanArgs<'a> {
    /// Root of the manifests tree (models/manifests)
    pub root: &'a Path,
    /// Root of the blobs directory (models/blobs)
    pub blobs_root: &'a Path,
    /// Include entries whose components (namespace, tag, etc.) start with '.'
    pub include_hidden: bool,
    /// Include extra detail (layer list, total size, mtime, blob info)
    pub verbose: bool,
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

// TODO: Clean this up somehow
/// Interpret path components as (host?, namespace, model, tag).
fn parse_components(comps: Vec<String>, include_hidden: bool) -> Option<ModelId> {
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

    let (host, namespace, model, tag) = match comps.len() {
        4 => {
            let mut it = comps.into_iter();
            let host = it.next().unwrap();
            let namespace = it.next().unwrap();
            let model = it.next().unwrap();
            let tag = it.next().unwrap();
            (Some(host), Some(namespace), model, tag)
        }
        3 => {
            let mut it = comps.into_iter();
            let namespace = it.next().unwrap();
            let model = it.next().unwrap();
            let tag = it.next().unwrap();
            (None, Some(namespace), model, tag)
        }
        _ => unreachable!(), // length already validated above
    };
    Some(ModelId {
        host,
        namespace,
        model,
        tag,
    })
}

/// Read & parse a manifest JSON file into a strongly typed structure.
fn load_manifest(path: &Path) -> Option<ManifestData> {
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

/// Sum layer + config sizes, returning None if no declared sizes exist.
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

/// Build the final `ListedModel` structure from its pieces.
fn build_listed_model(
    model_id: ModelId,
    manifest: ManifestData,
    manifest_path: &Path,
    blobs_root: &Path,
    verbose: bool,
) -> ListedModel {
    let name = model_id.normalize();

    // Optional details only computed if requested.
    let total_size = if verbose {
        compute_total_size(&manifest.layers, manifest.config.as_ref())
    } else {
        None
    };
    let mtime = if verbose {
        compute_mtime(manifest_path)
    } else {
        None
    };

    // Blob path info (primary + list) only if either verbose or blob_paths requested.
    let (primary_blob_path, blob_paths) = if verbose {
        let (primary_digest, mut infos) =
            build_blob_infos(&manifest.layers, manifest.config.as_ref(), blobs_root);
        let primary_path = primary_digest
            .as_ref()
            .map(|d| digest_to_blob_path(blobs_root, d));
        if let Some(pd) = primary_digest {
            for bi in &mut infos {
                if bi.digest == pd {
                    bi.primary = true;
                }
            }
        }
        (primary_path, infos)
    } else {
        (None, Vec::new())
    };

    ListedModel {
        name,
        model_id,
        manifest_path: manifest_path.display().to_string(),
        layers: if verbose {
            Some(manifest.layers.clone())
        } else {
            None
        },
        config: if verbose {
            manifest.config.clone()
        } else {
            None
        },
        total_size,
        mtime,
        primary_blob_path,
        blob_paths: (!blob_paths.is_empty()).then_some(blob_paths),
    }
}

/// Attempt to turn a filesystem entry into a `ListedModel` (only if it's a manifest file
/// with valid components). Returns `None` for directories, hidden-excluded entries, or
/// any IO / parse failures.
fn process_entry(entry: &walkdir::DirEntry, args: &ScanArgs) -> Option<ListedModel> {
    if entry.file_type().is_dir() {
        return None;
    }
    let comps = relative_components(entry, args.root)?;
    let id = parse_components(comps, args.include_hidden)?;
    let manifest_path = entry.path();
    let manifest = load_manifest(manifest_path)?;
    Some(build_listed_model(
        id,
        manifest,
        manifest_path,
        args.blobs_root,
        args.verbose,
    ))
}

/// Scan manifests and construct `ListedModel` entries.
pub fn scan_manifests(args: ScanArgs) -> Vec<ListedModel> {
    let mut models = Vec::new();
    for entry_res in walkdir::WalkDir::new(args.root).follow_links(false) {
        let entry = match entry_res {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Skipping entry error: {e}");
                continue;
            }
        };
        if let Some(model) = process_entry(&entry, &args) {
            models.push(model);
        }
    }
    models.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Build blob path info list and decide primary digest.
/// Build blob info records for layers + optional config, returning the primary digest chosen.
/// Primary heuristic: largest (by declared size) layer; fall back to config if none.
pub fn build_blob_infos(
    layers: &[LayerInfo],
    config: Option<&LayerInfo>,
    blobs_root: &Path,
) -> (Option<String>, Vec<BlobPathInfo>) {
    let mut primary_digest_idx: Option<usize> = None;
    let mut max_size: u64 = 0;
    for (i, l) in layers.iter().enumerate() {
        if let Some(sz) = l.size {
            if sz > max_size {
                max_size = sz;
                primary_digest_idx = Some(i);
            }
        }
    }
    let primary_digest = primary_digest_idx
        .and_then(|i| layers.get(i).map(|l| l.digest.clone()))
        .or_else(|| config.map(|c| c.digest.clone()));

    let mut out = Vec::with_capacity(layers.len() + config.is_some() as usize);
    for l in layers {
        out.push(build_blob_path_info(l, blobs_root));
    }
    if let Some(cfg) = config {
        out.push(build_blob_path_info(cfg, blobs_root));
    }
    (primary_digest, out)
}

/// Produce a `BlobPathInfo` for the provided layer/config entry.
pub fn build_blob_path_info(l: &LayerInfo, blobs_root: &Path) -> BlobPathInfo {
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
        primary: false,
    }
}

/// Translate a content digest (e.g. `sha256:abcd...`) to Ollama's on-disk blob path.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_digest_to_blob_path() {
        let root = PathBuf::from("/tmp/blobs");
        let p = digest_to_blob_path(&root, "sha256:1234abcd");
        assert_eq!(p, PathBuf::from("/tmp/blobs/sha256-1234abcd"));
    }
}
