use std::{
    env, fs,
    io,
    path::{Path, PathBuf},
    time::SystemTime,
};

mod models;
use models::{BlobPathInfo, LayerInfo, ListedModel};

use crate::models::{ManifestData, ModelId};

/// Library wide result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Error enum describing all failure modes the library can encounter.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Environment variable error: {0}")] 
    EnvVar(#[from] env::VarError),
    #[error("Home directory not found")] 
    HomeDirNotFound,
    #[error("IO error at {path}: {source}")] 
    Io { path: PathBuf, source: io::Error },
    #[error("Walkdir error: {0}")] 
    WalkDir(#[from] walkdir::Error),
    #[error("JSON parse error at {path}: {source}")] 
    Json { path: PathBuf, source: serde_json::Error },
    #[error("Invalid path components for manifest under {0}")] 
    InvalidComponents(PathBuf),
}

/// Outcome of a scan: the successfully parsed models plus any errors that occurred.
#[derive(Debug)]
pub struct ScanOutcome {
    pub models: Vec<ListedModel>,
    pub errors: Vec<Error>,
}

/// Locate the models directory (`OLLAMA_MODELS` or fallback to $HOME/.ollama/models)
pub fn ollama_models_dir() -> PathBuf {
    if let Ok(p) = env::var("OLLAMA_MODELS") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    // Fallback to home, but if not found just current directory relative path
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
    if !include_hidden && tag.starts_with('.')
        || comps[..comps.len() - 1].iter().any(|c| c.starts_with('.'))
    {
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
fn load_manifest(path: &Path) -> Result<ManifestData> {
    let data = fs::read(path).map_err(|e| Error::Io { path: path.to_path_buf(), source: e })?;
    let parsed = serde_json::from_slice(&data).map_err(|e| Error::Json { path: path.to_path_buf(), source: e })?;
    Ok(parsed)
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

/// Attempt to turn a filesystem entry into a `ListedModel` (only if it's a manifest file
/// with valid components). Returns `None` for directories, hidden-excluded entries, or
/// any IO / parse failures.
fn process_entry(entry: &walkdir::DirEntry, args: &ScanArgs) -> Result<Option<ListedModel>> {
    if entry.file_type().is_dir() {
        return Ok(None);
    }
    let comps = match relative_components(entry, args.root) {
        Some(c) => c,
        None => return Ok(None),
    };
    let id = parse_components(comps, args.include_hidden)
        .ok_or_else(|| Error::InvalidComponents(entry.path().to_path_buf()))?;
    let manifest_path = entry.path();
    let manifest = load_manifest(manifest_path)?;
    let model = ListedModel::new(id, manifest_path);
    if args.verbose {
        Ok(Some(model.into_verbose(manifest, &args.blobs_root)))
    } else {
        Ok(Some(model))
    }
}

/// Scan manifests and construct `ListedModel` entries.
pub fn scan_manifests(args: ScanArgs) -> ScanOutcome {
    let mut models = Vec::new();
    let mut errors = Vec::new();
    for entry_res in walkdir::WalkDir::new(args.root).follow_links(false) {
        match entry_res {
            Ok(entry) => match process_entry(&entry, &args) {
                Ok(Some(model)) => models.push(model),
                Ok(None) => {}
                Err(e) => errors.push(e),
            },
            Err(e) => errors.push(Error::WalkDir(e)),
        }
    }
    models.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    ScanOutcome { models, errors }
}

/// Build blob path info list and decide primary digest.
/// Build blob info records for layers + optional config, returning the primary digest chosen.
/// Primary heuristic: largest (by declared size) layer; fall back to config if none.
pub fn build_blob_infos(
    layers: &[LayerInfo],
    config: Option<LayerInfo>,
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
    let mut out = Vec::with_capacity(layers.len() + config.is_some() as usize);
    let primary_digest = primary_digest_idx
        .and_then(|i| layers.get(i).map(|l| l.digest.clone()))
        .or_else(|| config.clone().map(|c| c.digest));
    for l in layers.iter().chain(config.iter()) {
        out.push(build_blob_path_info(l, blobs_root));
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
