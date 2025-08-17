use std::{
    env,
    fs, io,
    mem::take,
    path::{Path, PathBuf},
    time::SystemTime,
};

mod models;
pub use models::{BlobPathInfo, LayerInfo, ListedModel};

mod scan_args;
pub use scan_args::ScanArgs;

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
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("Invalid path components for manifest under {0}")]
    InvalidComponentPath(PathBuf),
    #[error("Invalid components: {0:?}")]
    InvalidComponents(Vec<String>),
}

/// Outcome of a scan: the successfully parsed models plus any errors that occurred.
#[derive(Debug)]
pub struct ScanOutcome {
    pub models: Vec<ListedModel>,
    pub errors: Vec<Error>,
}

/// Locate the models directory (`OLLAMA_MODELS` or fallback to $HOME/.ollama/models)
#[must_use] pub fn ollama_models_dir() -> PathBuf {
    if let Ok(p) = env::var("OLLAMA_MODELS")
        && !p.is_empty() {
            return PathBuf::from(p);
        }
    // Fallback to home, but if not found just current directory relative path
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".ollama").join("models")
}

/// Get the relative path components for a directory entry.
fn relative_components(entry: &walkdir::DirEntry, root: &Path) -> Result<Vec<String>> {
    if !entry.path().starts_with(root) {
        return Err(Error::InvalidComponentPath(entry.path().to_path_buf()));
    }
    let rel = entry.path().strip_prefix(root).expect("Should be relative");
    let comps: Vec<String> = rel
        .iter()
        .map(|c| c.to_string_lossy().to_string())
        .collect();
    if comps.is_empty() {
        return Err(Error::InvalidComponentPath(entry.path().to_path_buf()));
    }
    Ok(comps)
}

/// Interpret path components as (host?, namespace, model, tag).
fn parse_components(mut comps: Vec<String>, include_hidden: bool) -> Result<Option<ModelId>> {
    // Accept either:
    //   4 components: host / namespace / model / tag
    //   3 components:          namespace / model / tag
    match comps.len() {
        3 | 4 => {}
        _ => return Err(Error::InvalidComponents(comps)),
    }

    // Exclude any component starting with '.' unless explicitly allowed.
    if !include_hidden && comps.iter().any(|c| c.starts_with('.')) {
        return Ok(None);
    }

    // Destructure and clone only what we need.
    let (host, namespace, model, tag) = match comps.as_mut_slice() {
        [host, namespace, model, tag] => (
            Some(take(host)),
            Some(take(namespace)),
            take(model),
            take(tag),
        ),
        [namespace, model, tag] => (None, Some(take(namespace)), take(model), take(tag)),
        _ => unreachable!("Lengths other than 3 or 4 already returned above"),
    };

    Ok(Some(ModelId {
        host,
        namespace,
        model,
        tag,
    }))
}

/// Read & parse a manifest JSON file into a strongly typed structure.
fn load_manifest(path: &Path) -> Result<ManifestData> {
    let data = fs::read(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let parsed = serde_json::from_slice(&data).map_err(|e| Error::Json {
        path: path.to_path_buf(),
        source: e,
    })?;
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
    if let Some(cfg) = config
        && let Some(sz) = cfg.size {
            sum += sz;
            any = true;
        }
    if any { Some(sum) } else { None }
}

// Number of seconds since the file was last modified, if applicable
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
    let comps = relative_components(entry, &args.root)?;
    let Some(id) = parse_components(comps, args.include_hidden)? else {
        return Ok(None);
    };
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
#[must_use] pub fn scan_manifests(args: &ScanArgs) -> ScanOutcome {
    let mut models = Vec::new();
    let mut errors = Vec::new();
    for entry_res in walkdir::WalkDir::new(&args.root).follow_links(false) {
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
#[must_use] pub fn build_blob_infos<'a>(
    layers: &'a [LayerInfo],
    config: Option<&'a LayerInfo>,
    blobs_root: &Path,
) -> (Option<&'a str>, Vec<BlobPathInfo>) {
    let mut primary_digest_idx: Option<usize> = None;
    let mut max_size: u64 = 0;
    for (i, l) in layers.iter().enumerate() {
        if let Some(sz) = l.size
            && sz > max_size {
                max_size = sz;
                primary_digest_idx = Some(i);
            }
    }
    let mut out = Vec::with_capacity(layers.len() + usize::from(config.is_some()));
    let primary_digest = primary_digest_idx
        .and_then(|i| layers.get(i).map(|l| l.digest.as_ref()))
        .or_else(|| config.map(|c| c.digest.as_ref()));
    for l in layers.iter().chain(config.iter().copied()) {
        out.push(build_blob_path_info(l, blobs_root));
    }
    (primary_digest, out)
}

/// Produce a `BlobPathInfo` for the provided layer/config entry.
#[must_use] pub fn build_blob_path_info(l: &LayerInfo, blobs_root: &Path) -> BlobPathInfo {
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
        path,
        exists,
        size_ok,
        actual_size,
        primary: false,
    }
}

/// Translate a content digest (e.g. `sha256:abcd...`) to Ollama's on-disk blob path.
#[must_use] pub fn digest_to_blob_path(blobs_root: &Path, digest: &str) -> PathBuf {
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
