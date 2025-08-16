use std::{borrow::Cow, path::Path};

/// Arguments controlling a scan of the manifests directory.
pub struct ScanArgs<'a> {
    /// Root of the manifests tree (models/manifests)
    pub root: Cow<'a, Path>,
    /// Root of the blobs directory (models/blobs)
    pub blobs_root: Cow<'a, Path>,
    /// Include entries whose components (namespace, tag, etc.) start with '.'
    pub include_hidden: bool,
    /// Include extra detail (layer list, total size, mtime, blob info)
    pub verbose: bool,
}

impl<'a> ScanArgs<'a> {
    /// Create ScanArgs borrowing the provided paths.
    pub fn new<P1: Into<Cow<'a, Path>>, P2: Into<Cow<'a, Path>>>(root: P1, blobs_root: P2) -> Self {
        ScanArgs {
            root: root.into(),
            blobs_root: blobs_root.into(),
            ..Default::default()
        }
    }

    pub fn with_include_hidden(self, include_hidden: bool) -> Self {
        ScanArgs {
            include_hidden,
            ..self
        }
    }

    pub fn with_verbose(self, verbose: bool) -> Self {
        ScanArgs { verbose, ..self }
    }
}

impl Default for ScanArgs<'static> {
    fn default() -> Self {
        let models_dir = crate::ollama_models_dir();
        let manifests_root = models_dir.join("manifests");
        let blobs_root = models_dir.join("blobs");
        ScanArgs {
            root: manifests_root.into(),
            blobs_root: blobs_root.into(),
            include_hidden: false,
            verbose: false,
        }
    }
}
