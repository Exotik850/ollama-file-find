use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct ManifestData {
    #[serde(default)]
    pub layers: Vec<LayerInfo>,
    #[serde(default)]
    pub config: Option<LayerInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerInfo {
    pub digest: String,
    #[serde(rename = "mediaType")]
    #[serde(default)]
    pub media_type: String,
    pub size: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ListedModel {
    /// Normalized display name (matches `ollama list` style)
    pub name: String,
    #[serde(flatten)]
    pub model_id: ModelId,
    /// Filesystem path to manifest
    pub manifest_path: PathBuf,
    /// Layers (if verbose)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<LayerInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<LayerInfo>,
    /// Total summed size (if verbose)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_size: Option<u64>,
    /// Manifest mtime (if verbose)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtime: Option<u64>,
    /// Primary model blob path (if `blob_paths`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_blob_path: Option<PathBuf>,
    /// All blob paths (if `blob_paths`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_paths: Option<Vec<BlobPathInfo>>,
}

impl ListedModel {
    /// Construct a non-verbose (base) ListedModel. Only identity & manifest path are populated.
    pub fn new(model_id: ModelId, manifest_path: impl Into<PathBuf>) -> ListedModel {
        ListedModel {
            name: model_id.normalize(),
            model_id,
            manifest_path: manifest_path.into(),
            layers: None,
            config: None,
            total_size: None,
            mtime: None,
            primary_blob_path: None,
            blob_paths: None,
        }
    }

    pub fn into_verbose(self, manifest: ManifestData, blobs_root: impl AsRef<Path>) -> Self {
        let blobs_root = blobs_root.as_ref();
        let total_size = crate::compute_total_size(&manifest.layers, manifest.config.as_ref());
        let mtime = crate::compute_mtime(&self.manifest_path);
        let (primary_digest, mut infos) =
            crate::build_blob_infos(&manifest.layers, manifest.config.as_ref(), blobs_root);
        let primary_blob_path = primary_digest
            .as_ref()
            .map(|d| crate::digest_to_blob_path(blobs_root, d));
        if let Some(pd) = primary_digest {
            for bi in &mut infos {
                if bi.digest == pd {
                    bi.primary = true;
                }
            }
        }
        ListedModel {
            layers: Some(manifest.layers),
            config: manifest.config,
            total_size,
            mtime,
            primary_blob_path,
            blob_paths: Some(infos),
            ..self
        }
    }
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct BlobPathInfo {
    pub digest: String,
    pub media_type: String,
    pub declared_size: Option<u64>,
    pub path: PathBuf,
    pub exists: bool,
    pub size_ok: Option<bool>, // Only Some if both declared & actual size available
    pub actual_size: Option<u64>,
    pub primary: bool,
}

/// Internal helper grouping the model identity parts.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelId {
    pub host: Option<String>,
    pub namespace: Option<String>,
    pub model: String,
    pub tag: String,
}

impl ModelId {
    /// Attempt to mirror Ollama list naming rules
    pub fn normalize(&self) -> String {
        let Self {
            host,
            namespace,
            model,
            tag,
        } = self;
        let default_host = "registry.ollama.ai";
        let library_ns = "library";
        match (host, namespace) {
            (Some(h), Some(ns)) if h == default_host && ns == library_ns => {
                format!("{model}:{tag}")
            }
            (Some(h), Some(ns)) if h == default_host => format!("{ns}/{model}:{tag}"),
            (None, Some(ns)) if ns == library_ns => format!("{model}:{tag}"),
            (None, Some(ns)) => format!("{ns}/{model}:{tag}"),
            (Some(h), Some(ns)) => format!("{h}/{ns}/{model}:{tag}"),
            _ => format!("{model}:{tag}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_normalize() {
        assert_eq!(
            ModelId {
                host: None,
                namespace: None,
                model: "mistral".to_string(),
                tag: "7b".to_string(),
            }
            .normalize(),
            "mistral:7b"
        );
        assert_eq!(
            ModelId {
                host: Some("registry.ollama.ai".to_string()),
                namespace: Some("apple".to_string()),
                model: "OpenELM".to_string(),
                tag: "latest".to_string(),
            }
            .normalize(),
            "apple/OpenELM:latest"
        );
        assert_eq!(
            ModelId {
                host: None,
                namespace: Some("apple".to_string()),
                model: "OpenELM".to_string(),
                tag: "latest".to_string(),
            }
            .normalize(),
            "apple/OpenELM:latest"
        );
        assert_eq!(
            ModelId {
                host: Some("myhost".to_string()),
                namespace: Some("myns".to_string()),
                model: "lips".to_string(),
                tag: "code".to_string(),
            }
            .normalize(),
            "myhost/myns/lips:code"
        );
        assert_eq!(
            ModelId {
                host: None,
                namespace: Some("library".to_string()),
                model: "phi4".to_string(),
                tag: "latest".to_string(),
            }
            .normalize(),
            "phi4:latest"
        );
    }
}
