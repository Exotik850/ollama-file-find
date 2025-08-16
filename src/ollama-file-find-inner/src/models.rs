use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub(crate) struct ManifestData {
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
    pub manifest_path: String,
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

#[derive(Debug, serde::Serialize, Clone)]
pub struct BlobPathInfo {
    pub digest: String,
    pub media_type: String,
    pub declared_size: Option<u64>,
    pub path: String,
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
