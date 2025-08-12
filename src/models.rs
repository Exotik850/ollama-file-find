use std::path::PathBuf;

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct ManifestJson {
    #[serde(default)]
    pub layers: Vec<LayerJson>,
    #[serde(default)]
    pub config: Option<LayerJson>,
}

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct LayerJson {
    pub digest: String,
    #[serde(rename = "mediaType")]
    #[serde(default)]
    pub media_type: String,
    pub size: Option<u64>,
}

#[derive(Debug, serde::Serialize)]
pub struct ListedModel {
    /// Normalized display name (matches `ollama list` style)
    pub name: String,
    /// Raw components
    pub host: Option<String>,
    pub namespace: Option<String>,
    pub model: String,
    pub tag: String,
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
pub struct LayerInfo {
    pub digest: String,
    pub media_type: String,
    pub size: Option<u64>,
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
