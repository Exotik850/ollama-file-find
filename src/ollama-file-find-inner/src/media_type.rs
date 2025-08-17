//! Representation and parsing of Ollama layer media types.
//!
//! References (Ollama repository excerpts):
//! - Manifest docs: server/internal/manifest/manifest.go
//! - Runtime usage: server/images.go, server/model.go, server/create.go
//!
//! Examples of media types (with parameters):
//!   application/vnd.ollama.image.config; type=safetensors
//!   application/vnd.ollama.image.template
//!   application/vnd.ollama.image.template; name=chatml
//!   application/vnd.ollama.image.tensor; name=input; dtype=F32; shape=1,2,3
//!   application/vnd.ollama.image.tokenizer
//!   application/vnd.ollama.image.tokenizer.config
//!   application/vnd.ollama.image.license
//!
//! Legacy / additional observed:
//!   application/vnd.ollama.image.model              (deprecated umbrella)
//!   application/vnd.ollama.image.adapter
//!   application/vnd.ollama.image.projector
//!   application/vnd.ollama.image.system
//!   application/vnd.ollama.image.params
//!   application/vnd.ollama.image.messages
//!   application/vnd.ollama.image.prompt             (template-like)
//!   application/vnd.ollama.image.embed              (deprecated)
//!
//! Anything unrecognized is captured as `Unknown`.

use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Structured representation of a tensor layer's numeric shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape(pub Vec<u64>);

impl Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{d}")?;
        }
        Ok(())
    }
}

/// Errors that can arise while parsing a media type.
#[derive(Debug, thiserror::Error)]
pub enum MediaTypeParseError {
    #[error("empty media type string")]
    Empty,
    #[error("invalid base type (expected application/vnd.ollama.image.*): {0}")]
    InvalidBase(String),
    #[error("missing required parameter: {0}")]
    MissingParam(&'static str),
    #[error("invalid shape component (not an integer): {0}")]
    InvalidShapeComponent(String),
}

/// Enum representing known Ollama layer media types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OllamaMediaType {
    // Structured (new spec)
    Config {
        /// Underlying configuration format, e.g. "safetensors", "gguf"
        config_type: String,
    },
    Template {
        /// Optional template name (parameter 'name')
        name: Option<String>,
    },
    Tensor {
        name: String,
        dtype: String,
        /// Original shape string (preserved)
        shape_raw: String,
        /// Parsed shape (if all dimensions parse cleanly)
        shape: Option<TensorShape>,
    },
    Tokenizer,
    TokenizerConfig,
    License,

    // Additional / runtime / legacy media types
    #[default]
    Model,
    Adapter,
    Projector,
    System,
    Params,
    Messages,
    Prompt,
    EmbedLegacy,

    /// Any future or unknown media type; preserves original input.
    Unknown(String),
}

impl OllamaMediaType {
    /// True if this variant is considered deprecated.
    #[must_use]
    pub fn is_deprecated(&self) -> bool {
        matches!(self, Self::Model | Self::EmbedLegacy)
    }

    /// Return the canonical "base" (without parameters) for known types.
    #[must_use]
    pub fn base(&self) -> &str {
        match self {
            Self::Config { .. } => "application/vnd.ollama.image.config",
            Self::Template { .. } => "application/vnd.ollama.image.template",
            Self::Tensor { .. } => "application/vnd.ollama.image.tensor",
            Self::Tokenizer => "application/vnd.ollama.image.tokenizer",
            Self::TokenizerConfig => "application/vnd.ollama.image.tokenizer.config",
            Self::License => "application/vnd.ollama.image.license",
            Self::Model => "application/vnd.ollama.image.model",
            Self::Adapter => "application/vnd.ollama.image.adapter",
            Self::Projector => "application/vnd.ollama.image.projector",
            Self::System => "application/vnd.ollama.image.system",
            Self::Params => "application/vnd.ollama.image.params",
            Self::Messages => "application/vnd.ollama.image.messages",
            Self::Prompt => "application/vnd.ollama.image.prompt",
            Self::EmbedLegacy => "application/vnd.ollama.image.embed",
            Self::Unknown(s) => s,
        }
    }
}

impl Display for OllamaMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OllamaMediaType::Config { config_type } => {
                write!(f, "{}; type={}", self.base(), config_type)
            }
            OllamaMediaType::Template { name: Some(n) } => {
                write!(f, "{}; name={}", self.base(), n)
            }
            OllamaMediaType::Tensor {
                name,
                dtype,
                shape_raw,
                ..
            } => write!(
                f,
                "{}; name={}; dtype={}; shape={}",
                self.base(),
                name,
                dtype,
                shape_raw
            ),
            // Unparameterized forms:
            OllamaMediaType::Template { name: None }
            | OllamaMediaType::Tokenizer
            | OllamaMediaType::TokenizerConfig
            | OllamaMediaType::License
            | OllamaMediaType::Model
            | OllamaMediaType::Adapter
            | OllamaMediaType::Projector
            | OllamaMediaType::System
            | OllamaMediaType::Params
            | OllamaMediaType::Messages
            | OllamaMediaType::Prompt
            | OllamaMediaType::EmbedLegacy => f.write_str(self.base()),
            OllamaMediaType::Unknown(s) => f.write_str(s),
        }
    }
}

impl FromStr for OllamaMediaType {
    type Err = MediaTypeParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.trim();
        if input.is_empty() {
            return Err(MediaTypeParseError::Empty);
        }

        // Split into base and param chunks
        let mut parts = input.split(';').map(str::trim).collect::<Vec<_>>();
        let base = parts.remove(0);

        // Collect parameters into (k,v) map (simple key=value; no quoting support)
        let mut params = Vec::<(String, String)>::new();
        for p in parts {
            if p.is_empty() {
                continue;
            }
            let mut kv = p.splitn(2, '=');
            let k = kv.next().unwrap().trim();
            let v = kv.next().unwrap_or("").trim();
            if !k.is_empty() {
                params.push((k.to_string(), v.to_string()));
            }
        }
        let get_param = |key: &str| -> Option<String> {
            params
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| v.clone())
        };

        // Quick closure to validate base prefix but allow Unknown fallback
        let ensure_prefix = |b: &str| {
            if b.starts_with("application/vnd.ollama.image.") {
                Ok(())
            } else {
                Err(MediaTypeParseError::InvalidBase(b.to_string()))
            }
        };

        match base {
            // Structured new spec
            "application/vnd.ollama.image.config" => {
                ensure_prefix(base)?;
                let config_type =
                    get_param("type").ok_or(MediaTypeParseError::MissingParam("type"))?;
                Ok(Self::Config { config_type })
            }
            "application/vnd.ollama.image.template" => {
                ensure_prefix(base)?;
                let name = get_param("name");
                Ok(Self::Template { name })
            }
            "application/vnd.ollama.image.tensor" => {
                ensure_prefix(base)?;
                let name = get_param("name").ok_or(MediaTypeParseError::MissingParam("name"))?;
                let dtype = get_param("dtype").ok_or(MediaTypeParseError::MissingParam("dtype"))?;
                let shape_raw =
                    get_param("shape").ok_or(MediaTypeParseError::MissingParam("shape"))?;
                let shape = parse_shape(&shape_raw)?;
                Ok(Self::Tensor {
                    name,
                    dtype,
                    shape_raw,
                    shape,
                })
            }
            "application/vnd.ollama.image.tokenizer" => Ok(Self::Tokenizer),
            "application/vnd.ollama.image.tokenizer.config" => Ok(Self::TokenizerConfig),
            "application/vnd.ollama.image.license" => Ok(Self::License),

            // Additional / runtime / legacy:
            "application/vnd.ollama.image.model" => Ok(Self::Model),
            "application/vnd.ollama.image.adapter" => Ok(Self::Adapter),
            "application/vnd.ollama.image.projector" => Ok(Self::Projector),
            "application/vnd.ollama.image.system" => Ok(Self::System),
            "application/vnd.ollama.image.params" => Ok(Self::Params),
            "application/vnd.ollama.image.messages" => Ok(Self::Messages),
            "application/vnd.ollama.image.prompt" => Ok(Self::Prompt),
            "application/vnd.ollama.image.embed" => Ok(Self::EmbedLegacy),

            // Unknown but still maybe in reserved namespace:
            other if other.starts_with("application/vnd.ollama.image.") => {
                // Accept but mark unknown; keep parameters by reconstituting original string
                Ok(Self::Unknown(input.to_string()))
            }

            // Anything else: invalid base (decide whether to treat as Unknown or error)
            other => {
                // If you prefer strict rejection, return Err here instead.
                Err(MediaTypeParseError::InvalidBase(other.to_string()))
            }
        }
    }
}

fn parse_shape(raw: &str) -> Result<Option<TensorShape>, MediaTypeParseError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let mut dims = Vec::new();
    for part in trimmed.split(',') {
        let s = part.trim();
        if s.is_empty() {
            continue;
        }
        let v: u64 = s
            .parse()
            .map_err(|_| MediaTypeParseError::InvalidShapeComponent(s.to_string()))?;
        dims.push(v);
    }
    Ok(Some(TensorShape(dims)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config() {
        let mt: OllamaMediaType = "application/vnd.ollama.image.config; type=gguf"
            .parse()
            .unwrap();
        match mt {
            OllamaMediaType::Config { config_type } => assert_eq!(config_type, "gguf"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_template_with_name() {
        let mt: OllamaMediaType = "application/vnd.ollama.image.template; name=chatml"
            .parse()
            .unwrap();
        match mt {
            OllamaMediaType::Template { name } => assert_eq!(name.as_deref(), Some("chatml")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_tensor() {
        let mt: OllamaMediaType =
            "application/vnd.ollama.image.tensor; name=input; dtype=F32; shape=1, 2,3"
                .parse()
                .unwrap();
        match mt {
            OllamaMediaType::Tensor {
                name,
                dtype,
                shape_raw,
                shape: Some(ref s),
            } => {
                assert_eq!(name, "input");
                assert_eq!(dtype, "F32");
                assert_eq!(shape_raw, "1, 2,3");
                assert_eq!(s.0, vec![1, 2, 3]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_tensor() {
        let original = "application/vnd.ollama.image.tensor; name=output; dtype=I32; shape=4,5,6";
        let parsed: OllamaMediaType = original.parse().unwrap();
        assert_eq!(parsed.to_string(), original);
    }

    #[test]
    fn parse_legacy_model() {
        let mt: OllamaMediaType = "application/vnd.ollama.image.model".parse().unwrap();
        assert!(matches!(mt, OllamaMediaType::Model));
        assert!(mt.is_deprecated());
    }

    #[test]
    fn parse_unknown_reserved() {
        let original = "application/vnd.ollama.image.future; foo=bar";
        let mt: OllamaMediaType = original.parse().unwrap();
        match mt {
            OllamaMediaType::Unknown(s) => assert_eq!(s, original),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn reject_non_reserved() {
        let err = "text/plain".parse::<OllamaMediaType>().unwrap_err();
        assert!(matches!(err, MediaTypeParseError::InvalidBase(_)));
    }
}
