use std::path::PathBuf;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    version,
    author="@Exotik850",
    about = "List locally installed Ollama models by reading the manifests directory"
)]
pub(crate) struct Args {
    /// Emit plain text (just model names) instead of JSON
    #[arg(long)]
    pub plain: bool,

    /// Include hidden tags (those beginning with '.')
    #[arg(long)]
    pub include_hidden: bool,

    /// Show layer digests, sizes, total size, timestamps,
    /// and blob paths
    #[arg(long)]
    pub verbose: bool,

    /// Root of models directory (overrides env + fallback)
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
}
