type Result<T> = std::result::Result<T, anyhow::Error>;

mod args;
use args::Args;

use clap::Parser;
use ollama_file_find_inner::*;

fn main() -> Result<()> {
    let Args {
        plain,
        include_hidden,
        verbose,
        models_dir,
    } = Args::parse();

    let models_dir = models_dir.unwrap_or_else(ollama_models_dir);
    let manifests_root = models_dir.join("manifests");
    let blobs_root = models_dir.join("blobs");

    if !manifests_root.is_dir() {
        anyhow::bail!(
            "Manifests directory not found: {}",
            manifests_root.display()
        );
    }

    let models = scan_manifests(ScanArgs {
        root: &manifests_root,
        blobs_root: &blobs_root,
        include_hidden,
        verbose,
    });

    if plain && !verbose {
        for m in &models {
            println!("{}", m.name);
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&models)?);
    }

    Ok(())
}
