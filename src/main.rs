type Result<T> = std::result::Result<T, anyhow::Error>;

mod args;
use args::Args;

use clap::Parser;
use ollama_file_find::{ScanArgs, ollama_models_dir, scan_manifests};

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

    let outcome = scan_manifests(
        &ScanArgs::new(manifests_root, blobs_root)
            .with_include_hidden(include_hidden)
            .with_verbose(verbose),
    );

    for e in &outcome.errors {
        eprintln!("Warning: {e}");
    }

    if plain && !verbose {
        for m in &outcome.models {
            println!("{}", m.name);
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&outcome.models)?);
    }

    Ok(())
}
