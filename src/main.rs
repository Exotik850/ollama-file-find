type Result<T> = std::result::Result<T, anyhow::Error>;

mod args;
use args::Args;

use clap::Parser;
use ollama_file_find::*;

fn main() -> Result<()> {
    let args = Args::parse();
    let models_dir = resolve_models_dir(args.models_dir.as_deref());
    let manifests_root = models_dir.join("manifests");
    let blobs_root = models_dir.join("blobs");

    if !manifests_root.is_dir() {
        anyhow::bail!(
            "Manifests directory not found: {}",
            manifests_root.display()
        );
    }

    let mut models = scan_manifests(ScanArgs {
        root: &manifests_root,
        blobs_root: &blobs_root,
        models_dir: &models_dir,
        include_hidden: args.include_hidden,
        verbose: args.verbose,
        blob_paths: args.blob_paths,
    })?;

    // Sort for deterministic output (by display name)
    models.sort_unstable_by(|a, b| a.name.cmp(&b.name));

    if args.plain && !args.blob_paths && !args.verbose {
        for m in &models {
            println!("{}", m.name);
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&models)?);
    }

    Ok(())
}
