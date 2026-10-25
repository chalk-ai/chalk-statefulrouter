use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use chalk_statefulrouter::{config::Config, Runtime};

#[derive(Parser)]
#[command(about = "Session- and KV-cache-aware router for LLM inference backends")]
struct Args {
    /// Path to the YAML config.
    #[arg(long, short, env = "CHALK_STATEFULROUTER_CONFIG")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let (listen, admin_listen) = (config.listen, config.admin_listen);
    let runtime = Runtime::new(config).await?;

    let proxy = tokio::net::TcpListener::bind(listen).await?;
    let admin = tokio::net::TcpListener::bind(admin_listen).await?;
    tracing::info!(%listen, %admin_listen, "chalk-statefulrouter listening");

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
    };
    tokio::select! {
        r = axum::serve(proxy, runtime.proxy_app()) => r?,
        r = axum::serve(admin, runtime.admin_app()) => r?,
        _ = shutdown => {}
    }
    Ok(())
}
