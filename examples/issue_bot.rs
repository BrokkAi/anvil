mod common;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use common::{AnvilRun, default_agent_command, run_gh, run_prompt};

#[derive(Parser, Debug)]
#[command(about = "Analyze a GitHub issue with Anvil over ACP")]
struct Args {
    /// GitHub repository in owner/name form.
    #[arg(long)]
    repo: String,

    /// Issue number to fetch with gh. Optional when --issue-file or --issue-text is supplied.
    issue: Option<u64>,

    /// Read issue text from a local file instead of GitHub.
    #[arg(long, conflicts_with = "issue_text")]
    issue_file: Option<PathBuf>,

    /// Use this literal issue text instead of GitHub.
    #[arg(long, conflicts_with = "issue_file")]
    issue_text: Option<String>,

    /// Workspace path Anvil should inspect.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,

    /// ACP agent command. Defaults to ANVIL_AGENT, target/debug/anvil, or cargo run.
    #[arg(long)]
    agent: Option<String>,

    /// Post Anvil's answer back to the GitHub issue. Requires an issue number.
    #[arg(long)]
    post_comment: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let issue = load_issue(&args)?;

    let prompt = format!(
        "You are an issue triage bot for `{repo}`.\n\
         Use the GitHub issue context below and inspect the local checkout when useful.\n\
         Do not edit files, create commits, or run shell commands.\n\
         Return Markdown with these sections: Summary, Likely Cause, Relevant Code, Suggested Fix, Risk.\n\n\
         GitHub issue:\n\n{issue}",
        repo = args.repo,
    );

    let config = AnvilRun::read_only(
        args.agent.unwrap_or_else(default_agent_command),
        args.cwd.canonicalize()?,
    );
    let response = run_prompt(config, prompt).await?;

    if args.post_comment {
        let issue_number = args
            .issue
            .context("--post-comment requires an issue number")?
            .to_string();
        run_gh(&[
            "issue",
            "comment",
            &issue_number,
            "--repo",
            &args.repo,
            "--body",
            &response,
        ])?;
        eprintln!("posted comment to issue #{issue_number}");
    }

    Ok(())
}

fn load_issue(args: &Args) -> Result<String> {
    if let Some(text) = &args.issue_text {
        return Ok(text.clone());
    }

    if let Some(path) = &args.issue_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("read issue file {}", path.display()));
    }

    let issue_number = args
        .issue
        .context("provide an issue number, --issue-file, or --issue-text")?
        .to_string();
    run_gh(&[
        "issue",
        "view",
        &issue_number,
        "--repo",
        &args.repo,
        "--comments",
    ])
}
