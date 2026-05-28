mod common;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use common::{AnvilRun, default_agent_command, run_gh, run_prompt};

#[derive(Parser, Debug)]
#[command(about = "Analyze a GitHub issue with Anvil over ACP")]
struct Args {
    /// GitHub repository in owner/name form.
    #[arg(long)]
    repo: String,

    /// Issue number to analyze.
    issue: u64,

    /// Workspace path Anvil should inspect.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,

    /// ACP agent command. Defaults to ANVIL_AGENT, target/debug/anvil, or cargo run.
    #[arg(long)]
    agent: Option<String>,

    /// Post Anvil's answer back to the issue as a comment.
    #[arg(long)]
    post_comment: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let issue_number = args.issue.to_string();
    let issue = run_gh(&[
        "issue",
        "view",
        &issue_number,
        "--repo",
        &args.repo,
        "--comments",
    ])?;

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
