mod common;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use common::{AnvilRun, default_agent_command, run_gh, run_prompt};

#[derive(Parser, Debug)]
#[command(about = "Review a GitHub pull request with Anvil over ACP")]
struct Args {
    /// GitHub repository in owner/name form.
    #[arg(long)]
    repo: String,

    /// Pull request number to review.
    pr: u64,

    /// Workspace path Anvil should inspect.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,

    /// ACP agent command. Defaults to ANVIL_AGENT, target/debug/anvil, or cargo run.
    #[arg(long)]
    agent: Option<String>,

    /// Post Anvil's review back to the PR as a comment.
    #[arg(long)]
    post_comment: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let pr_number = args.pr.to_string();
    let pr_info = run_gh(&[
        "pr",
        "view",
        &pr_number,
        "--repo",
        &args.repo,
        "--json",
        "title,body,author,baseRefName,headRefName,url",
    ])?;
    let diff = run_gh(&["pr", "diff", &pr_number, "--repo", &args.repo])?;

    let prompt = format!(
        "You are a careful code review bot for `{repo}`.\n\
         Review the PR metadata and diff below. You may inspect the local checkout for surrounding context.\n\
         Do not edit files, create commits, or run shell commands.\n\
         Prioritize correctness, security, regressions, and missing tests. Ignore style nits.\n\
         If there are no concrete findings, say so clearly.\n\
         Return Markdown with Findings first, then Test Gaps, then a short Summary.\n\n\
         PR metadata JSON:\n{pr_info}\n\nDiff:\n{diff}",
        repo = args.repo,
    );

    let config = AnvilRun::read_only(
        args.agent.unwrap_or_else(default_agent_command),
        args.cwd.canonicalize()?,
    );
    let response = run_prompt(config, prompt).await?;

    if args.post_comment {
        run_gh(&[
            "pr", "comment", &pr_number, "--repo", &args.repo, "--body", &response,
        ])?;
        eprintln!("posted comment to PR #{pr_number}");
    }

    Ok(())
}
