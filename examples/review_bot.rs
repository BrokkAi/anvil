mod common;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use common::{AnvilRun, default_agent_command, run_gh, run_prompt};

#[derive(Parser, Debug)]
#[command(about = "Review a GitHub pull request with Anvil over ACP")]
struct Args {
    /// GitHub repository in owner/name form.
    #[arg(long)]
    repo: String,

    /// Pull request number to fetch with gh. Optional when reviewing a local diff.
    pr: Option<u64>,

    /// Read a diff from a local file instead of GitHub or git diff.
    #[arg(long, conflicts_with = "diff_text")]
    diff_file: Option<PathBuf>,

    /// Use this literal diff text instead of GitHub or git diff.
    #[arg(long, conflicts_with = "diff_file")]
    diff_text: Option<String>,

    /// Workspace path Anvil should inspect.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,

    /// ACP agent command. Defaults to ANVIL_AGENT, target/debug/anvil, or cargo run.
    #[arg(long)]
    agent: Option<String>,

    /// Post Anvil's review back to the GitHub PR. Requires a PR number.
    #[arg(long)]
    post_comment: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let (pr_info, diff) = load_review_input(&args)?;

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
        let pr_number = args
            .pr
            .context("--post-comment requires a PR number")?
            .to_string();
        run_gh(&[
            "pr", "comment", &pr_number, "--repo", &args.repo, "--body", &response,
        ])?;
        eprintln!("posted comment to PR #{pr_number}");
    }

    Ok(())
}

fn load_review_input(args: &Args) -> Result<(String, String)> {
    if let Some(path) = &args.diff_file {
        let diff = std::fs::read_to_string(path)
            .with_context(|| format!("read diff file {}", path.display()))?;
        return Ok(("local diff file".to_string(), diff));
    }

    if let Some(text) = &args.diff_text {
        return Ok(("local diff text".to_string(), text.clone()));
    }

    if let Some(pr) = args.pr {
        let pr_number = pr.to_string();
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
        return Ok((pr_info, diff));
    }

    let staged = run_git(&["diff", "--cached"]).unwrap_or_default();
    let unstaged = run_git(&["diff"]).unwrap_or_default();
    let diff = [staged.trim(), unstaged.trim()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    if diff.is_empty() {
        anyhow::bail!("no local git diff found; provide a PR number, --diff-file, or --diff-text");
    }

    Ok(("local git diff".to_string(), diff))
}

fn run_git(args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git {} failed: {}", args.join(" "), stderr.trim());
    }
}
