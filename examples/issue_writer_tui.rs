mod common;

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use common::{AnvilRun, default_agent_command, run_gh, run_prompt};

#[derive(Parser, Debug)]
#[command(about = "Interactive issue writer that drafts and creates GitHub issues via Anvil")]
struct Args {
    /// GitHub repository in owner/name form.
    #[arg(long)]
    repo: String,

    /// Workspace path Anvil should inspect.
    #[arg(long, default_value = ".")]
    cwd: PathBuf,

    /// ACP agent command. Defaults to ANVIL_AGENT, target/debug/anvil, or cargo run.
    #[arg(long)]
    agent: Option<String>,

    /// Print the draft without creating the GitHub issue.
    #[arg(long)]
    dry_run: bool,

    /// Non-interactive issue description.
    #[arg(long)]
    prompt: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let agent = args.agent.unwrap_or_else(default_agent_command);
    let cwd = args.cwd.canonicalize()?;

    println!("Anvil issue writer for {}", args.repo);
    let mut initial_description = args.prompt.clone();
    if initial_description.is_none() {
        println!("Describe the issue. Finish with an empty line.\n");
    }

    loop {
        let description = match initial_description.take() {
            Some(text) => text,
            None => read_multiline("> ")?,
        };
        if description.trim().is_empty() {
            println!("No description entered.");
            return Ok(());
        }

        let prompt = format!(
            "You are drafting a new GitHub issue for `{repo}`.\n\
             Use the user's description below. Inspect the local checkout when useful, but do not edit files or run shell commands.\n\
             Return exactly this format:\n\
             TITLE: <concise issue title>\n\
             BODY:\n\
             <markdown issue body with problem, expected behavior, affected code if known, and suggested starting points>\n\n\
             User description:\n{description}",
            repo = args.repo,
        );

        let mut config = AnvilRun::read_only(agent.clone(), cwd.clone());
        config.echo = false;
        let draft_text = run_prompt(config, prompt).await?;
        let draft = IssueDraft::parse(&draft_text);

        println!("\n--- Draft title ---\n{}\n", draft.title);
        println!("--- Draft body ---\n{}\n", draft.body);

        if args.dry_run {
            return Ok(());
        }

        match prompt_one("Create this issue? [y]es, [r]egenerate, [q]uit: ")? {
            'y' | 'Y' => {
                let url = create_issue(&args.repo, &draft)?;
                println!("Created {url}");
                return Ok(());
            }
            'r' | 'R' => {
                println!("\nRegenerating from a revised description. Finish with an empty line.\n");
            }
            'q' | 'Q' => return Ok(()),
            _ => println!("Please choose y, r, or q."),
        }
    }
}

#[derive(Debug)]
struct IssueDraft {
    title: String,
    body: String,
}

impl IssueDraft {
    fn parse(text: &str) -> Self {
        let mut title = String::new();
        let mut body = String::new();
        let mut in_body = false;

        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("TITLE:") {
                title = rest.trim().to_string();
                continue;
            }
            if line.trim() == "BODY:" {
                in_body = true;
                continue;
            }
            if in_body {
                body.push_str(line);
                body.push('\n');
            }
        }

        if title.is_empty() {
            title = text
                .lines()
                .next()
                .unwrap_or("New issue")
                .trim()
                .to_string();
        }
        if body.trim().is_empty() {
            body = text.to_string();
        }

        Self {
            title,
            body: body.trim().to_string(),
        }
    }
}

fn read_multiline(prompt: &str) -> Result<String> {
    let mut buf = String::new();
    loop {
        print!("{prompt}");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 || line.trim().is_empty() {
            break;
        }
        buf.push_str(&line);
    }
    Ok(buf)
}

fn prompt_one(prompt: &str) -> Result<char> {
    print!("{prompt}");
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.chars().next().unwrap_or('\n'))
}

fn create_issue(repo: &str, draft: &IssueDraft) -> Result<String> {
    run_gh(&[
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        &draft.title,
        "--body",
        &draft.body,
    ])
    .context("create GitHub issue")
}
