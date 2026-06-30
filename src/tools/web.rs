//! `web_search` built-in tool.
//!
//! Zero-config web search backed by DuckDuckGo's keyless HTML endpoint
//! (`html.duckduckgo.com/html/`). This is deliberately a harness-side built-in
//! rather than a bundled MCP server: a search is a stateless HTTP request, so an
//! in-process backend reuses the `reqwest`/`http_retry` machinery already in the
//! binary and keeps anvil a single self-contained executable -- no `npx`/`uvx`
//! toolchain, no subprocess to spawn and supervise, no silent tool loss when
//! that toolchain is missing.
//!
//! The provider is abstracted behind [`SearchProvider`] so a keyed backend
//! (Brave, Tavily, Exa, ...) is a new enum variant plus a selector change and
//! nothing else: the tool schema, dispatch, permission classification, and
//! announce title all stay put. DuckDuckGo's keyless endpoint is the default
//! because it needs no API key; it is also rate-limited and occasionally returns
//! nothing, so callers treat an empty result set as a normal (non-error)
//! outcome.

use super::{ToolResult, ToolStatus};
use crate::http_retry::send_with_retries;
use regex::Regex;
use std::sync::LazyLock;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Default result count when the caller omits `count`.
const DEFAULT_RESULTS: usize = 10;
/// Hard ceiling on results returned to the model, regardless of `count`.
const MAX_RESULTS: usize = 20;
/// Per-snippet character cap so one verbose result can't dominate the output.
const SNIPPET_MAX_CHARS: usize = 500;
/// Per-title character cap (titles are short, but guard against pathological markup).
const TITLE_MAX_CHARS: usize = 300;
/// Final guard on the whole formatted blob handed back to the model.
const OUTPUT_MAX_BYTES: usize = 16_000;
/// DuckDuckGo's keyless HTML results endpoint.
const DUCKDUCKGO_HTML_ENDPOINT: &str = "https://html.duckduckgo.com/html/";
/// Network timeout for a single search request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
/// Desktop User-Agent. DuckDuckGo's HTML endpoint returns an empty/blocked page
/// to clients that look like bots, so we present a normal browser UA.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
     (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// A single parsed search hit.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// The active web-search backend. Today only DuckDuckGo (keyless) is wired up;
/// adding a keyed provider is a new variant plus an arm in [`Self::search`] and
/// a branch in [`Self::selected`].
enum SearchProvider {
    DuckDuckGo,
}

impl SearchProvider {
    /// Pick the backend for this session. A future keyed provider would be
    /// selected here from config (and the tool would advertise the chosen
    /// provider in its output header).
    fn selected() -> Self {
        SearchProvider::DuckDuckGo
    }

    /// Human-readable provider name, surfaced in the result header.
    fn label(&self) -> &'static str {
        match self {
            SearchProvider::DuckDuckGo => "DuckDuckGo",
        }
    }

    async fn search(
        &self,
        client: &reqwest::Client,
        query: &str,
        cancel: Option<&CancellationToken>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        match self {
            SearchProvider::DuckDuckGo => {
                duckduckgo_search(client, DUCKDUCKGO_HTML_ENDPOINT, query, cancel).await
            }
        }
    }
}

/// Entry point invoked by the tool dispatcher. Validates arguments, runs the
/// selected backend, applies the optional domain filter, and formats the
/// results. An empty result set is a successful (non-error) outcome.
pub(super) async fn run_web_search(
    query: &str,
    count: Option<usize>,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    cancel: Option<&CancellationToken>,
) -> ToolResult {
    let query = query.trim();
    if query.is_empty() {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: "Invalid arguments for `web_search`: `query` must not be empty.".to_string(),
        };
    }

    let allowed = non_empty_domains(allowed_domains);
    let blocked = non_empty_domains(blocked_domains);
    if allowed.is_some() && blocked.is_some() {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: "Invalid arguments for `web_search`: provide `allowed_domains` or \
                     `blocked_domains`, but not both."
                .to_string(),
        };
    }

    let count = count.unwrap_or(DEFAULT_RESULTS).clamp(1, MAX_RESULTS);

    let provider = SearchProvider::selected();
    let results = match provider.search(&SEARCH_CLIENT, query, cancel).await {
        Ok(results) => results,
        Err(err) => {
            return ToolResult {
                status: ToolStatus::InternalError,
                output: format!(
                    "web_search failed via {}: {err}. The keyless endpoint can rate-limit; \
                     retrying or rephrasing may help.",
                    provider.label()
                ),
            };
        }
    };

    let results = apply_domain_filter(results, allowed.as_deref(), blocked.as_deref());
    let results: Vec<SearchResult> = results.into_iter().take(count).collect();

    ToolResult {
        status: ToolStatus::Success,
        output: truncate_bytes(
            &format_results(query, provider.label(), &results),
            OUTPUT_MAX_BYTES,
        ),
    }
}

/// Drop empty/whitespace-only domain entries; return `None` when nothing remains
/// so an empty array behaves like an absent filter.
fn non_empty_domains(domains: Option<Vec<String>>) -> Option<Vec<String>> {
    let cleaned: Vec<String> = domains
        .unwrap_or_default()
        .into_iter()
        .filter_map(|d| {
            let d = d.trim().trim_start_matches("www.").to_ascii_lowercase();
            (!d.is_empty()).then_some(d)
        })
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Keep results matching `allowed` (whitelist) or drop results matching
/// `blocked` (blacklist). At most one is `Some` (enforced by the caller).
fn apply_domain_filter(
    results: Vec<SearchResult>,
    allowed: Option<&[String]>,
    blocked: Option<&[String]>,
) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter(|r| {
            let Some(host) = host_of(&r.url) else {
                // A result we can't attribute to a host is kept only when there
                // is no whitelist to satisfy.
                return allowed.is_none();
            };
            if let Some(allowed) = allowed {
                return allowed.iter().any(|d| domain_matches(&host, d));
            }
            if let Some(blocked) = blocked {
                return !blocked.iter().any(|d| domain_matches(&host, d));
            }
            true
        })
        .collect()
}

/// Whether `host` is `domain` or a subdomain of it. Both are lowercased and
/// `www.`-stripped before this point.
fn domain_matches(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Best-effort host extraction without pulling in a URL parser: strip the
/// scheme, any userinfo, the path, and the port.
fn host_of(url: &str) -> Option<String> {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    let authority = after_scheme.split(['/', '?', '#']).next()?;
    let host = authority.rsplit('@').next()?; // drop userinfo if present
    let host = host.split(':').next()?; // drop port
    let host = host.strip_prefix("www.").unwrap_or(host);
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

/// Render results as a compact numbered list. Zero results yields an
/// explanatory note rather than an empty string, so the model understands the
/// search ran but found nothing.
fn format_results(query: &str, provider: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!(
            "No web results found for \"{query}\" (via {provider}). The keyless endpoint can \
             rate-limit or omit results intermittently; rephrasing the query or retrying may help."
        );
    }
    let mut out = format!(
        "Web search results for \"{query}\" (via {provider}), {} result(s):\n",
        results.len()
    );
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!("\n{}. {}\n   {}\n", i + 1, r.title, r.url));
        if !r.snippet.is_empty() {
            out.push_str(&format!("   {}\n", r.snippet));
        }
    }
    out
}

/// Query DuckDuckGo's HTML endpoint and parse the result list. `endpoint` is a
/// parameter so tests can point it at a local mock server.
async fn duckduckgo_search(
    client: &reqwest::Client,
    endpoint: &str,
    query: &str,
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<Vec<SearchResult>> {
    // POST the query as a form field: the HTML endpoint is more reliable with a
    // POST body than a GET query string. `send_with_retries` rebuilds the
    // request per attempt and retries transient 429/5xx responses.
    let response = send_with_retries(
        "web_search (DuckDuckGo)",
        || client.post(endpoint).form(&[("q", query)]),
        cancel,
    )
    .await?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("DuckDuckGo returned HTTP {status}");
    }
    let body = response.text().await?;
    Ok(parse_duckduckgo_html(&body))
}

/// Shared HTTP client for web searches. Built once: each `reqwest::Client`
/// owns a connection pool and TLS configuration, so rebuilding it per call
/// would reload native certs and discard pooled connections. Cloning is cheap
/// (an `Arc` internally), and `&SEARCH_CLIENT` derefs to `&reqwest::Client`.
static SEARCH_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(USER_AGENT)
        .build()
        // Static, valid configuration: a failure here means the TLS backend
        // itself is unavailable, which would equally break the LLM client, so
        // failing fast is acceptable and consistent with the regex statics.
        .expect("web_search HTTP client builds from a static configuration")
});

static RESULT_ANCHOR: LazyLock<Regex> = LazyLock::new(|| {
    // Capture the opening-tag attributes (group 1) and inner HTML (group 2) of
    // each `result__a` link. Attribute order varies, so we match the class
    // anywhere in the tag and pull `href` out of the captured attributes.
    Regex::new(r#"(?s)<a\b([^>]*\bclass="[^"]*\bresult__a\b[^"]*"[^>]*)>(.*?)</a>"#).unwrap()
});
static SNIPPET_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<a\b[^>]*\bclass="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>"#).unwrap()
});
static HREF_ATTR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"href="([^"]*)""#).unwrap());
static UDDG_PARAM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"[?&]uddg=([^&"]+)"#).unwrap());
static HTML_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?s)<[^>]+>"#).unwrap());
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\s+"#).unwrap());
static NUM_ENTITY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"&#(\d+);"#).unwrap());
static HEX_ENTITY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"&#[xX]([0-9a-fA-F]+);"#).unwrap());

/// Parse DuckDuckGo HTML into results. Each result anchor is paired with the
/// first snippet block that falls between it and the next result anchor (by byte
/// offset), so a result with no snippet -- or a skipped ad anchor -- cannot
/// shift snippets onto the wrong results. Entries without a recoverable
/// destination URL (ads, internal `y.js` links) are skipped.
fn parse_duckduckgo_html(html: &str) -> Vec<SearchResult> {
    // (byte offset, cleaned snippet) for every snippet block, in document order.
    let snippets: Vec<(usize, String)> = SNIPPET_BLOCK
        .captures_iter(html)
        .map(|c| {
            let start = c.get(0).expect("match always has group 0").start();
            (start, clean_text(&c[1], SNIPPET_MAX_CHARS))
        })
        .collect();

    // (byte offset, attrs, inner HTML) for every result anchor, in document order.
    let anchors: Vec<(usize, &str, &str)> = RESULT_ANCHOR
        .captures_iter(html)
        .map(|c| {
            let whole = c.get(0).expect("match always has group 0");
            (
                whole.start(),
                c.get(1).expect("result__a attrs group").as_str(),
                c.get(2).expect("result__a inner group").as_str(),
            )
        })
        .collect();

    let mut results = Vec::new();
    for (idx, &(anchor_start, attrs, inner)) in anchors.iter().enumerate() {
        let Some(href) = HREF_ATTR.captures(attrs).map(|c| c[1].to_string()) else {
            continue;
        };
        let Some(url) = resolve_result_url(&href) else {
            continue;
        };
        let title = clean_text(inner, TITLE_MAX_CHARS);
        if title.is_empty() {
            continue;
        }
        // A result's snippet lives between its anchor and the next result
        // anchor; bound the lookup so a snippetless result claims no snippet
        // rather than the following result's.
        let next_anchor_start = anchors.get(idx + 1).map_or(usize::MAX, |a| a.0);
        let snippet = snippets
            .iter()
            .find(|(pos, _)| *pos > anchor_start && *pos < next_anchor_start)
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        results.push(SearchResult {
            title,
            url,
            snippet,
        });
    }
    results
}

/// Turn a DuckDuckGo result href into a real destination URL. Most are redirect
/// links carrying the target in a percent-encoded `uddg` param; some are direct.
fn resolve_result_url(href: &str) -> Option<String> {
    if let Some(caps) = UDDG_PARAM.captures(href) {
        let decoded = percent_encoding::percent_decode_str(&caps[1])
            .decode_utf8_lossy()
            .into_owned();
        if !decoded.is_empty() {
            return Some(decoded);
        }
    }
    if href.starts_with("http://") || href.starts_with("https://") {
        return Some(href.to_string());
    }
    if let Some(rest) = href.strip_prefix("//") {
        // Protocol-relative non-redirect link.
        if !rest.starts_with("duckduckgo.com") {
            return Some(format!("https://{rest}"));
        }
    }
    None
}

/// Strip HTML tags, decode common entities, collapse whitespace, and cap length.
fn clean_text(fragment: &str, max_chars: usize) -> String {
    let no_tags = HTML_TAG.replace_all(fragment, " ");
    let decoded = decode_entities(&no_tags);
    let collapsed = WHITESPACE.replace_all(decoded.trim(), " ");
    let mut text = collapsed.into_owned();
    if text.chars().count() > max_chars {
        text = text.chars().take(max_chars).collect::<String>();
        text.push('\u{2026}'); // ellipsis
    }
    text
}

/// Decode the handful of HTML entities that show up in DuckDuckGo titles and
/// snippets, including numeric and hex character references.
fn decode_entities(input: &str) -> String {
    let numeric = NUM_ENTITY.replace_all(input, |caps: &regex::Captures| {
        caps[1]
            .parse::<u32>()
            .ok()
            .and_then(char::from_u32)
            .map(String::from)
            .unwrap_or_default()
    });
    let hex = HEX_ENTITY.replace_all(&numeric, |caps: &regex::Captures| {
        u32::from_str_radix(&caps[1], 16)
            .ok()
            .and_then(char::from_u32)
            .map(String::from)
            .unwrap_or_default()
    });
    hex.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        // `&amp;` last so it can't double-decode an entity above.
        .replace("&amp;", "&")
}

/// Truncate a string to at most `max` bytes on a char boundary.
fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated]", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const FIXTURE: &str = r#"
    <div class="result results_links results_links_deep web-result">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftokio.rs%2F&amp;rut=abc">Tokio &#8211; An asynchronous Rust runtime</a>
      </h2>
      <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftokio.rs%2F">Tokio is an <b>async</b> runtime for the Rust programming &amp; systems language.</a>
    </div>
    <div class="result results_links results_links_deep web-result">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Fsmol%2F&amp;rut=def">smol &#x2014; a small async runtime</a>
      </h2>
      <a class="result__snippet" href="x">A &quot;small and fast&quot; async runtime.</a>
    </div>
    <div class="result result--ad">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/y.js?ad_provider=foo">Sponsored result</a>
      </h2>
    </div>
    "#;

    #[test]
    fn parses_titles_urls_and_snippets() {
        let results = parse_duckduckgo_html(FIXTURE);
        assert_eq!(
            results.len(),
            2,
            "ad result without a uddg target is skipped"
        );

        assert_eq!(results[0].title, "Tokio – An asynchronous Rust runtime");
        assert_eq!(results[0].url, "https://tokio.rs/");
        assert_eq!(
            results[0].snippet,
            "Tokio is an async runtime for the Rust programming & systems language."
        );

        assert_eq!(results[1].title, "smol — a small async runtime");
        assert_eq!(results[1].url, "https://docs.rs/smol/");
        assert_eq!(results[1].snippet, "A \"small and fast\" async runtime.");
    }

    #[test]
    fn snippet_pairing_survives_snippetless_result() {
        // Site 2 has no snippet between its anchor and Site 3's; the old
        // index-based pairing would have handed Site 2 the "Snippet three"
        // text and left Site 3 empty. Position-bounded pairing keeps each
        // snippet with its own result.
        let html = r#"
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsite1.com%2F">Site 1</a>
        <a class="result__snippet" href="x">Snippet one.</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsite2.com%2F">Site 2</a>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsite3.com%2F">Site 3</a>
        <a class="result__snippet" href="x">Snippet three.</a>
        "#;
        let results = parse_duckduckgo_html(html);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].url, "https://site1.com/");
        assert_eq!(results[0].snippet, "Snippet one.");
        assert_eq!(results[1].url, "https://site2.com/");
        assert_eq!(
            results[1].snippet, "",
            "a snippetless result must not borrow the next result's snippet"
        );
        assert_eq!(results[2].url, "https://site3.com/");
        assert_eq!(results[2].snippet, "Snippet three.");
    }

    #[tokio::test]
    async fn duckduckgo_search_hits_endpoint_and_parses() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .mount(&server)
            .await;

        let client = reqwest::Client::builder().build().unwrap();
        let endpoint = format!("{}/", server.uri());
        let results = duckduckgo_search(&client, &endpoint, "rust async", None)
            .await
            .expect("search should succeed against mock");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://tokio.rs/");
    }

    #[tokio::test]
    async fn search_surfaces_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(403).set_body_string("blocked"))
            .mount(&server)
            .await;

        let client = reqwest::Client::builder().build().unwrap();
        let endpoint = format!("{}/", server.uri());
        let err = duckduckgo_search(&client, &endpoint, "x", None)
            .await
            .expect_err("non-2xx should be an error");
        assert!(err.to_string().contains("403"), "got: {err}");
    }

    #[test]
    fn allowed_domains_whitelist() {
        let results = sample_results();
        let kept = apply_domain_filter(results, Some(&["docs.rs".to_string()]), None);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].url, "https://docs.rs/smol/");
    }

    #[test]
    fn blocked_domains_blacklist_includes_subdomains() {
        let results = sample_results();
        let kept = apply_domain_filter(results, None, Some(&["tokio.rs".to_string()]));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].url, "https://docs.rs/smol/");
    }

    #[test]
    fn host_extraction_strips_scheme_path_and_www() {
        assert_eq!(
            host_of("https://www.Example.com/a/b?x=1").as_deref(),
            Some("example.com")
        );
        assert_eq!(
            host_of("http://sub.example.com:8080/").as_deref(),
            Some("sub.example.com")
        );
        assert!(domain_matches("sub.example.com", "example.com"));
        assert!(!domain_matches("notexample.com", "example.com"));
    }

    #[test]
    fn empty_results_render_explanatory_note() {
        let text = format_results("foo", "DuckDuckGo", &[]);
        assert!(text.contains("No web results found for \"foo\""));
    }

    #[tokio::test]
    async fn rejects_empty_query() {
        let result = run_web_search("   ", None, None, None, None).await;
        assert!(matches!(result.status, ToolStatus::RequestError));
        assert!(result.output.contains("must not be empty"));
    }

    #[tokio::test]
    async fn rejects_both_domain_filters() {
        let result = run_web_search(
            "rust",
            None,
            Some(vec!["a.com".to_string()]),
            Some(vec!["b.com".to_string()]),
            None,
        )
        .await;
        assert!(matches!(result.status, ToolStatus::RequestError));
        assert!(result.output.contains("not both"));
    }

    fn sample_results() -> Vec<SearchResult> {
        vec![
            SearchResult {
                title: "Tokio".to_string(),
                url: "https://tokio.rs/".to_string(),
                snippet: String::new(),
            },
            SearchResult {
                title: "smol".to_string(),
                url: "https://docs.rs/smol/".to_string(),
                snippet: String::new(),
            },
        ]
    }
}
