use std::fs::{self, File, Permissions};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, ValueEnum};
use serde_json::{Map, Value, json};

const ENTRY_NAME: &str = "Anvil";

#[derive(Args, Debug)]
pub(crate) struct InstallArgs {
    /// Editor integration to configure.
    #[arg(value_enum)]
    target: InstallTarget,

    /// Neovim plugin to configure. If omitted, an interactive terminal prompts.
    #[arg(long, value_enum)]
    plugin: Option<NeovimPlugin>,

    /// Replace an existing Anvil entry or generated Neovim module.
    #[arg(long)]
    force: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InstallTarget {
    Zed,
    Intellij,
    Jetbrains,
    Nvim,
    Neovim,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum NeovimPlugin {
    Codecompanion,
    Avante,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EditorKind {
    Zed,
    Intellij,
}

#[derive(Debug, Eq, PartialEq)]
enum NvimPatchStatus {
    Patched,
    AlreadyConfigured,
    Missing,
    Unsupported(String),
}

pub(crate) fn install(args: &InstallArgs) -> Result<()> {
    if args.plugin.is_some() && !matches!(args.target, InstallTarget::Nvim | InstallTarget::Neovim)
    {
        bail!("--plugin is only valid for the nvim/neovim target");
    }

    let binary = installed_binary_path()?;
    match args.target {
        InstallTarget::Zed => {
            let path = default_zed_settings_path()?;
            configure_editor(EditorKind::Zed, &path, &binary, args.force)?;
            println!("Configured Zed ACP integration in {}", path.display());
        }
        InstallTarget::Intellij | InstallTarget::Jetbrains => {
            let path = home_dir()?.join(".jetbrains").join("acp.json");
            configure_editor(EditorKind::Intellij, &path, &binary, args.force)?;
            println!("Configured JetBrains ACP integration in {}", path.display());
        }
        InstallTarget::Nvim | InstallTarget::Neovim => install_neovim(args, &binary)?,
    }
    println!("Anvil executable: {}", binary.display());
    Ok(())
}

fn installed_binary_path() -> Result<PathBuf> {
    let path = std::env::current_exe().context("cannot determine the running Anvil executable")?;
    path.canonicalize()
        .with_context(|| format!("cannot resolve Anvil executable {}", path.display()))
}

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().context("cannot determine the user's home directory")
}

fn default_zed_settings_path() -> Result<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            let appdata = PathBuf::from(appdata);
            if appdata.is_absolute() {
                return Ok(appdata.join("Zed").join("settings.json"));
            }
        }
        Ok(home_dir()?
            .join("AppData")
            .join("Roaming")
            .join("Zed")
            .join("settings.json"))
    }
    #[cfg(not(windows))]
    {
        Ok(home_dir()?
            .join(".config")
            .join("zed")
            .join("settings.json"))
    }
}

fn default_nvim_config_dir() -> Result<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(local_appdata) = std::env::var_os("LOCALAPPDATA") {
            let local_appdata = PathBuf::from(local_appdata);
            if local_appdata.is_absolute() {
                return Ok(local_appdata.join("nvim"));
            }
        }
    }
    Ok(home_dir()?.join(".config").join("nvim"))
}

fn configure_editor(
    kind: EditorKind,
    settings_path: &Path,
    binary: &Path,
    force: bool,
) -> Result<()> {
    let (prefix, mut root) = read_json_or_jsonc(settings_path, kind == EditorKind::Zed)?;
    let root_obj = root.as_object_mut().with_context(|| {
        format!(
            "expected a JSON object in editor settings {}",
            settings_path.display()
        )
    })?;

    if kind == EditorKind::Intellij {
        ensure_object(root_obj, "default_mcp_settings")?;
    }
    let agent_servers = ensure_object(root_obj, "agent_servers")?;
    if agent_servers.contains_key(ENTRY_NAME) && !force {
        bail!(
            "agent_servers['{ENTRY_NAME}'] already exists in {}; use --force to overwrite it",
            settings_path.display()
        );
    }

    let command = binary
        .to_str()
        .context("the Anvil executable path is not valid UTF-8")?;
    let mut entry = Map::new();
    if kind == EditorKind::Zed {
        entry.insert("type".into(), json!("custom"));
        entry.insert(
            "favorite_config_option_values".into(),
            json!({
                "reasoning": ["medium"],
                "mode": ["LUTZ"]
            }),
        );
    }
    entry.insert("command".into(), json!(command));
    entry.insert("args".into(), json!([]));
    entry.insert("env".into(), json!({}));
    agent_servers.insert(ENTRY_NAME.into(), Value::Object(entry));

    let serialized = serde_json::to_string_pretty(&root)? + "\n";
    let output = if prefix.is_empty() {
        serialized
    } else {
        format!("{prefix}{serialized}")
    };
    atomic_write(settings_path, output.as_bytes())
}

fn ensure_object<'a>(
    root: &'a mut Map<String, Value>,
    key: &str,
) -> Result<&'a mut Map<String, Value>> {
    root.entry(key.to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| anyhow!("expected '{key}' to be a JSON object"))
}

fn read_json_or_jsonc(path: &Path, preserve_prefix: bool) -> Result<(String, Value)> {
    if !path.exists() {
        return Ok((String::new(), json!({})));
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading editor settings {}", path.display()))?;
    if text.trim().is_empty() {
        return Ok((String::new(), json!({})));
    }

    let (prefix, json_text) = if preserve_prefix {
        split_leading_json_prefix(&text)
    } else {
        ("", text.as_str())
    };
    let cleaned = remove_trailing_commas(&strip_jsonc_comments(json_text));
    let parsed = serde_json::from_str(&cleaned)
        .with_context(|| format!("parsing JSON/JSONC in {}", path.display()))?;
    let prefix = if prefix.is_empty() {
        String::new()
    } else if prefix.ends_with('\n') {
        prefix.to_owned()
    } else {
        format!("{prefix}\n")
    };
    Ok((prefix, parsed))
}

fn split_leading_json_prefix(text: &str) -> (&str, &str) {
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut line_comment = false;
    let mut block_comment = false;
    while index < bytes.len() {
        let current = bytes[index];
        let next = bytes.get(index + 1).copied();
        if line_comment {
            if current == b'\n' {
                line_comment = false;
            }
            index += 1;
            continue;
        }
        if block_comment {
            if current == b'*' && next == Some(b'/') {
                block_comment = false;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if current == b'/' && next == Some(b'/') {
            line_comment = true;
            index += 2;
        } else if current == b'/' && next == Some(b'*') {
            block_comment = true;
            index += 2;
        } else if current == b'{' || current == b'[' {
            return text.split_at(index);
        } else {
            index += 1;
        }
    }
    (text, "")
}

fn strip_jsonc_comments(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment = false;
    while index < chars.len() {
        let current = chars[index];
        let next = chars.get(index + 1).copied();
        if line_comment {
            if current == '\n' {
                line_comment = false;
                output.push('\n');
            }
            index += 1;
            continue;
        }
        if block_comment {
            if current == '*' && next == Some('/') {
                block_comment = false;
                index += 2;
            } else {
                if current == '\n' {
                    output.push('\n');
                }
                index += 1;
            }
            continue;
        }
        if in_string {
            output.push(current);
            if escaped {
                escaped = false;
            } else if current == '\\' {
                escaped = true;
            } else if current == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if current == '"' {
            in_string = true;
            output.push(current);
            index += 1;
        } else if current == '/' && next == Some('/') {
            line_comment = true;
            index += 2;
        } else if current == '/' && next == Some('*') {
            block_comment = true;
            index += 2;
        } else {
            output.push(current);
            index += 1;
        }
    }
    output
}

fn remove_trailing_commas(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < chars.len() {
        let current = chars[index];
        if in_string {
            output.push(current);
            if escaped {
                escaped = false;
            } else if current == '\\' {
                escaped = true;
            } else if current == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if current == '"' {
            in_string = true;
            output.push(current);
            index += 1;
            continue;
        }
        if current == ',' {
            let mut lookahead = index + 1;
            while chars
                .get(lookahead)
                .is_some_and(|character| character.is_whitespace())
            {
                lookahead += 1;
            }
            if matches!(chars.get(lookahead), Some('}') | Some(']')) {
                index += 1;
                continue;
            }
        }
        output.push(current);
        index += 1;
    }
    output
}

fn install_neovim(args: &InstallArgs, binary: &Path) -> Result<()> {
    let plugin = match args.plugin {
        Some(plugin) => plugin,
        None if io::stdin().is_terminal() => prompt_neovim_plugin()?,
        None => NeovimPlugin::Codecompanion,
    };
    let config_dir = default_nvim_config_dir()?;
    let (module_path, contents, plugin_repo, module_name) = match plugin {
        NeovimPlugin::Codecompanion => (
            config_dir.join("lua/anvil/anvil_codecompanion.lua"),
            codecompanion_template(binary)?,
            "olimorris/codecompanion.nvim",
            "anvil.anvil_codecompanion",
        ),
        NeovimPlugin::Avante => (
            config_dir.join("lua/anvil/anvil_avante.lua"),
            avante_template(binary)?,
            "yetone/avante.nvim",
            "anvil.anvil_avante",
        ),
    };
    if module_path.exists() && !args.force {
        bail!(
            "{} already exists; use --force to overwrite it",
            module_path.display()
        );
    }
    atomic_write(&module_path, contents.as_bytes())?;
    println!(
        "Configured Neovim {} ACP integration in {}",
        match plugin {
            NeovimPlugin::Codecompanion => "CodeCompanion",
            NeovimPlugin::Avante => "Avante",
        },
        module_path.display()
    );

    let init_path = config_dir.join("init.lua");
    match wire_nvim_plugin_setup(&init_path, plugin_repo, module_name)? {
        NvimPatchStatus::Patched => println!("Updated {} to load Anvil.", init_path.display()),
        NvimPatchStatus::AlreadyConfigured => {
            println!("{} already loads Anvil.", init_path.display())
        }
        NvimPatchStatus::Missing => println!(
            "Load the generated module from your Neovim plugin setup; {} does not exist.",
            init_path.display()
        ),
        NvimPatchStatus::Unsupported(detail) => {
            println!("Load the generated module from your Neovim plugin setup; {detail}.")
        }
    }
    Ok(())
}

fn prompt_neovim_plugin() -> Result<NeovimPlugin> {
    println!("Choose a Neovim plugin integration:");
    println!("1) CodeCompanion (ACP adapter)");
    println!("2) Avante (ACP provider)");
    print!("Selection [1/2] (default: 1): ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    match input.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "codecompanion" => Ok(NeovimPlugin::Codecompanion),
        "2" | "avante" => Ok(NeovimPlugin::Avante),
        other => bail!("invalid Neovim plugin selection '{other}'"),
    }
}

fn lua_string(path: &Path) -> Result<String> {
    let path = path
        .to_str()
        .context("the Anvil executable path is not valid UTF-8")?;
    Ok(format!(
        "\"{}\"",
        path.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}

fn codecompanion_template(binary: &Path) -> Result<String> {
    let command = lua_string(binary)?;
    Ok(format!(
        r#"-- Generated by `anvil install nvim`
-- Requires codecompanion.nvim: https://github.com/olimorris/codecompanion.nvim
-- Load with: require("codecompanion").setup(require("anvil.anvil_codecompanion"))

local acp_helpers = require("codecompanion.adapters.acp.helpers")

return {{
  interactions = {{
    chat = {{ adapter = "anvil" }},
  }},
  adapters = {{
    acp = {{
      anvil = function()
        return {{
          name = "anvil",
          formatted_name = "Anvil",
          type = "acp",
          roles = {{ llm = "assistant", user = "user" }},
          commands = {{
            default = {{ {command} }},
          }},
          defaults = {{ mcpServers = {{}}, timeout = 20000 }},
          env = {{}},
          parameters = {{
            protocolVersion = 1,
            clientCapabilities = {{
              fs = {{ readTextFile = true, writeTextFile = true }},
            }},
            clientInfo = {{ name = "CodeCompanion.nvim", version = "1.0.0" }},
          }},
          handlers = {{
            setup = function(self) return true end,
            form_messages = function(self, messages, capabilities)
              return acp_helpers.form_messages(self, messages, capabilities)
            end,
            on_exit = function(self, code) end,
          }},
        }}
      end,
    }},
  }},
}}
"#
    ))
}

fn avante_template(binary: &Path) -> Result<String> {
    let command = lua_string(binary)?;
    Ok(format!(
        r#"-- Generated by `anvil install neovim --plugin avante`
-- Requires avante.nvim: https://github.com/yetone/avante.nvim
-- Load with:
-- require("avante").setup(vim.tbl_deep_extend("force", require("anvil.anvil_avante"), {{}}))

return {{
  provider = "anvil",
  acp_providers = {{
    anvil = {{
      command = {command},
      args = {{}},
      env = {{
        HOME = os.getenv("HOME"),
        PATH = os.getenv("PATH"),
      }},
    }},
  }},
}}
"#
    ))
}

fn wire_nvim_plugin_setup(
    init_path: &Path,
    plugin_repo: &str,
    module_name: &str,
) -> Result<NvimPatchStatus> {
    if !init_path.exists() {
        return Ok(NvimPatchStatus::Missing);
    }
    let text = fs::read_to_string(init_path)
        .with_context(|| format!("reading {}", init_path.display()))?;
    if text.contains(module_name) {
        return Ok(NvimPatchStatus::AlreadyConfigured);
    }
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let Some(plugin_line) = lines.iter().position(|line| line.contains(plugin_repo)) else {
        return Ok(NvimPatchStatus::Unsupported(format!(
            "could not find plugin block for {plugin_repo}"
        )));
    };
    let Some(start) = (0..=plugin_line)
        .rev()
        .find(|index| lines[*index].trim() == "{")
    else {
        return Ok(NvimPatchStatus::Unsupported(format!(
            "could not find the start of the {plugin_repo} plugin block"
        )));
    };
    let Some(end) = (plugin_line + 1..lines.len()).find(|index| lines[*index].trim() == "},")
    else {
        return Ok(NvimPatchStatus::Unsupported(format!(
            "could not find the end of the {plugin_repo} plugin block"
        )));
    };
    let Some(opts_index) = (start..=end).find(|index| lines[*index].trim() == "opts = {},") else {
        return Ok(NvimPatchStatus::Unsupported(format!(
            "found {plugin_repo}, but not a simple `opts = {{}}` block that can be patched safely"
        )));
    };
    let indent: String = lines[opts_index]
        .chars()
        .take_while(|character| character.is_whitespace())
        .collect();
    lines.splice(
        opts_index..=opts_index,
        [
            format!("{indent}opts = function()"),
            format!("{indent}  return require(\"{module_name}\")"),
            format!("{indent}end,"),
        ],
    );
    atomic_write(init_path, format!("{}\n", lines.join("\n")).as_bytes())?;
    Ok(NvimPatchStatus::Patched)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating parent directory {}", parent.display()))?;
    let existing_permissions: Option<Permissions> = fs::metadata(path)
        .ok()
        .map(|metadata| metadata.permissions());
    let temp_path = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("anvil"),
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> Result<()> {
        let mut temp = File::create(&temp_path)
            .with_context(|| format!("creating temporary file {}", temp_path.display()))?;
        temp.write_all(contents)
            .with_context(|| format!("writing temporary file {}", temp_path.display()))?;
        temp.sync_all()
            .with_context(|| format!("syncing temporary file {}", temp_path.display()))?;
        if let Some(permissions) = existing_permissions {
            fs::set_permissions(&temp_path, permissions)
                .with_context(|| format!("preserving permissions for {}", path.display()))?;
        }
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(path)
                .with_context(|| format!("replacing existing file {}", path.display()))?;
        }
        fs::rename(&temp_path, path)
            .with_context(|| format!("installing generated configuration {}", path.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn zed_install_merges_jsonc_and_preserves_prefix() {
        let temp = tempdir().unwrap();
        let settings = temp.path().join("settings.json");
        fs::write(
            &settings,
            "// Zed settings\n{\n  \"theme\": \"One Dark\",\n  \"agent_servers\": {},\n}\n",
        )
        .unwrap();
        configure_editor(EditorKind::Zed, &settings, Path::new("/opt/anvil"), false).unwrap();
        let updated = fs::read_to_string(&settings).unwrap();
        assert!(updated.starts_with("// Zed settings\n"));
        let (_, json_text) = split_leading_json_prefix(&updated);
        let root: Value = serde_json::from_str(json_text).unwrap();
        assert_eq!(root["theme"], "One Dark");
        assert_eq!(root["agent_servers"]["Anvil"]["command"], "/opt/anvil");
        assert_eq!(root["agent_servers"]["Anvil"]["args"], json!([]));
        assert_eq!(root["agent_servers"]["Anvil"]["type"], "custom");
    }

    #[test]
    fn editor_install_refuses_existing_entry_without_force() {
        let temp = tempdir().unwrap();
        let settings = temp.path().join("settings.json");
        fs::write(
            &settings,
            r#"{"agent_servers":{"Anvil":{"command":"old"}}}"#,
        )
        .unwrap();
        let error = configure_editor(EditorKind::Zed, &settings, Path::new("/opt/anvil"), false)
            .unwrap_err();
        assert!(error.to_string().contains("--force"));
        configure_editor(EditorKind::Zed, &settings, Path::new("/new/anvil"), true).unwrap();
        let root: Value = serde_json::from_str(&fs::read_to_string(settings).unwrap()).unwrap();
        assert_eq!(root["agent_servers"]["Anvil"]["command"], "/new/anvil");
    }

    #[test]
    fn intellij_install_preserves_other_settings_and_adds_mcp_object() {
        let temp = tempdir().unwrap();
        let settings = temp.path().join("acp.json");
        fs::write(&settings, r#"{"other":true}"#).unwrap();
        configure_editor(
            EditorKind::Intellij,
            &settings,
            Path::new("/opt/anvil"),
            false,
        )
        .unwrap();
        let root: Value = serde_json::from_str(&fs::read_to_string(settings).unwrap()).unwrap();
        assert_eq!(root["other"], true);
        assert_eq!(root["default_mcp_settings"], json!({}));
        assert_eq!(root["agent_servers"]["Anvil"]["command"], "/opt/anvil");
        assert!(root["agent_servers"]["Anvil"].get("type").is_none());
    }

    #[test]
    fn generated_neovim_modules_launch_anvil_directly() {
        let codecompanion = codecompanion_template(Path::new("/path with spaces/anvil")).unwrap();
        assert!(codecompanion.contains(r#"default = { "/path with spaces/anvil" }"#));
        assert!(!codecompanion.contains("uvx"));
        assert!(!codecompanion.contains("\"brokk\""));

        let avante = avante_template(Path::new(r#"C:\Program Files\Anvil\anvil.exe"#)).unwrap();
        assert!(avante.contains(r#"command = "C:\\Program Files\\Anvil\\anvil.exe""#));
        assert!(avante.contains("args = {}"));
    }

    #[test]
    #[cfg(unix)]
    fn nvim_patch_is_narrow_and_preserves_permissions() {
        let temp = tempdir().unwrap();
        let init = temp.path().join("init.lua");
        fs::write(
            &init,
            "require(\"lazy\").setup({\n  {\n    \"olimorris/codecompanion.nvim\",\n    opts = {},\n  },\n})\n",
        )
        .unwrap();
        fs::set_permissions(&init, Permissions::from_mode(0o640)).unwrap();
        assert_eq!(
            wire_nvim_plugin_setup(
                &init,
                "olimorris/codecompanion.nvim",
                "anvil.anvil_codecompanion"
            )
            .unwrap(),
            NvimPatchStatus::Patched
        );
        let updated = fs::read_to_string(&init).unwrap();
        assert!(updated.contains("return require(\"anvil.anvil_codecompanion\")"));
        assert_eq!(
            fs::metadata(init).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[test]
    #[cfg(unix)]
    fn atomic_write_preserves_existing_permissions() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("config");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, Permissions::from_mode(0o640)).unwrap();
        atomic_write(&path, b"new").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }
}
