const PLATFORM_PACKAGES = Object.freeze({
  "android-arm64": Object.freeze({
    packageName: "@brokkai/anvil-android-arm64",
    cpu: ["arm64"],
    os: ["android"],
    target: "aarch64-linux-android",
  }),
  "darwin-universal": Object.freeze({
    packageName: "@brokkai/anvil-darwin-universal",
    cpu: ["arm64", "x64"],
    os: ["darwin"],
    target: "universal-apple-darwin",
  }),
  "linux-arm64": Object.freeze({
    packageName: "@brokkai/anvil-linux-arm64",
    cpu: ["arm64"],
    os: ["linux"],
    target: "aarch64-unknown-linux-gnu",
  }),
  "linux-x64": Object.freeze({
    packageName: "@brokkai/anvil-linux-x64",
    cpu: ["x64"],
    os: ["linux"],
    target: "x86_64-unknown-linux-gnu",
  }),
  "win32-x64": Object.freeze({
    packageName: "@brokkai/anvil-win32-x64",
    cpu: ["x64"],
    os: ["win32"],
    target: "x86_64-pc-windows-msvc",
  }),
});

export function platformPackages() {
  return PLATFORM_PACKAGES;
}

export function platformPackageFor(platform, arch) {
  if (platform === "darwin" && (arch === "arm64" || arch === "x64")) {
    return PLATFORM_PACKAGES["darwin-universal"];
  }

  const selected = PLATFORM_PACKAGES[`${platform}-${arch}`];
  if (selected) {
    return selected;
  }

  throw new Error(`@brokkai/anvil does not support ${platform}-${arch}`);
}

export function nativeBinaryName(platform) {
  return platform === "win32" ? "anvil.exe" : "anvil";
}
