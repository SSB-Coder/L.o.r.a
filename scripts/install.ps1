# Install LORA on Windows.
#
#   irm https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.ps1 | iex
#   & ([scriptblock]::Create((irm .../install.ps1))) -Channel nightly
#   & ([scriptblock]::Create((irm .../install.ps1))) -Channel v0.2.1
#
# Installs to %LOCALAPPDATA%\Programs\LORA, adds it to your user PATH, and adds a Start menu shortcut.
param(
  [string]$Channel = "stable",
  [string]$Repo = $(if ($env:LORA_REPO) { $env:LORA_REPO } else { "SSB-Coder/L.o.r.a" }),
  [string]$Dest = "$env:LOCALAPPDATA\Programs\LORA"
)
$ErrorActionPreference = "Stop"
$api = "https://api.github.com/repos/$Repo"

$tag = switch -Regex ($Channel) {
  "^stable$" { (Invoke-RestMethod "$api/releases/latest").tag_name }
  "^nightly$" { (Invoke-RestMethod "$api/releases?per_page=30" | Where-Object prerelease | Select-Object -First 1).tag_name }
  "^v" { $Channel }
  default { throw "Channel must be stable, nightly, or a tag like v0.2.1 (got $Channel)." }
}
if (-not $tag) { throw "No $Channel release of LORA found." }

$version = $tag.TrimStart("v")
$name = "lora-$version-windows-x64"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
try {
  Write-Host "Downloading LORA $version…"
  $base = "https://github.com/$Repo/releases/download/$tag"
  Invoke-WebRequest "$base/$name.zip" -OutFile "$tmp\$name.zip"
  Invoke-WebRequest "$base/SHA256SUMS" -OutFile "$tmp\SHA256SUMS"

  $expected = (Get-Content "$tmp\SHA256SUMS" | Where-Object { $_ -match " $([regex]::Escape("$name.zip"))$" }) -split " " | Select-Object -First 1
  $actual = (Get-FileHash "$tmp\$name.zip" -Algorithm SHA256).Hash.ToLower()
  if (-not $expected -or $expected -ne $actual) { throw "Checksum mismatch for $name.zip; not installing." }

  Expand-Archive "$tmp\$name.zip" -DestinationPath $tmp
  New-Item -ItemType Directory -Force $Dest | Out-Null
  Copy-Item "$tmp\$name\*" $Dest -Recurse -Force
} finally {
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $Dest) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$Dest", "User")
  Write-Host "Added $Dest to your PATH (open a new terminal to use it)."
}

# Start menu shortcut; the console starts minimized, the app opens in its own window.
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\LORA.lnk")
$shortcut.TargetPath = "$Dest\lora.exe"
$shortcut.WorkingDirectory = $Dest
$shortcut.WindowStyle = 7
$shortcut.Save()

Write-Host "Installed $(& "$Dest\lora.exe" --version). Start Ollama, pull a model (ollama pull qwen3:8b), then open LORA from the Start menu."
