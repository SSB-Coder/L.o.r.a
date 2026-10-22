<#
.SYNOPSIS
    Configures Windows Firewall and Power Execution State for L.O.R.A Isolated Gateway.

.DESCRIPTION
    Creates or updates an inbound Windows Firewall rule scoped strictly to the isolated
    Ethernet interface and subnet for L.O.R.A mobile gateway. Also provides system sleep
    prevention using Win32 SetThreadExecutionState.

.PARAMETER InterfaceAlias
    The network adapter alias connected to the isolated router (e.g., 'Ethernet 2').

.PARAMETER LocalPort
    The TCP port L.O.R.A gateway listens on (default: 8765).

.PARAMETER RemoteSubnet
    The subnet allowed to connect (default: '192.168.50.0/24').

.PARAMETER RuleName
    Firewall rule display name (default: 'LORA Gateway (Isolated)').

.PARAMETER EnableKeepAwake
    Keep Windows awake while running by setting thread execution state.
#>

param (
    [string]$InterfaceAlias = "",
    [string]$LocalAddress = "192.168.50.10",
    [int]$LocalPort = 8765,
    [string]$RemoteSubnet = "192.168.50.0/24",
    [string]$RuleName = "LORA Gateway (Isolated)",
    [switch]$EnableKeepAwake
)

$ErrorActionPreference = "Stop"

Write-Host "=== L.O.R.A Gateway Network Setup ===" -ForegroundColor Cyan

# Check Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script requires Administrator privileges to configure Windows Firewall rules."
    Write-Warning "Please relaunch PowerShell as Administrator and run this script again."
    exit 1
}

Write-Host "[1/3] Checking network interface '$InterfaceAlias'..." -ForegroundColor Yellow
if (-not $InterfaceAlias) {
    $InterfaceAlias = (Get-NetIPAddress -IPAddress $LocalAddress -ErrorAction Stop).InterfaceAlias
    if (-not $InterfaceAlias) { throw "No adapter owns $LocalAddress" }
}
$adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction SilentlyContinue
if (-not $adapter) {
    Write-Warning "Adapter '$InterfaceAlias' not found. Available adapters:"
    Get-NetAdapter | Format-Table -Property Name, InterfaceDescription, Status, LinkSpeed
    throw "Cannot configure isolated firewall without the designated adapter"
} else {
    Write-Host "Found adapter '$InterfaceAlias' (Status: $($adapter.Status))" -ForegroundColor Green
}

Write-Host "[2/3] Configuring Windows Firewall rule '$RuleName'..." -ForegroundColor Yellow
try {
    # Remove existing rule if present
    $existingRule = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    if ($existingRule) {
        Write-Host "Removing previous rule '$RuleName'..." -ForegroundColor Gray
        Remove-NetFirewallRule -DisplayName $RuleName
    }

    # Create new rule scoped strictly to interface and subnet
    New-NetFirewallRule `
        -DisplayName $RuleName `
        -Description "Inbound allow rule for L.O.R.A mobile gateway isolated subnet" `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $LocalPort `
        -InterfaceAlias $InterfaceAlias `
        -RemoteAddress $RemoteSubnet `
        -Profile Private | Out-Null

    Write-Host "Successfully created firewall rule '$RuleName': Port $LocalPort on $InterfaceAlias ($RemoteSubnet)" -ForegroundColor Green
} catch {
    Write-Error "Failed to create firewall rule: $_"
    exit 1
}

Write-Host "[3/3] Power Management & Sleep Prevention..." -ForegroundColor Yellow
if ($EnableKeepAwake) {
    Write-Host "Activating Win32 SetThreadExecutionState (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)..." -ForegroundColor Cyan
    $signature = @"
    [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
"@
    $type = Add-Type -MemberDefinition $signature -Name "Win32Power" -Namespace "LORA" -PassThru
    $ES_CONTINUOUS = [uint32]0x80000000
    $ES_SYSTEM_REQUIRED = [uint32]0x00000001
    $result = $type::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    if ($result -ne 0) {
        Write-Host "System sleep prevention activated successfully." -ForegroundColor Green
    } else {
        Write-Warning "SetThreadExecutionState returned 0."
    }
} else {
    Write-Host "Keep-awake not requested. Pass -EnableKeepAwake to activate sleep prevention." -ForegroundColor Gray
}

Write-Host "`nNetwork isolation setup complete! Verify with: python -m lora.tools.net_check" -ForegroundColor Cyan
