# L.O.R.A Network Setup & Isolation Guide

This document describes the network setup for running L.O.R.A (Layer-Oriented Research Assistant) with secure mobile access over a dedicated, isolated local network.

---

## 1. Network Architecture Overview

To provide remote access from a mobile phone to L.O.R.A on your PC without exposing your desktop environment to the general home network or internet, L.O.R.A implements an **isolated dual-listener architecture**:

- **Desktop Listener**: Bound strictly to `127.0.0.1:8765` (`origin="desktop"`). Has full access to desktop controls, local files, and deep research modes.
- **Mobile Listener**: Bound strictly to a dedicated router-side interface IP (default: `192.168.50.10:8765`, `origin="mobile"`). Access is restricted exclusively to Tier 0 read-only research. Local file access and OS controls are strictly stripped and denied.

> **CRITICAL SECURITY INVARIANT:** The mobile gateway must **NEVER** bind to `0.0.0.0` and must **NEVER** be reachable from the primary home Wi-Fi subnet.

---

## 2. Topologies

### Topology A (Recommended: Dedicated Isolated Secondary Router)

This is the standard topology providing hardware-level network isolation.

```
[ Internet ]
     │ (Home Wi-Fi)
[ PC ] ── Ethernet (LAN port) ── [ Dedicated Old Router ] ── (Wi-Fi: "LORA-NET") ── [ Phone ]
```

1. **Secondary Router Hardware Configuration**:
   - Perform a factory reset on the secondary router.
   - Leave the router's **WAN port unplugged** (disconnected from the internet).
   - Configure a strong custom admin password.
   - Disable WPS, UPnP, and remote web management.
   - Configure Wi-Fi: WPA2-AES or WPA3, SSID e.g. `LORA-NET`.
   - Configure DHCP server: Subnet `192.168.50.0/24` (Gateway: `192.168.50.1`).
2. **PC Configuration**:
   - Connect the PC's secondary Ethernet port (or USB-Ethernet adapter) to one of the **LAN** ports on the secondary router.
   - Assign static IP reservation: `192.168.50.10`.
   - Ensure the PC's primary Wi-Fi remains connected to your home router for outbound web searches.
   - Set the Windows Network Profile for the secondary Ethernet adapter to **Private**.
3. **Phone Configuration**:
   - Connect phone to `LORA-NET`.
   - Android/iOS will notice no internet on this Wi-Fi. Choose **"Stay connected"** / **"Keep Wi-Fi connection"**.
   - Disable "Smart Network Switch" / "Mobile Data Assist" to prevent the OS from dropping the LAN connection while browsing L.O.R.A.

### Topology B (Double-NAT Alternative)

If cabling permits, connect the secondary router's WAN port to the primary router's LAN port.
- Secondary router subnet: `192.168.50.0/24`.
- Primary router subnet: `192.168.1.0/24`.
- The phone receives internet via the secondary router while remaining isolated behind the secondary router's firewall from the rest of the household.

### Topology 0 (Zero-Hardware Fallback: Single Wi-Fi with Host Firewall)

If a second router is unavailable, both PC and phone connect to the primary home Wi-Fi.
- Assign a static DHCP reservation to the phone on the home router (e.g. `192.168.1.75`).
- The Windows Firewall rule must restrict inbound traffic on port 8765 strictly to the phone's IP:
  ```powershell
  New-NetFirewallRule -DisplayName "LORA Gateway (Phone Only)" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress 192.168.1.75/32 -Action Allow -Profile Private
  ```
- All other devices on the home network remain blocked from reaching port 8765.

---

## 3. Firewall Setup & Power Management

Run the provided PowerShell script as Administrator:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\net_setup.ps1 -InterfaceAlias "Ethernet 2" -Subnet "192.168.50.0/24"
```

The script:
1. Adds or updates the inbound firewall rule `LORA Gateway (Isolated)` scoped strictly to the specified network interface and subnet.
2. Configures Windows to keep the system awake (`SetThreadExecutionState`) while L.O.R.A is actively serving requests.

---

## 4. Verification with `lora net-check`

Verify the network setup before starting the gateway:

```powershell
python -m lora.tools.net_check
```

The tool audits:
- Mobile listener IP configuration (strictly isolated subnet, never `0.0.0.0`).
- Windows Firewall rule existence, interface binding, and remote subnet scoping.
- Non-overlap between home Wi-Fi and isolated router subnets.
- Unreachability from non-isolated interfaces.
