"""Detection of hostnames that are IP addresses, including obfuscated IPv4 forms."""

import ipaddress

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_IPV4_MAX = 0xFFFFFFFF


def parse_ip_hostname(hostname: str) -> IPAddress | None:
    """Return the IP address a hostname denotes, or None if it is a domain name.

    Besides standard dotted-quad IPv4 and IPv6, this recognises the legacy
    IPv4 notations that browsers still resolve and that phishing URLs use to
    hide an address: a single integer ("3232235777"), hexadecimal ("0xC0A80001"),
    octal parts ("0300.0250.0.1") and shortened forms ("127.1").
    Parsing is purely textual; the OS resolver is never called.
    """
    host = hostname.strip("[]").rstrip(".")
    if not host:
        return None
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    return _parse_legacy_ipv4(host)


def _parse_legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse inet_aton-style IPv4 notation."""
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values: list[int] = []
    for part in parts:
        value = _parse_ipv4_number(part)
        if value is None:
            return None
        values.append(value)

    # All parts but the last are single bytes; the last fills the remaining bytes.
    *leading, last = values
    if any(value > 0xFF for value in leading):
        return None
    remaining_bits = 8 * (4 - len(leading))
    if last >= 1 << remaining_bits:
        return None

    address = 0
    for value in leading:
        address = (address << 8) | value
    address = (address << remaining_bits) | last
    if address > _IPV4_MAX:
        return None
    return ipaddress.IPv4Address(address)


def _parse_ipv4_number(part: str) -> int | None:
    """Parse one decimal, octal (leading 0) or hexadecimal (0x) IPv4 component."""
    if not part:
        return None
    lowered = part.lower()
    try:
        if lowered.startswith("0x"):
            return int(lowered[2:], 16) if len(lowered) > 2 else 0
        if len(lowered) > 1 and lowered.startswith("0"):
            return int(lowered[1:], 8)
        if lowered.isdigit() and lowered.isascii():
            return int(lowered)
    except ValueError:
        return None
    return None
