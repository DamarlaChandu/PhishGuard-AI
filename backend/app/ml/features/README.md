# URL Feature Extraction

Turns a raw URL string into a fixed-length vector of integers for a
machine-learning model. Everything here is **static string analysis**: the
module never resolves DNS, never makes HTTP requests, and never loads or runs
page content. It uses only the Python standard library.

## Usage

```python
from app.ml.features import FEATURE_NAMES, InvalidURLError, extract_url_features

try:
    features = extract_url_features("http://192.168.0.1:8080/paypal/login.php?u=1&p=2")
except InvalidURLError as exc:
    ...  # reject the input (e.g. return HTTP 422)

features.to_dict()    # {"url_length": 48, "hostname_length": 11, ...}
features.to_vector()  # [48, 11, 4, 0, 0, 14, 4, 0, 1, 1, 0, 1, 17, 2, 0]
FEATURE_NAMES         # column names, same order as to_vector()
```

`URLFeatures` is an immutable dataclass. Its field order is the column order
of `to_vector()` and `FEATURE_NAMES`. **Only append new features at the end**,
so vectors built for an already-trained model keep their meaning.

## Modules

| File | Responsibility |
|---|---|
| `normalization.py` | Validates input and produces a `NormalizedURL`; raises `InvalidURLError` |
| `ip_address.py` | Detects IP-address hosts, including obfuscated IPv4 notations |
| `extractor.py` | `URLFeatures`, `FEATURE_NAMES` and the per-feature functions |
| `constants.py` | Keyword list, shortener domains, multi-label suffixes and limits |

## Pipeline

1. **Validate** (`normalize_url`). The input must be a `str`, non-empty after
   trimming, at most 8192 characters, and free of whitespace and control
   characters.
2. **Scheme.** Only `http` and `https` are accepted. Inputs such as
   `javascript:`, `data:`, `mailto:`, `file:` and `ftp:` are rejected. If no
   scheme is given (`example.com/login`, `//example.com`, `localhost:8080`),
   `http://` is assumed.
3. **Parse** with `urllib.parse.urlsplit`. A non-numeric or out-of-range port
   is rejected.
4. **Hostname.** The host is required and lowercased, and a trailing dot is
   removed. It must be an IP address or a valid DNS name: at most 253
   characters, no empty labels, labels of at most 63 characters, and only
   letters (Unicode allowed), digits, `-` and `_`.
5. **Rebuild** the URL with the scheme and host lowercased. The userinfo, path,
   query and fragment are kept exactly as given. An empty port (`host:`) is
   dropped.
6. **Extract** all features from the normalized URL, so that `EXAMPLE.com`,
   `example.com.` and `example.com` produce the same vector.

## Features

All values are non-negative integers. Indicators are `0` or `1`. "URL" means
the normalized URL string.

| # | Name | Definition | Why it matters |
|---|---|---|---|
| 1 | `url_length` | Characters in the URL | Phishing URLs are often long, to hide the real domain or pack in lure text |
| 2 | `hostname_length` | Characters in the hostname (no port, userinfo or IPv6 brackets) | Long, generated or look-alike hostnames are common in malicious sites |
| 3 | `num_dots` | Count of `.` in the URL | Many dots suggest deep subdomain nesting or deceptive paths like `paypal.com.evil.net` |
| 4 | `num_hyphens` | Count of `-` in the URL | Hyphenated brand imitations (`secure-paypal-login.com`) |
| 5 | `num_subdomains` | Labels to the left of the registrable domain; `0` for IP hosts | `www.example.com` → 1, `a.b.example.co.uk` → 2; heavy nesting is a common disguise |
| 6 | `num_digits` | Count of digit characters in the URL | Random tokens, IPs and generated domains raise digit counts |
| 7 | `num_special_chars` | Non-alphanumeric characters **except** `.` `-` `/` `:` | Captures `@ ? = & % _ ~ + !` and so on; obfuscation and encoded payloads raise it. `.` and `-` have their own features, and `/` and `:` appear in every URL |
| 8 | `has_https` | `1` if the scheme is `https` | Missing TLS is a weak risk signal (many phishing sites do use HTTPS) |
| 9 | `has_ip_address` | `1` if the host is IPv4, IPv6 or obfuscated IPv4 | Legitimate public sites rarely use raw IPs |
| 10 | `has_suspicious_keywords` | `1` if a lure keyword appears in the host, path, query or fragment | Credential-harvesting vocabulary such as "login", "verify" or "account" |
| 11 | `has_at_symbol` | `1` if `@` appears anywhere in the URL | In the authority, everything before `@` is ignored by browsers: `http://paypal.com@evil.example/` goes to `evil.example` |
| 12 | `has_port` | `1` if a port is written explicitly (even `:443`) | Non-standard ports are unusual for legitimate public sites |
| 13 | `path_length` | Characters in the path (excluding query and fragment) | Long paths often imitate brand directory structures |
| 14 | `num_parameters` | Non-empty `&`-separated segments of the query | Many parameters can carry tracking, redirect or victim-identifying data |
| 15 | `is_shortened_url` | `1` if the host is a known shortener or a subdomain of one | Shorteners hide the final destination |

### Feature details and limitations

- **Missing scheme.** A URL entered without a scheme is treated as `http`, so
  `has_https` is `0` and the added `http://` counts toward `url_length`. Send
  the full URL when it is known.
- **Subdomain counting** needs to know the public suffix (`.com`, `.co.uk`).
  To avoid a new dependency and any network fetch of the Public Suffix List,
  `constants.MULTI_LABEL_PUBLIC_SUFFIXES` holds a short list of common
  two-label suffixes. Any other suffix is treated as one label, so
  uncommon multi-label suffixes may overcount by one.
- **Obfuscated IPv4.** These forms are detected because browsers resolve them
  and phishing kits use them: integer (`http://3232235777/`), hex
  (`http://0xC0A80001/`), octal (`http://0300.0250.0.1/`) and shortened
  (`http://127.1/`). Detection is purely textual.
- **Suspicious keywords.** This is a substring match against
  `constants.SUSPICIOUS_KEYWORDS`, done after percent-decoding (so `%6Cogin`
  matches) and lowercasing. The scheme and userinfo are not searched. It is a
  weak indicator: legitimate sites also have `/login` pages. The model is
  meant to weigh it, not treat it as a verdict.
- **URL shorteners.** Matched against `constants.URL_SHORTENER_DOMAINS`, on the
  exact domain or a subdomain of it. Look-alikes such as `notbit.ly` or
  `bit.ly.example.com` do not match.
- **Query parameters.** `?a=1&&b=2&` counts 2, and `?flag` counts 1. `;` is
  not treated as a separator.
- **Internationalized domains** are kept as Unicode, not converted to Punycode.

## Safety

- No network access: tests patch `socket` to fail and confirm extraction
  still works.
- Only `http` and `https` are accepted, so script-bearing schemes such as
  `javascript:` and `data:` are rejected before parsing.
- Input length is bounded (`MAX_URL_LENGTH`), and every function runs in
  linear time on the input.

## Tests

```bat
python -m pytest tests/test_url_normalization.py tests/test_url_features.py
```
