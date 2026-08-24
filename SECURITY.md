# Security Policy

Please report vulnerabilities privately to the repository maintainers before opening a public issue.

High-priority issues include API-key exposure, SSRF, redirect validation bypass, prompt-injection persistence, unsafe file access, sensitive report disclosure, and SQLite injection or corruption.

EvidencePilot never intentionally logs credentials or reasoning content. `.env`, artifacts, databases, caches, and build output are ignored. URL validation blocks URL credentials, non-HTTP schemes, non-standard ports, local hostnames, private, loopback, link-local, and non-global IPs. Every redirect hop is revalidated, HTTPS downgrade redirects are rejected, environment proxies are disabled, and response media type and decoded size are bounded. PDF input is parsed as text only with `pypdf`; embedded actions or files are not executed.

Known boundary: DNS validation and connection establishment are separate operations, so deployments facing hostile DNS must add network-level egress controls or a resolver/connect transport that pins the validated IP. Application-layer checks are not a substitute for denying access to private and metadata networks at the deployment boundary.
