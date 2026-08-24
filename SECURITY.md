# Security Policy

Please report vulnerabilities privately to the repository maintainers before opening a public issue.

High-priority issues include API-key exposure, SSRF, redirect validation bypass, prompt-injection persistence, unsafe file access, sensitive report disclosure, and SQLite injection or corruption.

EvidencePilot never intentionally logs credentials or reasoning content. `.env`, artifacts, databases, caches, and build output are ignored. URL validation blocks non-HTTP schemes, localhost, private, loopback, link-local, and non-global IPs, and repeats validation for every redirect hop.

Known boundary: DNS validation and connection establishment are separate operations, so deployments facing hostile DNS should add network-level egress controls or a resolver/connect transport that pins the validated IP.
