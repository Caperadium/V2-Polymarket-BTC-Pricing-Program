# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Document the optional mm-monitor dashboard unit: `deploy/README.md` gains a "mm_monitor dashboard over an SSH tunnel" section (full loopback-bound systemd unit snippet + `ssh -L` usage; no template file added -- the unit is optional and created directly on the box), `DOCS/guides/market-maker-deployment.md` gains a matching short section, and the CLAUDE.md deploy-kit paragraph mentions it. Docs-only change; the unit itself is already live on the VPS (127.0.0.1:8502).
