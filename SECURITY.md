# Security policy

## Supported scope

This repository contains educational and portfolio-oriented machine-learning implementations. It is not presented as hardened production infrastructure.

## Reporting

If you find a security issue, avoid public disclosure in an issue until the maintainer has reviewed it. Report the problem privately through the repository hosting platform's private reporting mechanism if available.

## What to report

Report issues such as:

- accidental inclusion of secrets
- unsafe deserialization or code execution paths
- dependency risks in development tooling
- examples or docs that expose private data

## Current known validation notes

Based on available validation evidence:

- secret/privacy scanning reported no secret findings and no privacy findings
- Bandit reported low-severity `assert` usage findings in runtime code paths

Those findings do not prove exploitability, but they should be reviewed before presenting the repository as polished public work.
