# Data directory

This project does not redistribute battery datasets. Obtain data from the original provider and check its license and citation requirements before use.

## Suggested local organization

```text
data/
├── raw/          # Original files; excluded from Git
├── interim/      # Intermediate files; exclude sensitive or large files
└── processed/    # Tables prepared for analysis
```

## Expected table format

Most examples in this repository use one row per cycle or observation. Typical fields may include a battery identifier, cycle index, capacity or SOH target, RUL target, and impedance measurements. Exact names vary between datasets, so specify a mapping for each source.

Before training, check for duplicate rows, missing values, irregular cycle indices, unit consistency, and observations recorded after the chosen end-of-life point. State how RUL is calculated and how post-EOL records are handled.

Do not commit raw or processed data unless the license explicitly permits redistribution and no privacy or confidentiality restriction applies.
