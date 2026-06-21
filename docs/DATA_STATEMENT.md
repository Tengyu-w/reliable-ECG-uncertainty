# Data Statement

## Data Availability

The raw ECG dataset used by this project is not redistributed in this
repository.

The code expects a local file named `RHYTHMS.mat` with MATLAB cell arrays for
the three rhythm classes:

- `SR`
- `VT`
- `VF`

The file is intentionally excluded from Git because the redistribution status,
licence, consent, ethics approval, and access conditions must be confirmed
before any public release.

## What Is Included Publicly

This repository includes:

- source code for the research pipeline;
- documentation of the method and experiment order;
- aggregate summary tables;
- aggregate non-identifiable figures.

The public evidence is stored in:

```text
results_public/
```

## What Is Not Included

This repository does not include:

- raw ECG records;
- window-level ECG arrays;
- patient or record identifiers;
- model weights or checkpoints;
- train/test embeddings;
- window-level prediction files;
- private review examples;
- generated clinical-looking waveform case studies.

## Research Scope

This is a research prototype for uncertainty-aware ECG classification and
review routing. It is not a medical device, not a clinical validation study,
and not intended for diagnosis.

Any future public release should document:

1. the dataset source;
2. the access URL or DOI, if available;
3. the licence and redistribution conditions;
4. consent or ethics status, where applicable;
5. preprocessing requirements;
6. whether external validation has been performed.

