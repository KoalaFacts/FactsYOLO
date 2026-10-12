# Model weights and dataset licensing

## Current inventory

No model weights, training datasets, sample photographs, or trained-model releases
are included in this foundation. There is no pretrained FactsYOLO model available
from this repository at this stage.

## Source license versus artifact license

The root Apache-2.0 license covers original source code and documentation. It is
not a default license grant for model weights or datasets. Every published model
artifact must have an explicit license and provenance record alongside it.

Apache-2.0 is the preferred license for publicly released, project-owned weights
when the project has the rights to make that grant. This preference does not
pre-approve an unreviewed model, dataset, training pipeline, or external checkpoint.

The project will not use or redistribute Ultralytics weights, including converted
or fine-tuned derivatives, as FactsYOLO model artifacts. General-purpose compute
frameworks are reviewed independently from detector implementations and weights.

## Required model release record

For each model artifact, maintain a model card with:

- Artifact name, version, checksum, architecture/version, source-code revision,
  training configuration, and intended task.
- Explicit artifact license, licensor/rightsholder information, notices, and
  redistribution terms, with a preserved copy of the applicable license.
- Initialization/base-checkpoint provenance, including whether training started
  from random initialization or from an authorized external checkpoint.
- Training-data and annotation provenance, usage permissions, applicable privacy
  or other rights restrictions, and the review outcome.
- Input and output contracts, preprocessing, labels, evaluation dataset,
  reproducibility information, measured accuracy, and limitations.

Random initialization is not a substitute for checking data or other rights.
Format conversion is not a licensing decision. Dataset terms do not automatically
become the software or weight license; review their actual conditions and the
intended training and distribution use.

## Release rule

Unclear provenance or missing artifact terms block publication of that artifact,
not the independent Apache-2.0 source-code project. Do not infer permission from
the ability to download a file or from the license of its surrounding repository.
