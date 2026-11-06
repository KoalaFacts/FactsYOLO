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

## Original TD1 experiment

`experiments/cpu-detector/model/src/data.rs` generates original colored rectangles
and background noise. Its source is Apache-2.0; no external images, annotations,
models or pretrained weights are inputs. The initial weights are seeded random
numbers. Generated pixels/weights remain ignored local or ephemeral CI outputs,
and the workflow uploads only commands, scalar metrics and loss histories. This
is not a public dataset/weight release or a blanket model artifact license.
Any later publication requires its own reviewed provenance/license record.

## TD2 T1 dataset terms (stage C1, awaiting owner approval)

Status: recorded 2026-11-06 for the owner's decision. Nothing listed here has
been downloaded, and nothing may be until the owner approves this record. This
is a provenance record, not legal advice.

### Open Images V7 (T1 training; validation split for evaluation)

Terms, from the publisher's
[facts and figures page](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
(read 2026-11-06):

- "The annotations are licensed by Google LLC under CC BY 4.0 license."
- "The images are listed as having a CC BY 2.0 license."
- "we make no representations or warranties regarding the license status of
  each image and you should verify the license for each image yourself."

Per-image metadata: the
[download page](https://storage.googleapis.com/openimages/web/download_v7.html)
documents the image information columns `ImageID, Subset, OriginalURL,
OriginalLandingURL, License, AuthorProfileURL, Author, Title, OriginalSize,
OriginalMD5, Thumbnail300KURL, Rotation`.

Proposed use rules:

1. Download only annotation and metadata CSVs first. Select images whose
   `License` column is a CC BY 2.0 URL; drop every other or empty value.
2. Before training, check each selected image's `OriginalLandingURL` licence
   where it is still reachable, and drop images that are gone or changed. Record
   counts of kept and dropped images in the subset manifest (stage C2).
3. Attribution: the manifest keeps `ImageID`, `Author`, `AuthorProfileURL`,
   `OriginalLandingURL` and `License` for every training image. Any weights
   trained on the subset ship with that attribution list, the CC BY 2.0 and
   CC BY 4.0 notices, the "Annotations by Google LLC" credit and a statement of
   changes (class remapping, box filtering). Whether CC BY obligations attach
   to trained weights is unsettled; this record assumes they do.
4. Pixels stay local or in ephemeral CI storage; they are never committed,
   uploaded as artifacts or redistributed. Only manifests (IDs and fingerprints)
   and scalar metrics are kept.
5. Privacy: the images include identifiable faces and plates published by
   third parties. They are used only to train detection (where a face or plate
   is), never identity or personal attributes (spec non-goals). Images are not shown,
   cropped out or stored beyond training.

Category mapping: the boxable list is
[`oidv7-class-descriptions-boxable.csv`](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
(600 classes per the publisher). Confirmed rows used by the TD2 design:
`/m/0dzct` Human face, `/m/01jfm_` Vehicle registration plate, `/m/01g317`
Person, `/m/0k4j` Car, `/m/0199g` Bicycle, `/m/0bt9lr` Dog, `/m/01yrx` Cat. The
full class set (at most 120, spec section 6) is frozen with its MID mapping in
stage C2. Proposed: COCO categories without an Open Images boxable equivalent are dropped
from training and reported as untrained in section 9 results, not imported from
COCO.

### COCO 2017 val (T1 evaluation only)

Use: local evaluation of the 80 COCO categories; never used for training, never
redistributed. The terms page (`cocodataset.org/#termsofuse`) renders by script
and could not be read in this run. The terms as commonly published (annotations
under CC BY 4.0; the COCO Consortium does not own image copyright; images are
used under the Flickr Terms of Use) are **unverified here** and must be read
from the page before the owner approves evaluation use.

### Owner decision needed

Approve or reject, separately: (a) Open Images V7 training under rules 1–5;
(b) Open Images V7 validation for local evaluation; (c) COCO 2017 val for local
evaluation once its terms are verified. A rejection removes the source from
spec section 8; there is no workaround download.
