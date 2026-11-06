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
- "Note: while we tried to identify images that are licensed under a Creative
  Commons Attribution license, we make no representations or warranties
  regarding the license status of each image and you should verify the license
  for each image yourself."

CC BY 2.0 and CC BY 4.0 permit commercial use with attribution, which is what
spec section 8 requires of training data. Weight-release licensing stays a
separate decision (see "Source license versus artifact license" above).

Per-image metadata: the
[download page](https://storage.googleapis.com/openimages/web/download_v7.html)
documents the image information columns `ImageID, Subset, OriginalURL,
OriginalLandingURL, License, AuthorProfileURL, Author, Title, OriginalSize,
OriginalMD5, Thumbnail300KURL, Rotation`.

Privacy and consent: the publisher documents no consent or model release from
the people shown. CC BY covers the photographer's copyright only, not the
privacy, publicity or biometric rights of people depicted.

Proposed use rules (rules 1, 2, 4 and 5 also apply to the validation split):

1. Download only annotation and metadata CSVs first. Select images whose
   `License` column is a CC BY 2.0 URL; drop every other or empty value.
2. Before training, drop images whose `OriginalLandingURL` is unreachable or no
   longer shows CC BY 2.0. Record counts of kept and dropped images in the
   subset manifest (stage C2). This check is rate-limited and subject to the
   host site's terms, and it cannot detect an uploader who was not the
   rightsholder; that residual risk remains.
3. Attribution: the manifest keeps `ImageID`, `Title`, `Author`,
   `AuthorProfileURL`, `OriginalLandingURL` and `License` for every training
   image. Any weights trained on the subset ship with that attribution list,
   the CC BY 2.0 and CC BY 4.0 notices, "Open Images Dataset V7 annotations ©
   Google LLC, CC BY 4.0" with the licence and dataset links, and a statement of
   changes (class remapping, box filtering). Whether CC BY obligations attach to
   trained weights is unsettled; this record assumes they do.
4. Pixels stay on the local machine or in ephemeral CI storage; they are never
   committed, uploaded as artifacts or redistributed. Training on an owner-run
   notebook service (spec section 11) needs its own approval, with pixels
   deleted after the run. What is kept is the manifest (IDs, fingerprints and
   the attribution fields; author names are third-party personal data kept only
   for attribution) and scalar metrics.
5. Faces and plates are used only to train detection (where a face or plate
   is). These images are not used for T4 face recognition, whose data is
   chosen separately in stage J, and never for personal attributes (spec
   section 3 non-goals). Images are not shown, cropped out or stored beyond
   training.

Category mapping: the boxable list is
[`oidv7-class-descriptions-boxable.csv`](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
(the publisher says 600 classes; the file read 2026-11-06 has 601 rows).
Confirmed rows used by the TD2 design: `/m/0dzct` Human face, `/m/01jfm_`
Vehicle registration plate, `/m/01g317` Person, `/m/0k4j` Car, `/m/0199g`
Bicycle, `/m/0bt9lr` Dog, `/m/01yrx` Cat. The full class set (at most 120,
spec section 6) is frozen with its MID mapping in stage C2. Any of the 80 COCO
categories without an Open Images boxable equivalent is listed there, and how
section 9 scores it is settled in a spec PR, not here.

### COCO 2017 val (T1 evaluation only)

Use: local evaluation of the 80 COCO categories; never used for training, never
redistributed. The terms page (`cocodataset.org/#termsofuse`) renders by script
and could not be read in this run. The terms as commonly published (annotations
under CC BY 4.0; the COCO Consortium does not own image copyright; images are
used under the Flickr Terms of Use) are **unverified here**. Each image carries
its own Flickr licence in the annotation file, and some of those licences are
non-commercial (unverified here). Spec section 8 requires evaluation data to
allow local commercial evaluation, so decision (c) must settle this: either
evaluate only on images with commercial-use licences (changes the section 9
holdouts; needs a spec PR) or accept non-commercial images for local
evaluation. As with Open Images, no subject consent is documented.

### Owner decision needed

Approve or reject, separately: (a) Open Images V7 training under rules 1–5;
(b) Open Images V7 validation for local evaluation; (c) COCO 2017 val for local
evaluation once its terms and per-image licences are verified, including the
non-commercial question above. A rejection removes the source from spec section
8; there is no workaround download.
