# V5 benchmark matrix and upstream files

The V5 freeze has 1,147,701 observation rows. The observation, design and editor-context tables in this deposit are byte-identical to the V5 files in the active benchmark core. Observation records carry source, row key, assay scale, label type and task eligibility. In this historical schema some task-view filenames use an earlier ordering. Use the task index in this package for the final T1–T7 definitions.

The `matrix-v5.zip` asset contains the full observation and design CSVs. The `matrix-source-inputs.zip` asset contains the 61 input files used by the matrix builder. [matrix_source_files.csv](matrix_source_files.csv) lists the exact file, version, original download link, SHA-256, observation count and expected path. The source-family totals are 464,898 BE-Hive rows, 283,336 BEDICT2 rows, and 399,467 CRISPRon-BE development rows. The last set includes multiple measurement endpoints. Its row count is distinct from the number of unique guides.

- BE-Hive processed efficiency and bystander data, Arbab et al., CC BY 4.0. [Efficiency v1](https://doi.org/10.6084/m9.figshare.10673816.v1) · [Bystander v1](https://doi.org/10.6084/m9.figshare.10678097.v1)
- BEDICT2 absolute-efficiency workbook, Kissling et al., Additional file 4, CC BY 4.0 as indicated in the source article. [Genome Biology article](https://doi.org/10.1186/s13059-025-03586-7)
- CRISPRon-BE development files, Sun et al., version 1.0. The four files aggregate data from SURRO-seq, Song, Arbab and Kissling. Their version and source are listed per file. [Source and dataset terms](https://github.com/RTH-tools/crispron-BE/tree/c5b28d4362b84c7e933e3715346ffbc1181fd6f2/data_for_model_development_V1.0) · [CRISPRon-BE COPYING](CRISPRon-BE_COPYING.txt) · [Article](https://doi.org/10.1038/s41467-025-65200-5)

The matrix harmonizes source rows into derived design, observation and editor-context identifiers, applies endpoint-specific conversions and records eligibility. The source files in the matrix-source asset retain their original bytes. [figshare_file_index.csv](figshare_file_index.csv) provides DOI versions, exact download URLs and hashes of locally available bystander and other Figshare inputs. [upstream_catalog.csv](upstream_catalog.csv) provides supplementary-file URLs and raw sequencing accessions. Large original public sequencing archives remain at their accession records.

Some frozen source columns retain historical path strings as provenance. The replay scripts resolve input files through the package index.

## Rebuild the matrix from the exact input files

After extracting `matrix-v5.zip` and `matrix-source-inputs.zip` and installing the analysis environment, run the command below from the repository root. It builds a fresh project tree from the 61 source files, recreates the harmonized tables and historical task views, and verifies the regenerated observation count.

```bash
python -B reproducibility/scripts/rebuild_matrix.py --work-dir benchmark-replay/matrix
```

The Marquart transfer task uses Supplementary Data 2 and 4 from the original publication. The two workbooks and the selected source-site and Excel-row identifiers are available through `additional-task-inputs.zip`. [Marquart et al.](https://doi.org/10.1038/s41467-021-25375-z)

The rebuild restores the archived source identifiers using the verified filename and worksheet mapping in [observation_source_ids_v5.csv](matrix/observation_source_ids_v5.csv). The source path field records the location used for the new build.
