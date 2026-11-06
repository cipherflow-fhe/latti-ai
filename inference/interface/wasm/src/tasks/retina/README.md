# Retina task inputs

Run the task's existing segmentation tools offline and supply their image,
disk, lesion and clinical workbook files. `prepare.py` calls the existing
`gen_client_inputs_seg.py` functions unchanged with 512px and normalization.
It requires that script's Python dependencies (NumPy, Pillow, openpyxl, xlrd).

```sh
python prepare.py --reference /task/gen_csv/gen_client_inputs_seg.py \
  --samples samples.jsonl --xlsx risk_factor_5.xlsx \
  --task-client /current-task/client --output /prepared-retina
```

Each sample is a JSONL record containing `case_id`, `image`, `disk`, `lesion`
(file paths), and independent `label` (0=DN, 1=NDRD). The output contains
`input_0/`, `input_1/`, `input_2/`, `cases.json` and preprocessing provenance.
Pass the three directories directly to `encryptBatch({mode:'server', ...})`.
Deliver these prepared outputs with the authorized dataset; browser inference
does not run a replacement segmentation model. `postprocess.js` consumes the
actual decrypted BatchResult and independent JSONL labels.

The delivery task archive includes the original reference under
`tools/gen_client_inputs_seg.py`. Use that path for `--reference` when preparing
another authorized dataset. In the browser, `preprocessDirectory` from
`preprocess.js` validates every prepared CSV against the task shapes and returns
the three input directories for `encryptBatch`. It streams numeric blocks and
does not rerun segmentation or clinical feature extraction.
