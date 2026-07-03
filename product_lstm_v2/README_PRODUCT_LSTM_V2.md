# Product LSTM v2

This folder contains a separate patch-and-train pipeline for the Aruba product documentation LSTM.

## What it does

1. Loads the existing Product LSTM dataset from `Data/product_docs_final`.
2. Loads repaired/Ollama product-doc rows from the repair-source folders.
3. Patches only `target_value` fields when there is a safe exact question match.
4. Writes patched JSONL splits into `outputs_product_lstm_v2`.
5. Trains a new BiLSTM classifier on the patched splits.
6. Evaluates the trained model and saves separate reports.

## Files

- `patch_existing_product_lstm.py`
- `train_product_lstm_v2.py`
- `evaluate_product_lstm_v2.py`
- `infer_product_lstm_v2.py`
- `product_lstm_v2_utils.py`

## Outputs

All generated artifacts go to `outputs_product_lstm_v2/`.

## Run order

```powershell
python .\product_lstm_v2\patch_existing_product_lstm.py
python .\product_lstm_v2\train_product_lstm_v2.py --device auto
python .\product_lstm_v2\evaluate_product_lstm_v2.py --device auto
```

## Notes

- The old Product LSTM pipeline is untouched.
- The release-note pipeline is untouched.
- Only `target_value` is patched.
- The BiLSTM model uses GPU automatically when available.
- Use `--workers` on the patch step and `--num_workers` on train/eval to speed up local processing.
