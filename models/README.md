# Models (download separately)

The trained nnU-Net weights are **not committed to git** (they can be 10–100GB).

## Download link

- **Google Drive**: `https://drive.google.com/drive/folders/1fo5rLCdP0IDlba-jFBOXYmQuDVAMLvZ0?q=owner:me%20parent:1fo5rLCdP0IDlba-jFBOXYmQuDVAMLvZ0`

Download the zipped trained model from the link above and extract it.

## Expected folder on disk

After downloading, the folder must end up here:

`working/nnUNet_results/`

Example expected path:

`working/nnUNet_results/Dataset001_ECG/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth`

## Quick install (recommended)

From repo root:

```bash
bash scripts/download_models.sh "https://drive.google.com/drive/folders/1fo5rLCdP0IDlba-jFBOXYmQuDVAMLvZ0?q=owner:me%20parent:1fo5rLCdP0IDlba-jFBOXYmQuDVAMLvZ0"
```

Or manually:
1. Download the zip file from the Google Drive link
2. Extract it to the `working/` directory
3. Ensure the extracted folder structure matches the expected path above

